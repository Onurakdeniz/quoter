pub mod analytics;
pub mod graph;
pub mod pathfinder;
pub mod quoting;
pub mod simulation;

use crate::data_management::component_tracker::ComponentTracker;
use crate::data_management::cache::{QuoteCache, QuoteCacheKey, CachedQuote, PathCacheKey, CachedPaths};
use crate::config::AppConfig;
use graph::TokenGraph;
use pathfinder::Pathfinder;
use quoting::{PriceQuote, SinglePathQuote};

use tycho_simulation::tycho_common::Bytes;
use num_traits::cast::ToPrimitive;
use rust_decimal::Decimal;
use std::sync::Arc;
use std::sync::RwLock;
use std::collections::{HashSet, HashMap};
use std::time::Instant;
use itertools::Itertools;
use petgraph::prelude::{NodeIndex, EdgeIndex};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use std::str::FromStr;

use futures::future::join_all; // For collecting async tasks

// Default WETH address (mainnet)
const DEFAULT_ETH_ADDRESS_STR: &str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
// Default probe depth: 1 unit (e.g., 1 ETH or 1 USDC, assuming 18 decimals for default numeraire)
const DEFAULT_PROBE_DEPTH: u128 = 1_000_000_000_000_000_000;
const DEFAULT_AVG_GAS_UNITS_PER_SWAP: u64 = 150_000;
const DEFAULT_NATIVE_DECIMALS: u32 = 18; // Changed to u32

fn default_eth_address_bytes() -> Bytes {
    Bytes::from_str(DEFAULT_ETH_ADDRESS_STR).expect("Failed to parse default ETH address")
}

/// The main price engine struct.
pub struct PriceEngine {
    pub tracker: ComponentTracker,
    pub graph: Arc<RwLock<TokenGraph>>,
    pub pathfinder: Pathfinder,
    pub cache: Arc<RwLock<QuoteCache>>, // shared LRU cache
    pub gas_price_wei: Arc<RwLock<u128>>,
    pub max_hops: usize,
    pub numeraire_token: Option<Bytes>,
    pub probe_depth: Option<u128>,
    pub native_token_address: Bytes, // Added
    pub avg_gas_units_per_swap: u64, // Added, no longer Option
}

impl PriceEngine {
    pub fn new(tracker: ComponentTracker, graph: Arc<RwLock<TokenGraph>>) -> Self {
        let pathfinder = Pathfinder::new(graph.clone());
        let cache = Arc::new(RwLock::new(QuoteCache::new()));
        let default_gas_price_wei = 30_000_000_000u128; // 30 Gwei

        Self {
            tracker,
            graph,
            pathfinder,
            cache,
            gas_price_wei: Arc::new(RwLock::new(default_gas_price_wei)),
            max_hops: 3,
            numeraire_token: None,
            probe_depth: None,
            native_token_address: default_eth_address_bytes(), // Default native token
            avg_gas_units_per_swap: DEFAULT_AVG_GAS_UNITS_PER_SWAP, // Default gas units
        }
    }

    /// Create a PriceEngine reusing an existing shared cache with custom gas price and hop limit.
    // This constructor might need updating if native_token_address and avg_gas_units_per_swap are critical.
    // For now, it uses defaults like ::new(). Consider passing them as args if customization is needed here.
    pub fn with_cache(tracker: ComponentTracker, graph: Arc<RwLock<TokenGraph>>, cache: Arc<RwLock<QuoteCache>>, gas_price_wei: Arc<RwLock<u128>>, max_hops: usize) -> Self {
        let pathfinder = Pathfinder::new(graph.clone());
        Self { 
            tracker, 
            graph, 
            pathfinder, 
            cache, 
            gas_price_wei, 
            max_hops, 
            numeraire_token: None, 
            probe_depth: None,
            native_token_address: default_eth_address_bytes(), // Default native token
            avg_gas_units_per_swap: DEFAULT_AVG_GAS_UNITS_PER_SWAP, // Default gas units
        }
    }
    
    pub fn from_config(tracker: ComponentTracker, graph: Arc<RwLock<TokenGraph>>, cache: Arc<RwLock<QuoteCache>>, config: &AppConfig) -> Self {
        let pathfinder = Pathfinder::new(graph.clone());
        let max_hops = config.max_hops.unwrap_or(3);
        let numeraire = config.numeraire_token.clone();
        let probe_depth = config.probe_depth;

        let new_gas_price_wei = config.gas_price_gwei
            .map(|gwei| u128::from(gwei) * 1_000_000_000) // Convert Gwei to Wei
            .unwrap_or(30_000_000_000u128); // Default to 30 Gwei (30*10^9 Wei)

        let native_token_address = config.native_token_address.clone().unwrap_or_else(default_eth_address_bytes);
        let avg_gas_units_per_swap = config.avg_gas_units_per_swap.unwrap_or(DEFAULT_AVG_GAS_UNITS_PER_SWAP);

        Self {
            tracker,
            graph,
            pathfinder,
            cache,
            gas_price_wei: Arc::new(RwLock::new(new_gas_price_wei)),
            max_hops,
            numeraire_token: numeraire,
            probe_depth,
            native_token_address,
            avg_gas_units_per_swap,
        }
    }

    /// Compute a price quote for a given input token, output token, and amount, using the best path only.
    // This should also become async if quote_multi is async
    pub async fn quote(&self, token_in: &Bytes, token_out: &Bytes, amount_in: u128, block: Option<u64>) -> PriceQuote {
        let current_block = block.unwrap_or(0);
        let cache_key = QuoteCacheKey { sell_token: token_in.clone(), buy_token: token_out.clone(), amount: amount_in, block: current_block };
        
        if let Some(cached) = self.cache.write().unwrap().get(&cache_key).cloned() {
            return PriceQuote {
                amount_out: Some(cached.amount_out),
                route: cached.route,
                price_impact_bps: cached.price_impact_bps,
                mid_price: cached.mid_price,
                slippage_bps: cached.slippage_bps,
                fee_bps: cached.fee_bps,
                gas_estimate: cached.gas_estimate,
                path_details: Vec::new(),
                gross_amount_out: cached.gross_amount_out,
                spread_bps: cached.spread_bps,
                depth_metrics: None,
                cache_block: Some(cached.block),
            };
        }

        let pq = self.quote_multi(token_in, token_out, amount_in, 1, block).await; // Added .await

        if let Some(amt) = pq.amount_out {
            let cached_value = CachedQuote {
                amount_out: amt,
                route: pq.route.clone(),
                route_pools: pq.path_details.get(0).map(|d| d.pools.clone()).unwrap_or_default(),
                mid_price: pq.mid_price,
                slippage_bps: pq.slippage_bps,
                spread_bps: pq.spread_bps,
                block: current_block, 
                gross_amount_out: pq.gross_amount_out,
                fee_bps: pq.fee_bps,
                gas_estimate: pq.gas_estimate,
                price_impact_bps: pq.price_impact_bps,
            };
            self.cache.write().unwrap().insert(cache_key, cached_value);
        }
        pq
    }

    /// Compute a price quote for a given input token, output token, and amount, simulating up to K paths.
    pub async fn quote_multi(&self, token_in: &Bytes, token_out: &Bytes, amount_in: u128, k: usize, block: Option<u64>) -> PriceQuote { // Made async
        let current_block = block.unwrap_or(0);
        
        let all_paths_nodes: Vec<Vec<NodeIndex>> = {
            let graph_r = self.graph.read().unwrap();
            let path_key = PathCacheKey {
                sell_token: token_in.clone(),
                buy_token: token_out.clone(),
                block: 0, 
                k: self.max_hops, 
            };
            let mut cache_w = self.cache.write().unwrap();
            if let Some(cached) = cache_w.get_paths(&path_key).cloned() {
                let mut reconstructed: Vec<Vec<NodeIndex>> = Vec::new();
                for addr_path in cached.paths {
                    let mut node_path = Vec::with_capacity(addr_path.len());
                    let mut valid = true;
                    for addr in addr_path {
                        if let Some(&idx) = graph_r.token_indices.get(&addr) {
                            if graph_r.graph.node_weight(idx).is_some() {
                                node_path.push(idx);
                            } else {
                                valid = false; break;
                            }
                        } else {
                            valid = false; break;
                        }
                    }
                    if valid { reconstructed.push(node_path); }
                }
                reconstructed
            } else {
                let enumerated = self.pathfinder.enumerate_paths(token_in, token_out, self.max_hops);
                let addr_paths: Vec<Vec<Bytes>> = enumerated.iter().map(|p| {
                    p.iter().filter_map(|idx| graph_r.graph.node_weight(*idx).map(|n| n.address.clone())).collect()
                }).collect();
                if !enumerated.is_empty() {
                    let cached_value = CachedPaths { paths: addr_paths, block: 0, timestamp: Instant::now() };
                    cache_w.insert_paths(path_key, cached_value);
                }
                enumerated
            }
        };

        let mut seen_edge_seqs: HashSet<Vec<EdgeIndex>> = HashSet::new();
        let mut tasks: Vec<(Vec<NodeIndex>, Vec<EdgeIndex>)> = Vec::new();

        for node_path in &all_paths_nodes {
            if node_path.len() < 2 { continue; }
            let mut all_hop_edges: Vec<Vec<EdgeIndex>> = Vec::new();
            let mut path_possible = true;
            for win in node_path.windows(2) {
                let from_idx = win[0];
                let to_idx = win[1];
                let graph_r = self.graph.read().unwrap();
                let edges: Vec<_> = graph_r.graph.edges_connecting(from_idx, to_idx).map(|e| e.id()).collect();
                if edges.is_empty() {
                    path_possible = false; break;
                }
                all_hop_edges.push(edges);
            }
            if !path_possible { continue; }
            for edge_seq in all_hop_edges.into_iter().multi_cartesian_product() {
                if !seen_edge_seqs.insert(edge_seq.clone()) { continue; }
                tasks.push((node_path.clone(), edge_seq));
            }
        }

        // Asynchronous evaluation of paths
        let mut path_futures = Vec::new();
        for (node_path, edge_seq) in tasks {
            // Clone `self` or necessary Arcs for each async task if `quote_single_path_with_edges` needs `&self`
            // For now, assuming `quote_single_path_with_edges` can be called if `self` is Arc-ed or if it takes Arcs.
            // Let's make it take `self: Arc<Self>` or pass necessary Arcs directly.
            // Simpler: `self.quote_single_path_with_edges` takes `&self`, so we need to ensure `self` outlives the futures.
            // This is fine if `quote_multi` is awaited before `self` is dropped.
            let future = self.quote_single_path_with_edges(
                token_in.clone(), 
                token_out.clone(), 
                amount_in, 
                node_path, // node_path is Vec<NodeIndex>, can be moved
                edge_seq, // edge_seq is Vec<EdgeIndex>, can be moved
                Some(current_block)
            );
            path_futures.push(future);
        }
        
        let evaluated_paths_results: Vec<SinglePathQuote> = join_all(path_futures).await
            .into_iter()
            .filter(|pq| pq.amount_out.is_some()) // Filter out paths that resulted in None amount_out
            .collect();
        
        // Sort by net amount_out descending
        let mut sorted_paths = evaluated_paths_results; // Use the collected results
        let mut sorted_paths = evaluated_paths;
        sorted_paths.sort_by(|a, b| b.amount_out.cmp(&a.amount_out));
        sorted_paths.truncate(k);

        if sorted_paths.is_empty() {
            return PriceQuote { amount_out: None, route: vec![], price_impact_bps: None, mid_price: None, slippage_bps: None, fee_bps: None, gas_estimate: None, path_details: vec![], gross_amount_out: None, spread_bps: None, depth_metrics: None, cache_block: None };
        }

        let best_path_details = sorted_paths[0].clone();
        
        // Calculate mid_price based on the new two-way swap logic using engine's numeraire
        let mid_price_of_token_in_vs_token_out = if *token_in == *token_out { // Deref Bytes for comparison
            Some(Decimal::ONE)
        } else {
            let price_in_vs_n_fut = self.get_token_price(token_in, Some(current_block));
            let price_out_vs_n_fut = self.get_token_price(token_out, Some(current_block));
            let (price_in_vs_n, price_out_vs_n) = tokio::join!(price_in_vs_n_fut, price_out_vs_n_fut); // Await futures concurrently

            if let (Some(p_in), Some(p_out)) = (price_in_vs_n, price_out_vs_n) {
                if !p_out.is_zero() {
                    Some(p_in / p_out)
                } else {
                    None
                }
            } else {
                None
            }
        };

        let mut depth_metrics_map = HashMap::new();

        // Calculate depth metrics for the best path if numeraire is set
        if self.numeraire_token.is_some() && self.probe_depth.is_some() {
             if let Some(mid_price_val) = mid_price_of_token_in_vs_token_out {
                for slippage_target_str in ["0.1%", "0.5%", "1.0%", "2.0%", "5.0%"] {
                    let slippage_target_f64 = slippage_target_str.trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                    if let Some(depth_amount) = analytics::find_depth_for_slippage(
                        &self.tracker,
                        &self.graph.read().unwrap(),
                        token_in, 
                        token_out, 
                        mid_price_val, 
                        &best_path_details.node_path, 
                        &best_path_details.edge_seq, 
                        slippage_target_f64, 
                        Some(current_block)
                    ) {
                        depth_metrics_map.insert(slippage_target_str.to_string(), depth_amount);
                    }
                }
            }
        }
        
        PriceQuote {
            amount_out: best_path_details.amount_out,
            route: best_path_details.route.clone(),
            price_impact_bps: best_path_details.price_impact_bps,
            mid_price: mid_price_of_token_in_vs_token_out,
            slippage_bps: best_path_details.slippage_bps,
            fee_bps: best_path_details.fee_bps,
            gas_estimate: best_path_details.gas_estimate,
            path_details: sorted_paths,
            gross_amount_out: best_path_details.gross_amount_out,
            spread_bps: best_path_details.spread_bps,
            depth_metrics: if depth_metrics_map.is_empty() { None } else { Some(depth_metrics_map) },
            cache_block: None, // This is a fresh quote
        }
    }
    
    // Most other methods from the original PriceEngine will be moved to quoting.rs, simulation.rs, or analytics.rs
    // For example, quote_single_path_with_edges, simulate_path_gross, calculate_*, etc.
    // We need to make them public in their respective modules and call them here, or make them part of PriceEngine if they use `self` extensively.

    // This is a method that would now live in quoting.rs or be called from there.
    // For now, it stays here to ensure `quote_multi` compiles.
    // Made async because it calls async self.get_token_price
    pub async fn quote_single_path_with_edges(&self, token_in: Bytes, token_out: Bytes, amount_in: u128, path: Vec<NodeIndex>, edge_seq: Vec<EdgeIndex>, block: Option<u64>) -> SinglePathQuote {
        // Note: token_in, token_out, path, edge_seq are now owned params due to async context.
        let graph_r = self.graph.read().unwrap(); // This lock needs to be managed carefully in async.
                                                 // If held across .await, it can cause deadlocks or contention.
                                                 // For now, assume it's short-lived.
        let gross_amount_out = simulation::simulate_path_gross(&self.tracker, &graph_r, amount_in, &path, &edge_seq, block);

        if gross_amount_out.is_none() {
            return quoting::invalid_path_quote(path, edge_seq, amount_in);
        }
        let gross_amount_out_val = gross_amount_out.unwrap();
        let current_block_num = block.unwrap_or(0); // For get_token_price calls

        // --- Gas Calculation Start ---
        let gas_estimate_units = edge_seq.len() as u64 * self.avg_gas_units_per_swap;
        
        let native_token_decimals = self.tracker.all_tokens.read().unwrap()
            .get(&self.native_token_address)
            .map_or(DEFAULT_NATIVE_DECIMALS, |token_meta| token_meta.decimals as u32);

        let current_gas_price_wei_val = *self.gas_price_wei.read().unwrap();
        let gas_cost_in_native_token_units = gas_estimate_units as u128 * current_gas_price_wei_val;
        
        let gas_cost_native_decimal = Decimal::from_i128_with_scale(gas_cost_in_native_token_units as i128, native_token_decimals);
        
        let mut gas_cost_in_token_out_decimal = Decimal::ZERO;

        if token_out == self.native_token_address { // Deref not needed due to owned Bytes
            gas_cost_in_token_out_decimal = gas_cost_native_decimal;
        } else {
            // Await the get_token_price calls
            let price_native_vs_numeraire_fut = self.get_token_price(&self.native_token_address, Some(current_block_num));
            let price_out_vs_numeraire_fut = self.get_token_price(&token_out, Some(current_block_num)); // Use owned token_out
            let (price_native_vs_numeraire, price_out_vs_numeraire) = tokio::join!(price_native_vs_numeraire_fut, price_out_vs_numeraire_fut);


            if let (Some(p_native), Some(p_out)) = (price_native_vs_numeraire, price_out_vs_numeraire) {
                if !p_out.is_zero() {
                    let price_of_native_in_terms_of_token_out = p_native / p_out;
                    gas_cost_in_token_out_decimal = gas_cost_native_decimal * price_of_native_in_terms_of_token_out;
                } else {
                    // Price of output token is zero, cannot convert gas cost. Log or handle as error.
                    // For now, gas_cost_in_token_out_decimal remains zero.
                }
            } else {
                // Could not get prices for conversion. Log or handle.
                // For now, gas_cost_in_token_out_decimal remains zero.
            }
        }
        // --- Gas Calculation End ---

        let amount_in_dec = Decimal::from(amount_in);
        let gross_amount_out_dec = Decimal::from(gross_amount_out_val);
        
        let token_in_model = self.tracker.all_tokens.read().unwrap().get(&token_in).cloned(); // Use owned token_in
        let token_out_model = self.tracker.all_tokens.read().unwrap().get(&token_out).cloned(); // Use owned token_out

        if token_in_model.is_none() || token_out_model.is_none() {
             return quoting::invalid_path_quote(&path, &edge_seq, amount_in); // Pass slices
        }
        let _t_in = token_in_model.unwrap(); 
        let _t_out = token_out_model.unwrap(); // No longer t_out, use _t_out for consistency

        // Calculate total protocol fee in BPS and as a Decimal amount
        let mut total_protocol_fee_bps = Decimal::ZERO;
        let mut path_protocol_fees_bps: Vec<(String, Decimal)> = Vec::new();

        for edge_idx in edge_seq.iter() {
            if let Some(pool_edge) = graph_r.graph.edge_weight(*edge_idx) { // Now pool_edge is &PoolEdge
                // Assuming all edges in the path are PoolEdges for protocol fee calculation.
                // If VirtualEdges or other types exist and need different handling for fees,
                // this logic would need to be adapted, perhaps by inspecting pool_edge.protocol or similar.
                let fee_percent = pool_edge.fee; // This is Option<f64> from PoolEdge
                // Convert f64 to Decimal. fee_percent is a direct factor (e.g., 0.003 for 0.3%)
                let fee_decimal_factor = Decimal::from_f64_retain(fee_percent.unwrap_or(0.0)).unwrap_or_default();
                let fee_bps_for_edge = fee_decimal_factor * Decimal::new(10000, 0); // Convert factor to BPS
                total_protocol_fee_bps += fee_bps_for_edge;
                path_protocol_fees_bps.push((pool_edge.pool_id.clone(), fee_bps_for_edge));
            }
        }
        
        let protocol_fee_amount_dec = if !total_protocol_fee_bps.is_zero() && !gross_amount_out_dec.is_zero() {
            gross_amount_out_dec * (total_protocol_fee_bps / Decimal::new(10000, 0))
        } else {
            Decimal::ZERO
        };

        // Use the new gas_cost_in_token_out_decimal
        let net_amount_out_dec = gross_amount_out_dec - protocol_fee_amount_dec - gas_cost_in_token_out_decimal;
        let net_amount_out_dec = net_amount_out_dec.max(Decimal::ZERO); // Ensure not negative

        let net_amount_out = net_amount_out_dec.to_u128().unwrap_or(0);

        // Mid price for this specific path simulation (effective price before fees and gas)
        let mid_price_approx_gross = if !amount_in_dec.is_zero() && !gross_amount_out_dec.is_zero() {
            Some(gross_amount_out_dec / amount_in_dec)
        } else {
            None
        };

        let price_impact = mid_price_approx_gross.and_then(|mp| analytics::calculate_price_impact_bps(amount_in_dec, gross_amount_out_dec, mp));
        // Slippage and spread should be calculated on net_amount_out_dec if they are meant to reflect the final price after all costs.
        // Or, if they are about the pool's performance before external costs, gross_amount_out_dec might be used for some.
        // For now, let's assume slippage and spread are about the final net price.
        let mid_price_approx_net = if !amount_in_dec.is_zero() && !net_amount_out_dec.is_zero() {
            Some(net_amount_out_dec / amount_in_dec)
        } else {
            None
        };

        let slippage = mid_price_approx_net.and_then(|mp| analytics::calculate_slippage_bps(amount_in_dec, net_amount_out_dec, mp));
        let spread = mid_price_approx_net.and_then(|mp| analytics::calculate_spread_bps(amount_in_dec, net_amount_out_dec, mp));
        
        // This was previously `fee_bps_final`. If this represents the total protocol fee, it's `total_protocol_fee_bps`
        let fee_bps_for_quote = Some(total_protocol_fee_bps);

        SinglePathQuote {
            amount_out: Some(net_amount_out),
            route: path.iter().map(|idx| graph_r.graph.node_weight(*idx).unwrap().address.clone()).collect(), // path is Vec
            mid_price: mid_price_approx_net, 
            slippage_bps: slippage,
            fee_bps: fee_bps_for_quote, 
            gas_estimate: Some(gas_estimate_units), // Use the calculated gas units
            gross_amount_out: Some(gross_amount_out_val),
            spread_bps: spread,
            price_impact_bps: price_impact, 
            pools: edge_seq.iter().map(|e_idx| graph_r.graph.edge_weight(*e_idx).unwrap().pool_id.clone()).collect(), // edge_seq is Vec
            input_amount: Some(amount_in),
            node_path: path, // path is now owned Vec
            edge_seq: edge_seq, // edge_seq is now owned Vec
            gas_cost_native: Some(gas_cost_native_decimal), 
            gas_cost_in_token_out: Some(gas_cost_in_token_out_decimal), 
        }
    }

    pub fn update_graph_from_tracker_state(&self) {
        let mut graph_w = self.graph.write().unwrap();
        // The tracker is part of PriceEngine, so we can access its components directly.
        // No need to pass them as arguments.
        graph_w.update_from_components_with_tracker(
            &self.tracker.all_pools.read().unwrap(),
            &self.tracker.pool_states.read().unwrap(),
            &self.tracker.all_tokens.read().unwrap(),
        );
    }

    // Renamed from the sketch to avoid potential naming conflicts if it were public
    async fn calculate_token_price_in_numeraire_impl( // Made async
        &self,
        target_token: &Bytes,
        numeraire_token: &Bytes,
        probe_amount_of_numeraire: u128,
        path_nodes_numeraire_to_target: &[NodeIndex],
        path_edges_numeraire_to_target: &[EdgeIndex],
        path_nodes_target_to_numeraire: &[NodeIndex],
        path_edges_target_to_numeraire: &[EdgeIndex],
        block: Option<u64>,
    ) -> Option<Decimal> {
        if numeraire_token == target_token {
            return Some(Decimal::ONE);
        }
        if probe_amount_of_numeraire == 0 {
            return None;
        }

        let graph_r = self.graph.read().unwrap();

        let numeraire_decimals_opt = graph_r.token_indices.get(numeraire_token)
            .and_then(|idx| graph_r.graph.node_weight(*idx))
            .map(|node| node.decimals);
        let target_decimals_opt = graph_r.token_indices.get(target_token)
            .and_then(|idx| graph_r.graph.node_weight(*idx))
            .map(|node| node.decimals);

        if numeraire_decimals_opt.is_none() || target_decimals_opt.is_none() {
            return None; // Cannot proceed without decimal info
        }
        let numeraire_decimals = numeraire_decimals_opt.unwrap();
        let target_decimals = target_decimals_opt.unwrap();

        // 1. Forward Swap: numeraire_token -> target_token
        let amount_target_out_option = simulation::simulate_path_gross(
            &self.tracker,
            &graph_r,
            probe_amount_of_numeraire,
            path_nodes_numeraire_to_target,
            path_edges_numeraire_to_target,
            block,
        );

        if amount_target_out_option.is_none() || amount_target_out_option.unwrap() == 0 {
            return None;
        }
        let amount_target_out = amount_target_out_option.unwrap();

        // 2. Backward Swap: amount_target_out (of target_token) -> numeraire_token
        let amount_numeraire_returned_option = simulation::simulate_path_gross(
            &self.tracker,
            &graph_r,
            amount_target_out, // input for backward swap
            path_nodes_target_to_numeraire,
            path_edges_target_to_numeraire,
            block,
        );

        if amount_numeraire_returned_option.is_none() || amount_numeraire_returned_option.unwrap() == 0 {
            return None;
        }
        let amount_numeraire_returned = amount_numeraire_returned_option.unwrap();

        // Use Decimal::from_i128_with_scale for correct decimal handling
        let probe_amount_numeraire_dec = Decimal::from_i128_with_scale(probe_amount_of_numeraire as i128, numeraire_decimals as u32);
        let amount_target_out_dec = Decimal::from_i128_with_scale(amount_target_out as i128, target_decimals as u32);
        let amount_numeraire_returned_dec = Decimal::from_i128_with_scale(amount_numeraire_returned as i128, numeraire_decimals as u32);

        if probe_amount_numeraire_dec.is_zero() || amount_target_out_dec.is_zero() || amount_numeraire_returned_dec.is_zero() {
            return None; // Avoid division by zero
        }

        // Price from forward swap (target_token / numeraire_token)
        let price_forward = amount_target_out_dec / probe_amount_numeraire_dec;

        // Price from backward swap, normalized to (target_token / numeraire_token)
        // Raw backward price: amount_numeraire_returned_dec / amount_target_out_dec (numeraire_token / target_token)
        // Normalized: amount_target_out_dec / amount_numeraire_returned_dec (target_token / numeraire_token)
        let price_backward_normalized = amount_target_out_dec / amount_numeraire_returned_dec;

        let mean_price = (price_forward + price_backward_normalized) / Decimal::new(2, 0);
        Some(mean_price)
    }

    /// Calculates the mid-price of a token in terms of the engine's configured numeraire (or ETH default).
    /// Uses a two-way swap with a configured probe depth.
    pub async fn get_token_price(&self, token: &Bytes, block: Option<u64>) -> Option<Decimal> { // Made async
        let engine_numeraire = self.numeraire_token.clone().unwrap_or_else(default_eth_address_bytes);
        let probe_amount = self.probe_depth.unwrap_or(DEFAULT_PROBE_DEPTH);

        if *token == engine_numeraire {
            return Some(Decimal::ONE);
        }

        // Pathfinder methods will acquire their own locks internally.
        let path_n_to_t_nodes_opt = self.pathfinder.best_path(&engine_numeraire, token);
        let path_t_to_n_nodes_opt = self.pathfinder.best_path(token, &engine_numeraire);

        if path_n_to_t_nodes_opt.is_none() || path_t_to_n_nodes_opt.is_none() {
            return None;
        }
        let path_n_to_t_nodes = path_n_to_t_nodes_opt.unwrap();
        let path_t_to_n_nodes = path_t_to_n_nodes_opt.unwrap();
        
        let graph_r = self.graph.read().unwrap();
        let path_n_to_t_edges_opt = graph_r.derive_edges_for_node_path(&path_n_to_t_nodes);
        let path_t_to_n_edges_opt = graph_r.derive_edges_for_node_path(&path_t_to_n_nodes);

        if path_n_to_t_edges_opt.is_none() || path_t_to_n_edges_opt.is_none() {
            return None;
        }
        let path_n_to_t_edges = path_n_to_t_edges_opt.unwrap();
        let path_t_to_n_edges = path_t_to_n_edges_opt.unwrap();

        self.calculate_token_price_in_numeraire_impl( // Await the async call
            token,              // target_token for the function signature
            &engine_numeraire,  // numeraire_token for the function signature
            probe_amount,       // probe_amount_of_numeraire
            &path_n_to_t_nodes[..], 
            &path_n_to_t_edges[..], 
            &path_t_to_n_nodes[..], 
            &path_t_to_n_edges[..], 
            block,
        ).await // Added await
    }

     // Other methods like precompute_all_quotes, log_all_paths, list_unit_price_vs_eth, etc.
     // will be added here or moved to submodules as appropriate.
} 