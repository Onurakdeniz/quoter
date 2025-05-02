//! Price computation, simulation, slippage, gas, quote logic.

use crate::component_tracker::ComponentTracker;
use crate::graph::TokenGraph;
use crate::pathfinder::Pathfinder;
use tycho_simulation::tycho_common::Bytes;
use num_traits::cast::ToPrimitive;
use num_traits::FromPrimitive;
use rust_decimal::Decimal;
use std::sync::Arc;
use std::sync::RwLock;
use crate::cache::{QuoteCache, QuoteCacheKey, CachedQuote};
use std::collections::{HashSet, HashMap};
use crate::config::AppConfig;
use crate::cache::{PathCacheKey, CachedPaths};
use std::time::Instant;
use itertools::Itertools;
use petgraph::visit::EdgeRef;
use petgraph::prelude::{NodeIndex, EdgeIndex};
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;
use hex;
use rayon::prelude::*;
use num_traits::{Zero, One};
use std::path::Path;

/// Result of a price quote computation.
pub struct PriceQuote {
    pub amount_out: Option<u128>,
    pub route: Vec<Bytes>,
    /// price impact over the whole route, in bps (10 000 bps = 1 %)
    pub price_impact_bps: Option<Decimal>,
    pub mid_price: Option<Decimal>,
    pub slippage_bps: Option<Decimal>,
    pub fee_bps: Option<Decimal>,
    pub gas_estimate: Option<u64>,
    pub path_details: Vec<SinglePathQuote>, // For multi-path
    pub gross_amount_out: Option<u128>,
    pub spread_bps: Option<Decimal>,
    /// Depth metrics: Input amount required to cause X% slippage. Key: "0.5%", "1.0%", etc. Value: input amount (u128)
    pub depth_metrics: Option<HashMap<String, u128>>,
    /// If this quote was returned from the cache, which block it was cached at
    pub cache_block: Option<u64>,
}

/// Per-path quote details for multi-path evaluation
#[derive(Clone)]
pub struct SinglePathQuote {
    pub amount_out: Option<u128>,
    pub route: Vec<Bytes>,
    pub mid_price: Option<Decimal>,
    pub slippage_bps: Option<Decimal>,
    pub fee_bps: Option<Decimal>,
    pub gas_estimate: Option<u64>,
    pub gross_amount_out: Option<u128>,
    pub spread_bps: Option<Decimal>,
    /// price impact for this path, in bps
    pub price_impact_bps: Option<Decimal>,
    pub pools: Vec<String>,
    pub input_amount: Option<u128>,
    pub node_path: Vec<NodeIndex>,
    pub edge_seq: Vec<EdgeIndex>,
}

/// The main price engine struct.
pub struct PriceEngine<'a> {
    pub tracker: ComponentTracker,
    pub graph: &'a TokenGraph,
    pub pathfinder: Pathfinder<'a>,
    pub cache: Arc<RwLock<QuoteCache>>, // shared LRU cache
    pub gas_price_wei: Arc<RwLock<u128>>,
    pub max_hops: usize,
    pub numeraire_token: Option<Bytes>,
    pub probe_depth: Option<u128>,
}

impl<'a> PriceEngine<'a> {
    pub fn new(tracker: ComponentTracker, graph: &'a TokenGraph) -> Self {
        let pathfinder = Pathfinder::new(graph);
        let cache = Arc::new(RwLock::new(QuoteCache::new()));
        // Register callback to invalidate cache entries when pool states change
        let cache_clone = cache.clone();
        tracker.register_callback(move |update| {
            let mut cache = cache_clone.write().unwrap();
            for pool_id in update.new_pairs.keys() {
                cache.invalidate_pool(pool_id);
            }
            for pool_id in update.removed_pairs.keys() {
                cache.invalidate_pool(pool_id);
            }
            for pool_id in update.states.keys() {
                cache.invalidate_pool(pool_id);
            }
        });
        Self { tracker, graph, pathfinder, cache, gas_price_wei: Arc::new(RwLock::new(30_000_000_000u128)), max_hops: 3, numeraire_token: None, probe_depth: None }
    }

    /// Create a PriceEngine reusing an existing shared cache with custom gas price and hop limit.
    pub fn with_cache(tracker: ComponentTracker, graph: &'a TokenGraph, cache: Arc<RwLock<QuoteCache>>, gas_price_wei: Arc<RwLock<u128>>, max_hops: usize) -> Self {
        let pathfinder = Pathfinder::new(graph);
        // Register callback to invalidate cache entries when pool states change
        let cache_clone = cache.clone();
        tracker.register_callback(move |update| {
            let mut cache = cache_clone.write().unwrap();
            for pool_id in update.new_pairs.keys() {
                cache.invalidate_pool(pool_id);
            }
            for pool_id in update.removed_pairs.keys() {
                cache.invalidate_pool(pool_id);
            }
            for pool_id in update.states.keys() {
                cache.invalidate_pool(pool_id);
            }
        });
        Self { tracker, graph, pathfinder, cache, gas_price_wei, max_hops, numeraire_token: None, probe_depth: None }
    }

    /// Compute a price quote for a given input token, output token, and amount, using the best path only.
    pub fn quote(&self, token_in: &Bytes, token_out: &Bytes, amount_in: u128, block: Option<u64>) -> PriceQuote {
        // Use a shared cache keyed by block height
        let current_block = block.unwrap_or(0); // Use provided block or default to 0
        let cache_key = QuoteCacheKey { sell_token: token_in.clone(), buy_token: token_out.clone(), amount: amount_in, block: current_block };
        if let Some(cached) = self.cache.write().unwrap().get(&cache_key).cloned() {
            // Return cached result directly if no relevant pool state changes
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

        // Evaluate all candidate paths up to `max_hops` and pick the one delivering
        // the highest net amount (after protocol fee AND gas cost).
        let pq = self.quote_multi(token_in, token_out, amount_in, 1, block); // k=1: only keep best in path_details

        if let Some(amt) = pq.amount_out {
            // Cache the result (excluding depth metrics for now)
            let cached = CachedQuote {
                amount_out: amt,
                route: pq.route.clone(),
                route_pools: pq.path_details.get(0).map(|d| d.pools.clone()).unwrap_or_default(),
                mid_price: pq.mid_price,
                slippage_bps: pq.slippage_bps,
                spread_bps: pq.spread_bps,
                block: current_block, // Use current block for caching
                gross_amount_out: pq.gross_amount_out,
                fee_bps: pq.fee_bps,
                gas_estimate: pq.gas_estimate,
                price_impact_bps: pq.price_impact_bps,
            };
            self.cache.write().unwrap().insert(cache_key, cached);
        }

        pq
    }

    /// Compute a price quote for a given input token, output token, and amount, simulating up to K paths.
    pub fn quote_multi(&self, token_in: &Bytes, token_out: &Bytes, amount_in: u128, k: usize, block: Option<u64>) -> PriceQuote {
        let current_block = block.unwrap_or(0); // Use provided block or default to 0
        // First, attempt to serve paths from cache.
        let all_paths_nodes: Vec<Vec<NodeIndex>> = { // Renamed variable
            // Use block=0 for path cache to persist across blocks
            let path_key = PathCacheKey {
                sell_token: token_in.clone(),
                buy_token: token_out.clone(),
                block: 0, // Always 0 to cache paths independent of block height
                k: self.max_hops, // Cache based on max_hops to ensure all candidates are considered
            };

            let mut cache_w = self.cache.write().unwrap();

            if let Some(cached) = cache_w.get_paths(&path_key).cloned() {
                let mut reconstructed: Vec<Vec<NodeIndex>> = Vec::new();
                for addr_path in cached.paths {
                    let mut node_path = Vec::with_capacity(addr_path.len());
                    let mut valid = true;
                    for addr in addr_path {
                        if let Some(&idx) = self.graph.token_indices.get(&addr) {
                            // Verify node still exists in the current graph state
                            if self.graph.graph.node_weight(idx).is_some() {
                            node_path.push(idx);
                            } else {
                                valid = false;
                                break;
                            }
                        } else {
                            valid = false;
                            break;
                        }
                    }
                    if valid { reconstructed.push(node_path); }
                }
                reconstructed
            } else {
                // Cache miss – compute and store.
                let enumerated = self.pathfinder.enumerate_paths(token_in, token_out, self.max_hops);
                let addr_paths: Vec<Vec<Bytes>> = enumerated.iter().map(|p| {
                    p.iter().filter_map(|idx| self.graph.graph.node_weight(*idx).map(|n| n.address.clone())).collect()
                }).collect();
                let cached_value = CachedPaths { paths: addr_paths, block: current_block, timestamp: Instant::now() };
                // Only insert if computation was successful
                if !enumerated.is_empty() {
                cache_w.insert_paths(path_key, cached_value);
                }
                enumerated
            }
        };

        let mut seen_edge_seqs: HashSet<Vec<EdgeIndex>> = HashSet::new();

        // === Stage 1: Build a flat list of (node_path, edge_seq) tasks to evaluate ===
        let mut tasks: Vec<(Vec<NodeIndex>, Vec<EdgeIndex>)> = Vec::new();

        for node_path in &all_paths_nodes {
            if node_path.len() < 2 { continue; }

            // Collect edge indices for each hop
            let mut all_hop_edges: Vec<Vec<EdgeIndex>> = Vec::new();
            let mut path_possible = true;
            for win in node_path.windows(2) {
                let from_idx = win[0];
                let to_idx = win[1];
                let edges: Vec<_> = self.graph.graph.edges_connecting(from_idx, to_idx).map(|e| e.id()).collect();
                if edges.is_empty() {
                    path_possible = false;
                    break;
                }
                all_hop_edges.push(edges);
            }
            if !path_possible { continue; }

            // Enumerate the cartesian product of edge choices for this path.
            for edge_seq in all_hop_edges.into_iter().multi_cartesian_product() {
                // Deduplicate by full edge sequence across ALL paths
                if !seen_edge_seqs.insert(edge_seq.clone()) { continue; }

                tasks.push((node_path.clone(), edge_seq));
            }
        }

        // === Stage 2: Evaluate all tasks in parallel ===
        let evaluated: Vec<(u128, SinglePathQuote)> = tasks
            .into_par_iter()
            .filter_map(|(node_path, edge_seq)| {
                // Pass block number down if needed by optimization/simulation, TBD
                let q = self.optimize_depth_for_path(token_in, token_out, amount_in, &node_path, &edge_seq, block);
                q.amount_out.map(|net_amount| (net_amount, q))
            })
            .collect();

        // Sort by net amount_out descending (higher is better)
        let mut sorted_quotes: Vec<SinglePathQuote> = evaluated.into_iter().map(|(_, quote)| quote).collect();
        sorted_quotes.sort_by(|a, b| a.amount_out.cmp(&b.amount_out));
        sorted_quotes.reverse();

        // Determine best and calculate depth metrics for it
        let final_result = if let Some(best) = sorted_quotes.first().cloned() {
            let mut depth_metrics_map = HashMap::new();

            // --- Refined Depth Calculation --- 
            // Calculate a reference mid-price using a tiny amount simulation on the best path
            let reference_mid_price = if !best.node_path.is_empty() && !best.edge_seq.is_empty() {
                // Use a small fraction of the token unit instead of 1 wei
                let tokens_r_ref = self.tracker.all_tokens.read().unwrap();
                let start_decimals_ref = tokens_r_ref.get(token_in).map(|t| t.decimals).unwrap_or(18);
                let end_decimals_ref = tokens_r_ref.get(token_out).map(|t| t.decimals).unwrap_or(18);
                drop(tokens_r_ref);

                // Calculate tiny_amount_in (e.g., 0.0001 of the token)
                // Use saturating_sub to handle cases where start_decimals < 4
                let exponent = start_decimals_ref.saturating_sub(4);
                let tiny_amount_in = 10u128.pow(exponent as u32);
                eprintln!("[DEPTH_SEARCH_INFO] Using tiny_amount_in={} (decimals={}) for reference mid-price.", tiny_amount_in, start_decimals_ref);

                // Pass block number down if needed by simulation
                if let Some(tiny_output) = self.simulate_path_gross(tiny_amount_in, &best.node_path, &best.edge_seq, block) {
                    if tiny_output > 0 {
                         let tiny_input_dec = Decimal::from_u128(tiny_amount_in).unwrap_or_default() / Decimal::from_u128(10u128.pow(start_decimals_ref as u32)).unwrap_or_else(|| Decimal::ONE);
                         let tiny_output_dec = Decimal::from_u128(tiny_output).unwrap_or_default() / Decimal::from_u128(10u128.pow(end_decimals_ref as u32)).unwrap_or_else(|| Decimal::ONE);

                         if !tiny_input_dec.is_zero() {
                             Some(tiny_output_dec / tiny_input_dec)
                         } else {
                             eprintln!("[DEPTH_SEARCH_WARN] Tiny input decimal was zero.");
                             None
                         }
                    } else {
                        eprintln!("[DEPTH_SEARCH_WARN] Zero output for tiny amount: {}", tiny_amount_in);
                        None // Zero output for tiny amount
                    }
                } else {
                    eprintln!("[DEPTH_SEARCH_WARN] Simulation failed for tiny amount: {}", tiny_amount_in);
                    None // Simulation failed for tiny amount
                }
            } else {
                None // Best path invalid
            };

            // Ensure we have a valid *reference* mid-price and path to calculate depth
            if let Some(ref_mid_price) = reference_mid_price {
                 if !best.node_path.is_empty() && !best.edge_seq.is_empty() {
                    eprintln!("[DEPTH_SEARCH_INFO] Using reference mid-price for depth calc: {}", ref_mid_price);
                    for target_perc in [0.5, 1.0, 2.0] {
                        // Pass block number down
                        if let Some(depth_amount) = self.find_depth_for_slippage(
                            token_in,
                            token_out,
                            ref_mid_price, // Use the refined reference mid-price
                            &best.node_path, // Use the stored path
                            &best.edge_seq, // Use the stored edges
                            target_perc,
                            block,
                        ) {
                            depth_metrics_map.insert(format!("{:.1}%", target_perc), depth_amount);
                        }
                    }
                } else {
                    eprintln!("[DEPTH_SEARCH_WARN] Best path node/edge info missing, cannot calculate depth.");
                }
            } else {
                 eprintln!("[DEPTH_SEARCH_WARN] Could not calculate reference mid-price, skipping depth calculation.");
            }
            // --- End Refined Depth Calculation --- 

            let path_details = if k == 0 { sorted_quotes.clone() } else { sorted_quotes.iter().take(k).cloned().collect() };
            PriceQuote {
                // Keep the original quote results (amount_out, mid_price, etc.) based on the actual input amount
                amount_out: best.amount_out,
                route: best.route.clone(),
                mid_price: best.mid_price, // Report the mid-price for the actual trade size
                slippage_bps: best.slippage_bps,
                fee_bps: best.fee_bps,
                gas_estimate: best.gas_estimate,
                path_details,
                gross_amount_out: best.gross_amount_out,
                spread_bps: best.spread_bps,
                price_impact_bps: best.price_impact_bps,
                // Add the depth metrics calculated using the reference price
                depth_metrics: if depth_metrics_map.is_empty() { None } else { Some(depth_metrics_map) },
                cache_block: None,
            }
        } else {
            // Fallback to previous behavior (no path found)
            PriceQuote {
                amount_out: None,
                route: vec![],
                mid_price: None,
                slippage_bps: None,
                fee_bps: None,
                gas_estimate: None,
                path_details: vec![],
                gross_amount_out: None,
                spread_bps: None,
                price_impact_bps: None,
                depth_metrics: None,
                cache_block: None,
            }
        };

        // Log all evaluated paths to file (block=0 for now, can be updated to real block number if available)
        self.log_all_paths(token_in, token_out, amount_in, &sorted_quotes, current_block);
        final_result
    }

    /// Simulates a trade along a specific path and edge sequence, returning the gross output amount before gas/protocol fees.
    fn simulate_path_gross(
        &self,
        amount_in: u128,
        path_nodes: &[petgraph::prelude::NodeIndex],
        path_edges: &[petgraph::graph::EdgeIndex],
        block: Option<u64>, // Add block parameter
    ) -> Option<u128> {
        if path_nodes.len() < 2 || path_nodes.len() != path_edges.len() + 1 {
             eprintln!("[DEPTH_SIM_ERROR] Invalid path structure: nodes={}, edges={}", path_nodes.len(), path_edges.len());
             return None;
        }
        let mut current_amount = amount_in;
        let tokens_r = self.tracker.all_tokens.read().unwrap();
        let states_r = self.tracker.pool_states.read().unwrap();

        for (win, &edge_idx) in path_nodes.windows(2).zip(path_edges.iter()) {
            let from_idx = win[0];
            let to_idx = win[1];
            let from_node = self.graph.graph.node_weight(from_idx)?;
            let to_node = self.graph.graph.node_weight(to_idx)?;
            let edge_data = self.graph.graph.edge_weight(edge_idx)?;

            let pool_state = states_r.get(&edge_data.pool_id)?;
            let token_in_info = tokens_r.get(&from_node.address)?;
            let token_out_info = tokens_r.get(&to_node.address)?;

            // Use catch_unwind to handle potential panics in underlying simulation
                    let sim_result_unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                 pool_state.get_amount_out(current_amount.into(), token_in_info, token_out_info)
                    }));

                    match sim_result_unwind {
                        Ok(Ok(sim_result)) => {
                     match sim_result.amount.to_u128() {
                         Some(mut hop_output) => {
                             // Apply pool-specific fee if defined, but NO gas/protocol fee
                             if let Some(fee_frac) = edge_data.fee {
                                 if fee_frac > 0.0 && fee_frac < 1.0 { // Ensure fee is valid
                                     let hop_f = hop_output as f64;
                                     let net_f = hop_f * (1.0 - fee_frac);
                                     hop_output = net_f.floor() as u128; // Simulate fee deduction
                            } else if fee_frac >= 1.0 {
                                      eprintln!("[DEPTH_SIM_WARN] Pool: {} | Invalid fee >= 1.0 found: {}", edge_data.pool_id, fee_frac);
                                      hop_output = 0; // Assume 100% fee means zero output
                             }
                             }
                            current_amount = hop_output;
                             if current_amount == 0 { break; } // No point continuing if output is zero
                        }
                         None => {
                              eprintln!("[DEPTH_SIM_ERROR] Pool: {} | Tokens: {} -> {} | Reason: get_amount_out returned None (could not convert amount) for input {}", edge_data.pool_id, hex::encode(&from_node.address), hex::encode(&to_node.address), current_amount);
                              return None; // Conversion failed
                         }
                     }
                }
                 Ok(Err(e)) => {
                     eprintln!("[DEPTH_SIM_ERROR] Pool: {} | Tokens: {} -> {} | get_amount_out error: {:?}", edge_data.pool_id, hex::encode(&from_node.address), hex::encode(&to_node.address), e);
                     return None; // Simulation error
                }
                Err(panic_payload) => {
                     let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                         *s
                     } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                         s.as_str()
                                } else {
                         "Unknown panic reason"
                     };
                     eprintln!("[DEPTH_SIM_ERROR] Pool: {} | Tokens: {} -> {} | get_amount_out panicked: {}", edge_data.pool_id, hex::encode(&from_node.address), hex::encode(&to_node.address), panic_msg);
                     return None; // Panic during simulation
                }
            }
        }
        Some(current_amount)
    }

    /// Uses binary search to find the input amount that causes price slippage of `target_slippage_percent`.
    fn find_depth_for_slippage(
        &self,
        token_in: &Bytes,
        token_out: &Bytes,
        mid_price: Decimal, // The reference mid-price for the best path
        path_nodes: &[petgraph::prelude::NodeIndex],
        path_edges: &[petgraph::graph::EdgeIndex],
        target_slippage_percent: f64, // e.g., 0.5, 1.0, 2.0
        block: Option<u64>, // Add block parameter
    ) -> Option<u128> {
        if mid_price.is_zero() || path_nodes.len() < 2 { return None; }

        // Pre-pull data
        let tokens_r = self.tracker.all_tokens.read().unwrap();
        let states_r = self.tracker.pool_states.read().unwrap();

        struct Step<'a> {
            pool: &'a dyn tycho_simulation::protocol::state::ProtocolSim,
            token_in: &'a tycho_simulation::models::Token,
            token_out: &'a tycho_simulation::models::Token,
            fee: Option<f64>,
        }
        let mut steps: Vec<Step> = Vec::with_capacity(path_nodes.len()-1);
        for (win, &edge_idx) in path_nodes.windows(2).zip(path_edges.iter()) {
            let from_idx = win[0];
            let to_idx   = win[1];
            let from_node = self.graph.graph.node_weight(from_idx)?;
            let to_node   = self.graph.graph.node_weight(to_idx)?;
            let edge_data = self.graph.graph.edge_weight(edge_idx)?;
            let pool = states_r.get(&edge_data.pool_id)?;
            let token_in_info  = tokens_r.get(&from_node.address)?;
            let token_out_info = tokens_r.get(&to_node.address)?;
            steps.push(Step{ pool:&**pool, token_in:token_in_info, token_out:token_out_info, fee: edge_data.fee });
        }
        let start_decimals = tokens_r.get(token_in)?.decimals;
        let end_decimals   = tokens_r.get(token_out)?.decimals;

        // helper to create Decimal safely from u128, falling back to f64 if overflow
        fn to_dec(val: u128) -> Decimal {
            Decimal::from_u128(val).or_else(|| Decimal::from_f64(val as f64)).unwrap_or_else(|| Decimal::ZERO)
        }

        // Pre-compute denominators for Decimal scaling; ensure non-zero
        let start_den: Decimal = {
            let d = to_dec(10u128.pow(start_decimals as u32));
            if d.is_zero() { Decimal::ONE } else { d }
        };
        let end_den: Decimal = {
            let d = to_dec(10u128.pow(end_decimals as u32));
            if d.is_zero() { Decimal::ONE } else { d }
        };

        // fast sim
        let sim = |amt: u128| -> Option<u128> {
            let mut cur = amt;
            for s in &steps {
                let res = s.pool.get_amount_out(cur.into(), s.token_in, s.token_out).ok()?;
                let mut out = res.amount.to_u128()?;
                if let Some(f) = s.fee { if f>0.0 && f<1.0 { out=((out as f64)*(1.0-f)).floor() as u128; } else if f>=1.0 { return Some(0);} }
                cur = out; if cur==0 { return Some(0);} }
            Some(cur)
        };

        let target_factor = Decimal::from_f64(1.0 - target_slippage_percent/100.0)?;
        let target_price  = mid_price * target_factor;

        let mut low = 1u128;
        let mut high = 10u128.pow(start_decimals.min(38) as u32); // cap exponent to avoid overflow

        // expand high
        for _ in 0..40 {
            if let Some(out) = sim(high) {
                let inp_dec = to_dec(high) / start_den;
                let out_dec = to_dec(out) / end_den;
                if !inp_dec.is_zero() && out_dec / inp_dec <= target_price {
                    break;
                }
            }
            high = high.saturating_mul(10);
            // prevent overflow into zero
            if high == 0 { break; }
        }

        let mut best=None;
        for _ in 0..100 {
            if high<=low { break; }
            let mid=low+((high-low)/2);
            let out=sim(mid)?;

            // Handle zero output case: Treat as infinite slippage
            if out == 0 {
                best = Some(mid);
                high = mid;
                if high <= low + 1 { break; }
                continue; // Skip decimal conversion for zero output
            }

            let inp_dec = to_dec(mid) / start_den;
            if inp_dec.is_zero() {
                low = mid;
                if high <= low + 1 { break; }
                continue;
            }
            let out_dec = to_dec(out) / end_den;
            let eff = out_dec / inp_dec;
            if eff <= target_price {
                best = Some(mid);
                high = mid;
            } else {
                low = mid;
            }
            if high<=low+1 { break; }
        }
        best
    }

    /// Optimize input depth across multiple sampling points to maximize net output per input for a single path.
    fn optimize_depth_for_path(&self, token_in: &Bytes, token_out: &Bytes, default_amount_in: u128, path: &[NodeIndex], edge_seq: &[EdgeIndex], block: Option<u64>) -> SinglePathQuote {
        if let Some(probe) = self.probe_depth {
            // Sample depths around the configured probe_depth
            let mut depths = vec![probe / 10, probe / 2, probe, probe.saturating_mul(2), probe.saturating_mul(10)];
            // Keep only positive, sort and dedupe
            depths.retain(|&d| d > 0);
            depths.sort_unstable(); depths.dedup();
            // Start with the base probe depth quote
            let mut best_quote = self.quote_single_path_with_edges(token_in, token_out, probe, path, edge_seq, block);
            let mut best_ratio = best_quote.amount_out.map(|o| o as f64 / probe as f64).unwrap_or(0.0);
            // Evaluate each sampled depth
            for &amt in &depths {
                let q = self.quote_single_path_with_edges(token_in, token_out, amt, path, edge_seq, block);
                if let Some(out) = q.amount_out {
                    let ratio = out as f64 / amt as f64;
                    if ratio > best_ratio {
                        best_ratio = ratio;
                        best_quote = q;
                    }
                }
            }
            best_quote
        } else {
            // No probe_depth configured, fallback to using the default input amount
            self.quote_single_path_with_edges(token_in, token_out, default_amount_in, path, edge_seq, block)
        }
    }

    pub fn from_config(tracker: ComponentTracker, graph: &'a TokenGraph, cache: Arc<RwLock<QuoteCache>>, gas_price_wei: Arc<RwLock<u128>>, config: &AppConfig) -> Self {
        let max_hops = config.max_hops.unwrap_or(3);
        let engine = Self { tracker, graph, pathfinder: Pathfinder::new(graph), cache, gas_price_wei, max_hops, numeraire_token: config.numeraire_token.clone(), probe_depth: config.probe_depth };
        // Register callback to invalidate cache entries for pools and tokens when any pool state changes
        {
            let cache_clone = engine.cache.clone();
            let tracker_clone = engine.tracker.clone();
            engine.tracker.register_callback(move |update| {
                let mut cache = cache_clone.write().unwrap();
                // Combine pool IDs from new, removed, and state updates
                for pool_id in update.new_pairs.keys().chain(update.removed_pairs.keys()).chain(update.states.keys()) {
                    // Invalidate per-pool caches
                    cache.invalidate_pool(pool_id);
                    // Invalidate per-token caches for tokens in this pool
                    if let Ok(pools_map) = tracker_clone.all_pools.read() {
                        if let Some(component) = pools_map.get(pool_id) {
                            for token in &component.tokens {
                                cache.invalidate_token(&token.address);
                            }
                        }
                    }
                }
            });
        }
        engine
    }

    pub fn precompute_all_quotes(&self, block: Option<u64>) {
        // Path cache invalidation is handled incrementally via invalidate_pool and purge_expired,
        // so we do **not** clear the entire path cache here. Retaining cached paths enables
        // path-hit metrics and avoids recomputation when the underlying graph has not changed.

        // Only run if both a numeraire token and probe depth are configured.
        let numeraire = match &self.numeraire_token {
            Some(t) => t.clone(),
            None => {
                // Nothing to do if no numeraire token was supplied via config/CLI
                return;
            }
        };
        let amount_in = match self.probe_depth {
            Some(d) => d,
            None => {
                // Without a probe depth, we cannot meaningfully pre-compute quotes.
                return;
            }
        };

        // Take a snapshot of all token addresses under read lock so we can iterate without
        // holding the lock during heavy computations.
        let token_addresses: Vec<Bytes> = {
            let tokens_r = self.tracker.all_tokens.read().unwrap();
            tokens_r.keys().cloned().collect()
        };

        for token_addr in token_addresses {
            if token_addr == numeraire { continue; }

            // Compute quote via quote_multi (spec-compliant) and persist best result manually.
            let pq = self.quote_multi(&numeraire, &token_addr, amount_in, 1, block);

            if let Some(best_out) = pq.amount_out {
                let current_block = block.unwrap_or(0); // Use provided block or default
                let cache_key = QuoteCacheKey {
                    sell_token: numeraire.clone(),
                    buy_token: token_addr.clone(),
                    amount: amount_in,
                    block: current_block, // Use current block for caching
                };

                let cached = CachedQuote {
                    amount_out: best_out,
                    route: pq.route.clone(),
                    route_pools: pq.path_details.get(0).map(|d| d.pools.clone()).unwrap_or_default(),
                    mid_price: pq.mid_price,
                    slippage_bps: pq.slippage_bps,
                    spread_bps: pq.spread_bps,
                    block: current_block, // Use current block for caching
                    gross_amount_out: pq.gross_amount_out,
                    fee_bps: pq.fee_bps,
                    gas_estimate: pq.gas_estimate,
                    price_impact_bps: pq.price_impact_bps,
                };
                self.cache.write().unwrap().insert(cache_key, cached);

                // Persist price history for this quote (P-5)
                let timestamp = Utc::now().to_rfc3339();
                let price_history_path = "price_history.csv";
                let file_exists = Path::new(price_history_path).exists();
                let mut ph_file = match OpenOptions::new().create(true).append(true).open(price_history_path) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("[PRICE_HISTORY_ERROR] Could not open {}: {}", price_history_path, e);
                        continue;
                    }
                };
                if !file_exists {
                    writeln!(ph_file, "timestamp,sell_token,buy_token,sell_amount,amount_out,mid_price,slippage_bps,fee_bps,gas_estimate,gross_amount_out,spread_bps").ok();
                }
                writeln!(ph_file, "{},{},{},{},{},{},{},{},{},{},{}",
                         timestamp,
                         format!("0x{}", hex::encode(&numeraire)),
                         format!("0x{}", hex::encode(&token_addr)),
                         amount_in,
                         best_out,
                         pq.mid_price.map(|d| d.to_string()).unwrap_or_default(),
                         pq.slippage_bps.map(|d| d.to_string()).unwrap_or_default(),
                         pq.fee_bps.map(|d| d.to_string()).unwrap_or_default(),
                         pq.gas_estimate.map(|g| g.to_string()).unwrap_or_default(),
                         pq.gross_amount_out.map(|a| a.to_string()).unwrap_or_default(),
                         pq.spread_bps.map(|d| d.to_string()).unwrap_or_default()
                ).ok();
            }
        }
    }

    /// Write all evaluated paths and their net amounts to a log file for the current block or timestamp
    fn log_all_paths(&self, token_in: &Bytes, token_out: &Bytes, amount_in: u128, evaluated: &Vec<SinglePathQuote>, block: u64) {
        let now = Utc::now();
        let filename = if block > 0 {
            format!("all_paths_block_{}.log", block)
        } else {
            format!("all_paths_{}.log", now.format("%Y%m%dT%H%M%S"))
        };
        // Overwrite existing log file each run to reset path numbering
        let mut file = match OpenOptions::new().create(true).write(true).truncate(true).open(&filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[LOGGING ERROR] Could not open log file {}: {}", filename, e);
                return;
            }
        };
        writeln!(file, "==== All Paths for {} -> {} (amount_in: {}) at block {} ====", hex::encode(token_in), hex::encode(token_out), amount_in, block).ok();
        for (i, path) in evaluated.iter().enumerate() {
            let route_str = path.route.iter().map(|b| format!("0x{}", hex::encode(b))).collect::<Vec<_>>().join(" -> ");
            let pools_str = path.pools.join(", ");
            let amt = path.amount_out.map(|a| a.to_string()).unwrap_or("None".to_string());
            writeln!(file, "Path {}: Route=[{}], Pools=[{}], Net Output Amount={}", i + 1, route_str, pools_str, amt).ok();
        }
        writeln!(file, "==== END ====").ok();
    }

    /// Calculate price impact in basis points (bps).
    /// price_impact = (expected_amount_at_mid_price - gross_actual_amount) / expected_amount_at_mid_price
    fn calculate_price_impact_bps(
        start_amount_dec: Decimal,
        gross_actual_amount_dec: Decimal,
        mid_price: Decimal,
    ) -> Option<Decimal> {
         if start_amount_dec.is_zero() || mid_price.is_zero() { return None; }
         let expected_amount_at_mid_price = start_amount_dec * mid_price;
         if expected_amount_at_mid_price.is_zero() { return None; }

         Some(((expected_amount_at_mid_price - gross_actual_amount_dec) / expected_amount_at_mid_price) * Decimal::new(10_000, 0))
    }

    /// Calculate slippage in basis points (bps) relative to the *final* mid-price of the executed path.
    /// slippage = (expected_amount_at_final_mid_price - net_actual_amount) / expected_amount_at_final_mid_price
    fn calculate_slippage_bps(
        start_amount_dec: Decimal,
        net_actual_amount_dec: Decimal, // Amount after gas and all fees
        final_mid_price: Decimal, // The effective mid-price calculated for this specific path execution
    ) -> Option<Decimal> {
        if start_amount_dec.is_zero() || final_mid_price.is_zero() {
            return None;
        }
        let expected_amount = start_amount_dec * final_mid_price;
        if expected_amount.is_zero() {
            return None;
        }
        Some(((expected_amount - net_actual_amount_dec) / expected_amount) * Decimal::new(10000, 0))
    }

    /// Calculate spread in basis points (bps) - often similar to slippage definition in this context.
    /// Using the same calculation as slippage for now.
    fn calculate_spread_bps(
        start_amount_dec: Decimal,
        net_actual_amount_dec: Decimal,
        final_mid_price: Decimal,
    ) -> Option<Decimal> {
        Self::calculate_slippage_bps(start_amount_dec, net_actual_amount_dec, final_mid_price)
    }

    /// Simulate a single path with explicit pool-edge sequence and return detailed quote metrics.
    pub fn quote_single_path_with_edges(&self, token_in: &Bytes, token_out: &Bytes, amount_in: u128, path: &[NodeIndex], edge_seq: &[EdgeIndex], block: Option<u64>) -> SinglePathQuote {
        let mut route_addresses = vec![];
        let mut pool_ids = vec![];
        let mut current_amount = amount_in;
        let mut cumulative_mid_price = Decimal::ONE;
        let mut ok = true;
        let mut total_fee_bps_sum = Decimal::ZERO;
        let mut cumulative_fee_mult = Decimal::ONE;
        let mut gas_estimate = 0u64;

        let tokens_r = self.tracker.all_tokens.read().unwrap();
        let start_dec = match tokens_r.get(token_in) { Some(t)=>t.decimals, None=>{return self.invalid_path_quote(path, edge_seq, amount_in);} };
        let end_dec   = match tokens_r.get(token_out){ Some(t)=>t.decimals, None=>{return self.invalid_path_quote(path, edge_seq, amount_in);} };
        drop(tokens_r);

        let protocol_fee_frac = Decimal::from_f64(0.0025).unwrap_or_default();

        for (i, win) in path.windows(2).enumerate() {
            let edge_idx = match edge_seq.get(i) { Some(e)=>*e, None=>{ ok=false; break;} };
            let from_idx = win[0]; let to_idx = win[1];
            let from_node = match self.graph.graph.node_weight(from_idx){Some(n)=>n, None=>{ok=false;break;}};
            let to_node   = match self.graph.graph.node_weight(to_idx){Some(n)=>n, None=>{ok=false;break;}};

            if i==0 { route_addresses.push(from_node.address.clone()); }

            let states_r = self.tracker.pool_states.read().unwrap();
            let tokens_r = self.tracker.all_tokens.read().unwrap();
            let edge_data = match self.graph.graph.edge_weight(edge_idx){Some(e)=>e, None=>{ok=false;break;}};
            pool_ids.push(edge_data.pool_id.clone());

            let pool_state = match states_r.get(&edge_data.pool_id){Some(p)=>p, None=>{ok=false;break;}};
            let token_in_info  = match tokens_r.get(&from_node.address){Some(t)=>t, None=>{ok=false;break;}};
            let token_out_info = match tokens_r.get(&to_node.address){Some(t)=>t, None=>{ok=false;break;}};

            let sim_result = match pool_state.get_amount_out(current_amount.into(), token_in_info, token_out_info) {
                Ok(res) => res,
                Err(_) => { ok = false; break; }
            };
            let mut hop_output = match sim_result.amount.to_u128() {
                Some(v) => v,
                None => { ok = false; break; }
            };

            if let Some(fee) = edge_data.fee { if fee>0.0 && fee<1.0 {
                let fee_dec = Decimal::from_f64(fee).unwrap_or_default();
                cumulative_fee_mult *= Decimal::ONE - fee_dec;
                total_fee_bps_sum += fee_dec*Decimal::new(10000,0);
                hop_output = ((hop_output as f64)*(1.0-fee)).floor() as u128;
            }}

            current_amount = hop_output; if current_amount==0 { ok=false; break; }

            // mid-price
            if let Ok(spot_f64) = pool_state.spot_price(token_in_info, token_out_info) { if spot_f64>0.0 {
                let mut spot_dec = Decimal::from_f64(spot_f64).unwrap_or_default();
                if token_in_info.address > token_out_info.address && !spot_dec.is_zero() { spot_dec = Decimal::ONE/spot_dec; }
                if let Some(f) = edge_data.fee { if f>0.0 && f<1.0 { spot_dec*=Decimal::ONE-Decimal::from_f64(f).unwrap_or_default(); } }
                if !spot_dec.is_zero() { cumulative_mid_price*=spot_dec; }
            }}

            let gas_for_hop = sim_result.gas.to_u64().unwrap_or(30000);
            gas_estimate = gas_estimate.saturating_add(gas_for_hop);

            route_addresses.push(to_node.address.clone());
        }

        if !ok { return self.invalid_path_quote(path, edge_seq, amount_in); }

        let gross_amount_out = current_amount;
        // gas cost convert
        let gas_cost_token = if gas_estimate>0 { let gas_price=*self.gas_price_wei.read().unwrap(); let wei=gas_price as u128 * gas_estimate as u128; if end_dec>=18 { wei*10u128.pow((end_dec-18) as u32)} else { wei/10u128.pow((18-end_dec) as u32)} } else {0};
        let after_gas = gross_amount_out.saturating_sub(gas_cost_token);
        let protocol_fee_amt = (Decimal::from_u128(after_gas).unwrap_or_default()*protocol_fee_frac).floor().to_u128().unwrap_or(0);
        let net_amount_out = after_gas.saturating_sub(protocol_fee_amt);

        // If the output is zero, treat as invalid path
        if net_amount_out == 0 {
            return self.invalid_path_quote(path, edge_seq, amount_in);
        }

        let start_dec = Decimal::from_u128(amount_in).unwrap_or_default()/Decimal::from_u128(10u128.pow(start_dec as u32)).unwrap_or(Decimal::ONE);
        let gross_dec = Decimal::from_u128(gross_amount_out).unwrap_or_default()/Decimal::from_u128(10u128.pow(end_dec as u32)).unwrap_or(Decimal::ONE);
        let net_dec = Decimal::from_u128(net_amount_out).unwrap_or_default()/Decimal::from_u128(10u128.pow(end_dec as u32)).unwrap_or(Decimal::ONE);

        let final_mid_price = if start_dec.is_zero(){Decimal::ZERO}else{gross_dec/start_dec};
        let price_impact = Self::calculate_price_impact_bps(start_dec, gross_dec, final_mid_price);
        let slippage_bps = Self::calculate_slippage_bps(start_dec, net_dec, final_mid_price);
        let spread_bps = Self::calculate_spread_bps(start_dec, net_dec, final_mid_price);
        let total_fee_bps = total_fee_bps_sum + protocol_fee_frac*Decimal::new(10000,0);

        SinglePathQuote {
            amount_out: Some(net_amount_out),
            route: route_addresses,
            mid_price: Some(final_mid_price),
            slippage_bps,
            fee_bps: Some(total_fee_bps),
            gas_estimate: if gas_estimate>0{Some(gas_estimate)}else{None},
            gross_amount_out: Some(gross_amount_out),
            spread_bps,
            price_impact_bps: price_impact,
            pools: pool_ids,
            input_amount: Some(amount_in),
            node_path: path.to_vec(),
            edge_seq: edge_seq.to_vec(),
        }
    }

    /// Return a stubbed SinglePathQuote for error cases
    fn invalid_path_quote(&self, path: &[NodeIndex], edge_seq: &[EdgeIndex], amount_in: u128) -> SinglePathQuote {
        SinglePathQuote {
            amount_out: None,
            route: vec![],
            mid_price: None,
            slippage_bps: None,
            fee_bps: None,
            gas_estimate: None,
            gross_amount_out: None,
            spread_bps: None,
            price_impact_bps: None,
            pools: vec![],
            input_amount: Some(amount_in),
            node_path: path.to_vec(),
            edge_seq: edge_seq.to_vec(),
        }
    }

    /// For each token, simulate swapping 1 whole unit (10^decimals) into ETH on the current block.
    /// Always uses the full multi-path simulation to reflect any liquidity changes, bypassing the single-quote cache.
    pub fn quote_tokens_vs_eth(&self, tokens: &[Bytes], eth: &Bytes, _dummy: u128, block: Option<u64>) -> Vec<(Bytes, PriceQuote)> {
        tokens
            .par_iter()
            .filter(|t| *t != eth)
            .map(|t| {
                // Determine 1-token amount based on decimals (default 18)
                let decimals = {
                    let tokens_r = self.tracker.all_tokens.read().unwrap();
                    tokens_r.get(t).map(|ti| ti.decimals).unwrap_or(18)
                } as u32;
                let amount_per_token = 10u128.pow(decimals);

                // Use cached or fresh quote per token against ETH (enables cache_block annotation)
                let q = self.quote(t, eth, amount_per_token, block);

                (t.clone(), q)
            })
            .collect()
    }

    /// For each token, compute the pure unit price vs ETH ignoring trading and gas fees (spot-only).
    pub fn list_unit_price_vs_eth(&self, tokens: &[Bytes], eth: &Bytes) -> Vec<(Bytes, Decimal)> {
        tokens
            .par_iter()
            .filter(|t| *t != eth)
            .map(|t| {
                let price = self.unit_price_path(t, eth).unwrap_or(Decimal::ZERO);
                (t.clone(), price)
            })
            .collect()
    }

    /// Compute the best pure spot price for a single token vs ETH along all cached paths.
    fn unit_price_path(&self, token_in: &Bytes, token_out: &Bytes) -> Option<Decimal> {
        // Prepare path lookup key, always block=0 for path cache
        let path_key = PathCacheKey { sell_token: token_in.clone(), buy_token: token_out.clone(), block: 0, k: self.max_hops };
        let mut cache_w = self.cache.write().unwrap();
        let addr_paths = if let Some(cached) = cache_w.get_paths(&path_key).cloned() {
            cached.paths
        } else {
            // Enumerate paths and cache them
            let node_paths = self.pathfinder.enumerate_paths(token_in, token_out, self.max_hops);
            let addr_paths: Vec<Vec<Bytes>> = node_paths.iter().map(|p| {
                p.iter()
                 .filter_map(|idx| self.graph.graph.node_weight(*idx).map(|n| n.address.clone()))
                 .collect()
            }).collect();
            cache_w.insert_paths(path_key.clone(), CachedPaths { paths: addr_paths.clone(), block: 0, timestamp: Instant::now() });
            addr_paths
        };
        drop(cache_w);

        let tokens_r = self.tracker.all_tokens.read().unwrap();
        let states_r = self.tracker.pool_states.read().unwrap();

        let mut best: Option<Decimal> = None;
        for addr_path in addr_paths {
            // Reconstruct node indices and validate
            let mut node_idx_path = Vec::new();
            let mut valid = true;
            for addr in &addr_path {
                if let Some(&idx) = self.graph.token_indices.get(addr) {
                    if self.graph.graph.node_weight(idx).is_some() {
                        node_idx_path.push(idx);
                    } else { valid = false; break }
                } else { valid = false; break }
            }
            if !valid || node_idx_path.len() < 2 { continue }

            // Calculate pure spot price along this path
            let mut price = Decimal::ONE;
            for win in node_idx_path.windows(2) {
                let from_idx = win[0]; let to_idx = win[1];
                // pick an edge connecting the nodes
                let edge_idx = match self.graph.graph.edges_connecting(from_idx, to_idx).map(|e| e.id()).next() {
                    Some(e) => e,
                    None => { price = Decimal::ZERO; break }
                };
                let edge_data = self.graph.graph.edge_weight(edge_idx)?;
                let pool_state = states_r.get(&edge_data.pool_id)?;
                let from_addr = self.graph.graph.node_weight(from_idx)?.address.clone();
                let to_addr = self.graph.graph.node_weight(to_idx)?.address.clone();
                let token_in_info = tokens_r.get(&from_addr)?;
                let token_out_info = tokens_r.get(&to_addr)?;
                let spot_f64 = pool_state.spot_price(token_in_info, token_out_info).ok()?;
                let mut sp = Decimal::from_f64(spot_f64).unwrap_or_default();
                // invert if necessary
                if token_in_info.address > token_out_info.address && !sp.is_zero() {
                    sp = Decimal::ONE / sp;
                }
                price *= sp;
            }
            best = Some(match best {
                Some(prev) if prev > price => prev,
                _ => price
            });
        }
        best
    }
}

/// Calculate slippage in basis points (bps).
/// slippage = (expected_amount - actual_amount) / expected_amount
/// expected_amount = start_amount * mid_price (where mid_price is End/Start)
fn calculate_slippage_bps(
    start_amount: Decimal,
    final_amount: Decimal,
    mid_price: Decimal,
) -> Option<Decimal> {
    if start_amount == Decimal::ZERO || mid_price == Decimal::ZERO {
        return None; // Avoid division by zero or meaningless calculation
    }

    // expected amount = start_amount_in_start_token * price_of_start_in_end_token
    let expected_amount = start_amount * mid_price;

    if expected_amount == Decimal::ZERO {
        return None; // Avoid division by zero
    }

    // Slippage = (Expected - Actual) / Expected
    let slippage_ratio = (expected_amount - final_amount) / expected_amount;

    // Convert to basis points (1 bp = 0.01% = 0.0001)
    Some(slippage_ratio * Decimal::new(10000, 0))
}

/// Calculate spread in basis points (bps).
/// spread = (expected_amount - actual_amount) / expected_amount
/// expected_amount = start_amount * mid_price (where mid_price is End/Start)
fn calculate_spread_bps(
    start_amount: Decimal,
    final_amount: Decimal,
    mid_price: Decimal,
) -> Option<Decimal> {
    if start_amount == Decimal::ZERO || mid_price == Decimal::ZERO {
        return None; // Avoid division by zero or meaningless calculation
    }

    // expected amount = start_amount_in_start_token * price_of_start_in_end_token
    let expected_amount = start_amount * mid_price;

    if expected_amount == Decimal::ZERO {
        return None; // Avoid division by zero
    }

    // Spread = (Expected - Actual) / Expected
    let spread_ratio = (expected_amount - final_amount) / expected_amount;

    // Convert to basis points (1 bp = 0.01% = 0.0001)
    Some(spread_ratio * Decimal::new(10000, 0))
}

// --- price impact calculation helper ---
fn calculate_price_impact_bps(expected: Decimal, actual: Decimal) -> Option<Decimal> {
    if expected.is_zero() { return None }
    Some(((expected - actual) / expected) * Decimal::new(10_000, 0))
} 