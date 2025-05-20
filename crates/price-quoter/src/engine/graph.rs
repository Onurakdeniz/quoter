//! Token/pool graph model, edge/node management, graph updates.

use petgraph::stable_graph::StableDiGraph;
use indexmap::IndexMap;
use tycho_simulation::protocol::models::ProtocolComponent;
use tycho_simulation::tycho_common::Bytes;
use alloy_primitives::Address;
use tycho_simulation::protocol::errors::SimulationError;
use tycho_simulation::models::Token;
use std::collections::HashMap;
use tycho_simulation::protocol::state::ProtocolSim;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use petgraph::prelude::NodeIndex;
use petgraph::prelude::EdgeIndex;

/// Represents a token node in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenNode {
    pub address: Bytes,
    pub symbol: String,
    pub decimals: u8,
}

/// Represents a pool edge in the graph.
#[derive(Debug, Clone)]
pub struct PoolEdge {
    pub pool_id: String,
    pub protocol: String,
    pub fee: Option<f64>,
    pub weight: Option<f64>, // -ln(spot_price)
    pub reserves: Option<(f64, f64)>, // (reserve0, reserve1) if available
}

/// The main token/pool graph structure.
pub struct TokenGraph {
    pub graph: StableDiGraph<TokenNode, PoolEdge>,
    pub token_indices: IndexMap<Bytes, NodeIndex>,
    pub pool_ids: std::collections::HashSet<String>, // Track pool IDs currently in the graph
}

impl TokenGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            token_indices: IndexMap::new(),
            pool_ids: std::collections::HashSet::new(),
        }
    }

    /// Derives a sequence of edge indices for a given node path.
    /// For each pair of nodes (u, v) in the path, it selects the "best" available edge.
    /// "Best" is currently defined as:
    /// 1. An edge with a defined Some(weight).
    /// 2. Among those, one with the lowest non-negative weight.
    /// 3. If no weights or all negative, or no edges, it might fail for that segment.
    /// Returns None if any segment of the path has no suitable edge.
    pub fn derive_edges_for_node_path(&self, node_path: &[NodeIndex]) -> Option<Vec<EdgeIndex>> {
        if node_path.len() < 2 {
            return Some(Vec::new()); // No edges for a path shorter than 2 nodes
        }

        let mut edge_indices = Vec::with_capacity(node_path.len() - 1);

        for i in 0..(node_path.len() - 1) {
            let u = node_path[i];
            let v = node_path[i+1];

            let connecting_edges: Vec<_> = self.graph.edges_connecting(u, v).collect();

            if connecting_edges.is_empty() {
                return None; // No edge found for this segment
            }

            // Select the "best" edge.
            // Prioritize edges with Some(weight), then lowest non-negative weight.
            let best_edge = connecting_edges.iter()
                .filter_map(|edge_ref| edge_ref.weight().weight.map(|w| (edge_ref.id(), w)))
                .min_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((edge_id, _)) = best_edge {
                edge_indices.push(edge_id);
            } else {
                // If no edges with weights, or some other criteria, pick the first one.
                // Current pathfinder.best_path uses weights, so edges on that path should ideally have them.
                if let Some(first_edge) = connecting_edges.first() {
                    edge_indices.push(first_edge.id()); // Fallback to first edge if no weights found/comparable
                } else {
                    return None; // Should be caught by connecting_edges.is_empty() already
                }
            }
        }
        Some(edge_indices)
    }

    /// Helper to compute slippage-adjusted effective price for a small trade
    fn compute_log_slippage_weight(
        pool_state: &Box<dyn ProtocolSim + Send + Sync>,
        token_in: &Token,
        token_out: &Token,
        fee: f64,
    ) -> Option<f64> {
        // Use a small fraction of reserves as the trade size, e.g., 0.01%
        // If reserves are not available, fallback to None
        // We\'ll use 0.0001 (0.01%) of input token\'s one() as the trade size
        let decimals = token_in.decimals;
        let one = BigUint::from(10u64).pow(decimals as u32);
        let trade_size = &one / BigUint::from(1_000_000u64); // 0.0001% of one token
        // Simulate get_amount_out for this small trade
        match pool_state.get_amount_out(trade_size.clone(), token_in, token_out) {
            Ok(result) => {
                let amount_out = result.amount;
                if amount_out > BigUint::from(0u8) {
                    // Effective price = amount_out / trade_size (output per input)
                    let effective_price = amount_out.to_f64().unwrap_or(0.0) / trade_size.to_f64().unwrap_or(1.0);
                    if effective_price > 0.0 {
                        // Apply fee (if not already included)
                        let effective_price_with_fee = effective_price * (1.0 - fee);
                        if effective_price_with_fee > 0.0 {
                            return Some(-effective_price_with_fee.ln().abs());
                        }
                    }
                }
                None
            }
            Err(_) => None,
        }
    }

    /// Add or update tokens and pools from the latest state, using tracker for price/fee info.
    pub fn update_from_components_with_tracker(
        &mut self,
        pools: &std::collections::HashMap<String, ProtocolComponent>,
        pool_states: &HashMap<String, Box<dyn ProtocolSim + Send + Sync>>,
        all_tokens: &HashMap<Bytes, Token>,
    ) {
        let new_pool_ids: std::collections::HashSet<String> = pools.keys().cloned().collect();

        // 1. Remove nodes/edges associated with pools that are no longer present
        let pools_to_remove = self.pool_ids.difference(&new_pool_ids).cloned().collect::<Vec<_>>();
        for removed_pool_id in pools_to_remove {
            // Find edges associated with this pool and remove them
            let edges_to_remove: Vec<_> = self.graph.edge_indices()
                .filter(|&e| self.graph.edge_weight(e).map_or(false, |ew| ew.pool_id == removed_pool_id))
                .collect();
            for edge_index in edges_to_remove.into_iter().rev() {
                self.graph.remove_edge(edge_index);
            }
        }
        // Prune orphan nodes (no in-edges or out-edges)
        let orphan_nodes: Vec<_> = self.graph.node_indices()
            .filter(|&idx| self.graph.edges(idx).count() == 0)
            .collect();
        for idx in orphan_nodes.into_iter().rev() {
            let node = self.graph.node_weight(idx).cloned();
            self.graph.remove_node(idx);
            if let Some(node) = node {
                self.token_indices.shift_remove(&node.address);
            }
        }

        // Add new tokens as nodes
        for pool in pools.values() {
            for token in &pool.tokens {
                let addr = token.address.clone();
                if !self.token_indices.contains_key(&addr) {
                    let node = TokenNode {
                        address: addr.clone(),
                        symbol: token.symbol.clone(),
                        decimals: token.decimals as u8,
                    };
                    let idx = self.graph.add_node(node);
                    self.token_indices.insert(addr, idx);
                }
            }
        }
        // Incrementally update directed edges for each pool (A->B, B->A for 2-token pools)
        for (pool_id, pool) in pools.iter() {
            let tokens = &pool.tokens;
            if tokens.len() < 2 { continue; }
            // pre-fetch pool_state once
            let pool_state = if let Some(ps) = pool_states.get(pool_id) { ps } else { continue };
            let fee_default = match pool_state.fee() { f if f > 0.0 => Some(f), _ => Some(0.0025) };

            for i in 0..tokens.len() {
                for j in 0..tokens.len() {
                    if i==j { continue; }
                    let from_addr = &tokens[i].address;
                    let to_addr   = &tokens[j].address;
                    let (from_idx, to_idx) = match (self.token_indices.get(from_addr), self.token_indices.get(to_addr)) {
                        (Some(&f), Some(&t)) => (f,t), _=>continue
                    };

                    // Build / update edge weight metadata fast helper fn
                    let build_edge = |fee: Option<f64>| -> (Option<f64>, Option<(f64,f64)>) {
                        let mut weight=None; let mut reserves=None;
                        if let (Some(tok_in), Some(tok_out)) = (all_tokens.get(from_addr), all_tokens.get(to_addr)) {
                            if let Some(f) = fee {
                                weight = Self::compute_log_slippage_weight(pool_state, tok_in, tok_out, f);
                            }
                            if weight.is_none() {
                                if let Ok(spot)=pool_state.spot_price(tok_in, tok_out) { if spot>0.0 {
                                    let eff = spot*(1.0-fee.unwrap_or(0.0025)); if eff>0.0 { weight=Some((-eff.ln()).abs()); }
                                }}
                            }
                            if let (Ok(addr_in), Ok(addr_out)) = (Self::bytes_to_address(&tok_in.address), Self::bytes_to_address(&tok_out.address)) {
                                if let Ok((max_in, max_out)) = pool_state.get_limits(addr_in, addr_out) {
                                    reserves = Some((max_in.to_f64().unwrap_or(0.0), max_out.to_f64().unwrap_or(0.0)));
                                }
                            }
                        }
                        (weight,reserves)
                    };

                    let (weight,reserves) = build_edge(fee_default);

                    // Check if an edge for this pool already exists
                    let existing_edge_idx = self.graph
                        .edges_connecting(from_idx, to_idx)
                        .find(|edge| edge.weight().pool_id.as_str() == pool_id.as_str())
                        .map(|e| e.id());

                    match existing_edge_idx {
                        Some(eidx) => {
                            if let Some(edge_w) = self.graph.edge_weight_mut(eidx) {
                                edge_w.fee = fee_default;
                                edge_w.weight = weight;
                                edge_w.reserves = reserves;
                            }
                        }
                        None => {
                            let edge = PoolEdge {
                                pool_id: pool_id.clone(),
                                protocol: pool.protocol_system.clone(),
                                fee: fee_default,
                                weight,
                                reserves,
                            };
                            self.graph.add_edge(from_idx, to_idx, edge);
                        }
                    }
                }
            }
        }
        self.pool_ids = new_pool_ids;
    }

    /// Backward-compatible: update from components without tracker (no weights/fees)
    pub fn update_from_components(&mut self, pools: &std::collections::HashMap<String, ProtocolComponent>) {
        // This function might need removal or adjustment as tracker info is now mandatory
        // For now, let\'s pass empty HashMaps, but this likely won\'t provide fees/weights.
        self.update_from_components_with_tracker(pools, &HashMap::new(), &HashMap::new());
    }

    /// Remove a pool and its associated edges from the graph.
    pub fn remove_pool(&mut self, pool_id: &str) {
        let edges_to_remove: Vec<_> = self.graph.edge_indices()
            .filter(|&e| self.graph.edge_weight(e).map_or(false, |ew| ew.pool_id == pool_id))
            .collect();
        for edge_index in edges_to_remove.into_iter().rev() {
            self.graph.remove_edge(edge_index);
        }
        self.pool_ids.remove(pool_id);
        // Optionally prune orphan nodes
        let orphan_nodes: Vec<_> = self.graph.node_indices()
            .filter(|&idx| self.graph.edges(idx).count() == 0)
            .collect();
        for idx in orphan_nodes.into_iter().rev() {
            let node = self.graph.node_weight(idx).cloned();
            self.graph.remove_node(idx);
            if let Some(node) = node {
                self.token_indices.shift_remove(&node.address);
            }
        }
    }

    /// Remove a token node and all its edges from the graph.
    pub fn remove_token(&mut self, token_address: &Bytes) {
        if let Some(&idx) = self.token_indices.get(token_address) {
            self.graph.remove_node(idx);
            self.token_indices.shift_remove(token_address);
        }
    }

    /// Update the weight of an edge between two tokens for a given pool.
    pub fn update_edge_weight(&mut self, from: &Bytes, to: &Bytes, pool_id: &str, new_weight: f64) {
        if let (Some(&from_idx), Some(&to_idx)) = (self.token_indices.get(from), self.token_indices.get(to)) {
            // First, collect the edge indices to update
            let edge_indices: Vec<_> = self.graph.edges_connecting(from_idx, to_idx)
                .filter(|edge| edge.weight().pool_id == pool_id)
                .map(|edge| edge.id())
                .collect();
            // Now, mutate the edges
            for edge_idx in edge_indices {
                if let Some(edge_w) = self.graph.edge_weight_mut(edge_idx) {
                    edge_w.weight = Some(new_weight);
                }
            }
        }
    }


    /// Get a node index by its token address
    pub fn get_node_index(&self, token_address: &Bytes) -> Option<NodeIndex> {
        self.token_indices.get(token_address).copied()
    }

    /// Get all token addresses in the graph
    pub fn get_all_token_addresses(&self) -> Vec<Bytes> {
        self.token_indices.keys().cloned().collect()
    }

    pub fn get_edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn get_node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn to_csr(&self) -> csr::CsrGraph {
        let node_count = self.graph.node_count();
        let mut indptr = vec![0; node_count + 1];
        let mut indices = Vec::new();
        let mut weights = Vec::new();

        // Build adjacency list representation first to sort neighbors by index
        let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); node_count];

        for edge_ref in self.graph.edge_references() {
            let source_idx = self.graph.to_index(edge_ref.source());
            let target_idx = self.graph.to_index(edge_ref.target());
            // Use a default weight if None, or skip if that's preferred
            let weight = edge_ref.weight().weight.unwrap_or(f64::INFINITY) as f32;
            adj[source_idx].push((target_idx as u32, weight));
        }

        // Sort neighbors and fill CSR arrays
        for i in 0..node_count {
            adj[i].sort_by_key(|&(neighbor_idx, _)| neighbor_idx);
            indptr[i+1] = indptr[i] + adj[i].len();
            for (neighbor_idx, weight) in &adj[i] {
                indices.push(*neighbor_idx);
                weights.push(*weight);
            }
        }

        csr::CsrGraph { indptr, indices, weights }
    }

    // Utility to convert Bytes to alloy_primitives::Address
    // This should ideally live elsewhere, like a utils module, if used more broadly.
    pub(crate) fn bytes_to_address(address: &Bytes) -> Result<Address, SimulationError> {
        let hex_str = address.to_string(); // Assuming Bytes can be converted to a hex string
        if hex_str.len() != 42 || !hex_str.starts_with("0x") {
            // FIXME: PANICKING! Unknown SimulationError variants from tycho_simulation.
            // Replace with proper error handling once tycho_simulation::protocol::errors::SimulationError is understood.
            panic!("Invalid address format: {} - Cannot construct tycho_simulation::protocol::errors::SimulationError", hex_str);
            // Example of how it *might* be if a suitable variant existed:
            // return Err(SimulationError::InvalidInput(format!("Invalid address format: {}", hex_str)));
        }
        // FIXME: PANICKING! Unknown SimulationError variants from tycho_simulation.
        // Replace with proper error handling once tycho_simulation::protocol::errors::SimulationError is understood.
        hex_str.parse::<Address>().map_err(|e| {
            panic!("Failed to parse address {}: {} - Cannot construct tycho_simulation::protocol::errors::SimulationError", hex_str, e);
            // Example of how it *might* be if a suitable variant existed:
            // SimulationError::ParseError(format!("Failed to parse address {}: {}", hex_str, e))
        })
    }
}

impl Default for TokenGraph {
    fn default() -> Self {
        Self::new()
    }
}


pub mod csr {
    // Compressed Sparse Row (CSR) graph representation for efficient pathfinding.
    // This is suitable for libraries like `pathfinding::directed::dijkstra::dijkstra_all`
    // which expect a graph where nodes are `usize` and edges are `(usize, Weight)`.

    #[derive(Clone)]
    pub struct CsrGraph {
        pub indptr: Vec<usize>,
        pub indices: Vec<u32>, // Node indices
        pub weights: Vec<f32>, // Edge weights
    }
} 