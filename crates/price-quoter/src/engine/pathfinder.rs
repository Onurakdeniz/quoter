//! Pathfinding algorithms (Delta-stepping, Yen\'s K-Shortest Paths).

use crate::engine::graph::TokenGraph; // Adjusted path
use petgraph::algo::dijkstra;
use petgraph::prelude::*;
use petgraph::visit::EdgeRef;
use tycho_simulation::tycho_common::Bytes;
use tracing::debug;
use std::sync::{Arc, RwLock}; // Added imports

/// Main pathfinding struct, operates on TokenGraph.
pub struct Pathfinder { // Removed lifetime 'a
    pub graph: Arc<RwLock<TokenGraph>>, // Changed to Arc<RwLock<TokenGraph>>
}

impl Pathfinder { // Removed lifetime 'a
    pub fn new(graph: Arc<RwLock<TokenGraph>>) -> Self { // Changed graph type
        Self { graph }
    }

    /// Find the best path (by weighted cost if available, else hops) from source to target token.
    pub fn best_path(&self, source: &Bytes, target: &Bytes) -> Option<Vec<NodeIndex>> {
        let graph_r = self.graph.read().unwrap(); // Acquire read lock
        if source == target {
            if let Some(&idx) = graph_r.token_indices.get(source) { // Use graph_r
                return Some(vec![idx]);
            }
            return None;
        }

        // If CSR edges exist (after recent update), prefer Δ-Stepping SSSP
        #[cfg(feature = "delta_sssp")]
        {
            let csr = graph_r.to_csr(); // Use graph_r
            let source_idx = *graph_r.token_indices.get(source)? as usize; // Use graph_r
            let target_idx = *graph_r.token_indices.get(target)? as usize; // Use graph_r
            let n = csr.indptr.len() - 1;
            let mut dist = vec![f32::INFINITY; n];
            let mut prev = vec![u32::MAX; n];
            let dirty_nodes: Vec<usize> = Vec::new();
            delta::sssp_parallel(&csr, source_idx, &dirty_nodes, &mut dist, &mut prev);
            if dist[target_idx].is_finite() {
                let mut node = target_idx as u32;
                let mut rev_path = Vec::new();
                while node != u32::MAX {
                    rev_path.push(NodeIndex::new(node as usize));
                    node = prev[node as usize];
                }
                rev_path.reverse();
                return Some(rev_path);
            }
        }

        // --- Existing Dijkstra fallback ---
        // Use Dijkstra\'s algorithm with edge weights if available
        let source_idx = *graph_r.token_indices.get(source)?; // Use graph_r
        let target_idx = *graph_r.token_indices.get(target)?; // Use graph_r
        let path_map = dijkstra(
            &graph_r.graph, // Use graph_r
            source_idx,
            Some(target_idx),
            |e| e.weight().weight.unwrap_or(1.0),
        );
        if let Some(_cost) = path_map.get(&target_idx) {
            let mut path = vec![target_idx];
            let mut current = target_idx;
            while current != source_idx {
                let pred = graph_r.graph // Use graph_r
                    .edges_directed(current, petgraph::Direction::Incoming)
                    .filter_map(|e| {
                        let n = e.source();
                        let w = e.weight().weight.unwrap_or(1.0);
                        let prev_cost = path_map.get(&n)?;
                        if (*prev_cost + w - path_map[&current]).abs() < 1e-8 {
                            Some(n)
                        } else {
                            None
                        }
                    })
                    .next();
                if let Some(pred_idx) = pred {
                    path.push(pred_idx);
                    current = pred_idx;
                } else {
                    break;
                }
            }
            path.reverse();
            Some(path)
        } else {
            None
        }
    }

    /// Find up to K shortest paths (by cumulative negative‑log cost) from source to target.
    /// This is a pragmatic BFS enumeration up to a depth limit of 6 hops – sufficient for most
    /// ERC‑20 routing scenarios – and orders the discovered paths by total weight ascending.
    ///
    /// NOTE: For production use a full Yen/K‑shortest implementation would be preferable, but
    /// this lightweight approach yields good routes while avoiding heavy dependencies.
    pub fn k_shortest_paths(&self, source: &Bytes, target: &Bytes, k: usize, max_depth: usize) -> Vec<Vec<NodeIndex>> {
        let graph_r = self.graph.read().unwrap(); // Acquire read lock
        debug!(source = %hex::encode(source), target = %hex::encode(target), k=%k, max_depth=%max_depth, "Finding k-shortest paths");
        let (source_idx, target_idx) = match (graph_r.token_indices.get(source), graph_r.token_indices.get(target)) {
            (Some(&s), Some(&t)) => (s, t),
            _ => {
                debug!("Source or target token not found in graph indices");
                return vec![];
            }
        };
        debug!(?source_idx, ?target_idx, "Found node indices");

        // Each stack entry holds (path_so_far, cumulative_cost)
        let mut stack: Vec<(Vec<NodeIndex>, f64)> = Vec::new();
        stack.push((vec![source_idx], 0.0));

        let mut results: Vec<(f64, Vec<NodeIndex>)> = Vec::new();
        let mut visited_paths = std::collections::HashSet::new(); // To avoid duplicate paths

        while let Some((path, cost)) = stack.pop() {
            // Debug: Print current path being processed
            // let current_path_str = path.iter().map(|idx| format!("{:?}", idx)).collect::<Vec<_>>().join(" -> ");
            // debug!(path=%current_path_str, cost=%cost, "Popped path from stack");

            let last = *path.last().unwrap();

            // If path already ends at target we\'re done – record and continue (do not expand further)
            if last == target_idx {
                // Avoid adding duplicate paths to results
                if visited_paths.insert(path.clone()) {
                    results.push((cost, path.clone()));
                }
                continue;
            }

            if path.len() - 1 >= max_depth { // len‑1 = number of edges
                // debug!(path=%current_path_str, "Depth limit reached");
                continue; // depth limit reached
            }

            // Expand neighbours
            for edge in graph_r.graph.edges(last) {
                let next = edge.target();
                // Avoid cycles by skipping nodes already in path
                if path.contains(&next) {
                    // debug!(next_node=?next, "Skipping cycle");
                    continue;
                }
                let edge_weight = edge.weight().weight.unwrap_or(1.0);
                let new_cost = cost + edge_weight;
                let mut new_path = path.clone();
                new_path.push(next);
                stack.push((new_path, new_cost));
                // debug!(next_path=%new_path.iter().map(|idx| format!("{:?}", idx)).collect::<Vec<_>>().join(" -> "), new_cost=%new_cost, "Pushed path to stack");
            }
        }

        debug!(num_paths_found=%results.len(), "Finished BFS path search");
        // Sort by total cost (smaller cost = better price) and truncate to k
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        // Debug: Print final selected paths
        debug!("Final top-k paths selected:");
        for (cost, path) in &results {
            let path_str = path.iter()
                .map(|idx| graph_r.graph.node_weight(*idx).map_or("?".to_string(), |n| n.symbol.clone()))
                .collect::<Vec<_>>().join(" -> ");
            debug!(cost=%cost, path=%path_str);
        }

        results.into_iter().map(|(_, p)| p).collect()
    }

    /// Enumerate ALL simple paths (no cycles) between source and target up to `max_depth` hops.
    /// Returns a vector of paths where each path is a list of NodeIndex representing the route.
    /// This exhaustive enumeration is useful when downstream logic (e.g. simulation) will score
    /// the paths more accurately than the heuristic edge weights used for k‑shortest search.
    /// NOTE: The number of returned paths can grow exponentially with depth, so callers should
    /// take care to keep `max_depth` small (typically 3‑4 for ERC‑20 routing).
    pub fn enumerate_paths(&self, source: &Bytes, target: &Bytes, max_depth: usize) -> Vec<Vec<NodeIndex>> {
        let graph_r = self.graph.read().unwrap(); // Acquire read lock
        let (source_idx, target_idx) = match (graph_r.token_indices.get(source), graph_r.token_indices.get(target)) {
            (Some(&s), Some(&t)) => (s, t),
            _ => return vec![],
        };

        // Stack holds the current partial path to explore
        let mut stack: Vec<Vec<NodeIndex>> = vec![vec![source_idx]];
        let mut results: Vec<Vec<NodeIndex>> = Vec::new();

        while let Some(path) = stack.pop() {
            let last = *path.last().unwrap();
            if last == target_idx {
                results.push(path.clone());
                continue;
            }
            if path.len() - 1 >= max_depth { // reached hop limit (edges)
                continue;
            }
            for edge in graph_r.graph.edges(last) {
                let next = edge.target();
                if path.contains(&next) {
                    continue; // avoid cycles
                }
                let mut new_path = path.clone();
                new_path.push(next);
                stack.push(new_path);
            }
        }

        results
    }

    /// Compute the total cost of a path, considering slippage and fee if available.
    pub fn path_cost_with_slippage_fee(&self, path: &[NodeIndex]) -> Option<f64> {
        let graph_r = self.graph.read().unwrap(); // Acquire read lock
        let mut total_cost = 0.0;
        for w in path.windows(2) {
            let from = w[0];
            let to = w[1];
            let mut edge_cost = None;
            for edge in graph_r.graph.edges_connecting(from, to) {
                let ew = edge.weight();
                // Use weight if available, else fallback to 1.0
                edge_cost = Some(ew.weight.unwrap_or(1.0));
                // Optionally, add fee/slippage logic here
                // e.g., edge_cost = Some(compute_effective_weight(ew));
                break;
            }
            total_cost += edge_cost.unwrap_or(1.0);
        }
        Some(total_cost)
    }

    /// Parallel SSSP (Single Source Shortest Path) stub for future implementation.
    pub fn parallel_sssp(&self, _source: &Bytes) {
        // Placeholder: In production, use rayon or async tasks to parallelize SSSP
        // For now, this is a stub.
        unimplemented!("Parallel SSSP is not yet implemented");
    }

    // TODO: Add slippage/fee/price-aware weights, parallel SSSP, incremental updates, etc.
}

// New delta module implementing parallel Δ-Stepping SSSP
#[cfg(feature = "delta_sssp")]
pub mod delta {
    use super::*;
    // use rayon if desired; we implement sequential fallback
    // use rayon::prelude::*;

    const DEFAULT_DELTA: f32 = 1.0;

    // Public SSSP function
    pub fn sssp_parallel(
        csr: &CsrGraph,
        source: usize,
        _dirty_nodes: &[usize], // Currently unused, for future incremental updates
        dist: &mut [f32],
        prev: &mut [u32],
    ) {
        let n = csr.indptr.len() - 1;
        dist.fill(f32::INFINITY);
        prev.fill(u32::MAX); // Use u32::MAX to denote no predecessor

        dist[source] = 0.0;
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new()];
        buckets[0].push(source);
        let mut current_bucket_idx = 0;

        while current_bucket_idx < buckets.len() {
            if buckets[current_bucket_idx].is_empty() {
                current_bucket_idx += 1;
                continue;
            }

            let mut req: Vec<(usize, f32)> = Vec::new(); // Store (neighbor, weight)
            let mut light_edges_nodes: Vec<usize> = Vec::new();
            let mut heavy_edges_nodes: Vec<usize> = Vec::new();

            // Phase 1: Relax light edges for nodes in current bucket
            for &u in &buckets[current_bucket_idx] {
                if dist[u] < (current_bucket_idx as f32 * DEFAULT_DELTA) { continue; } // Node already settled earlier

                for i in csr.indptr[u]..csr.indptr[u + 1] {
                    let v = csr.indices[i] as usize;
                    let weight_uv = csr.weights[i];
                    if weight_uv <= DEFAULT_DELTA { // Light edge
                        if dist[u] + weight_uv < dist[v] {
                            dist[v] = dist[u] + weight_uv;
                            prev[v] = u as u32;
                            req.push((v, dist[v]));
                            light_edges_nodes.push(v);
                        }
                    } else { // Heavy edge
                        if dist[u] + weight_uv < dist[v] {
                            dist[v] = dist[u] + weight_uv;
                            prev[v] = u as u32;
                            req.push((v, dist[v])); // Keep track for potential re-bucketing
                            heavy_edges_nodes.push(v); // For later processing
                        }
                    }
                }
            }

            buckets[current_bucket_idx].clear(); // Clear current bucket

            // Re-bucket nodes relaxed by light edges
            for node in light_edges_nodes {
                let new_bucket_for_node = (dist[node] / DEFAULT_DELTA).floor() as usize;
                if new_bucket_for_node >= buckets.len() {
                    buckets.resize(new_bucket_for_node + 1, Vec::new());
                }
                if !buckets[new_bucket_for_node].contains(&node) { // Avoid duplicates
                    buckets[new_bucket_for_node].push(node);
                }
            }

            // Phase 2: Relax heavy edges (iteratively if needed, but simple version here)
            // This part might need refinement for correctness with many heavy edges or specific graph structures.
            // For now, we re-bucket nodes affected by heavy edges directly.
            for node in heavy_edges_nodes {
                let new_bucket_for_node = (dist[node] / DEFAULT_DELTA).floor() as usize;
                 if new_bucket_for_node >= buckets.len() {
                    buckets.resize(new_bucket_for_node + 1, Vec::new());
                }
                if !buckets[new_bucket_for_node].contains(&node) { // Avoid duplicates
                     buckets[new_bucket_for_node].push(node);
                }
            }

            // If no new nodes were added to current or future buckets, advance to next non-empty or finish.
            let mut all_future_buckets_empty = true;
            for b_idx in current_bucket_idx..buckets.len() {
                if !buckets[b_idx].is_empty() {
                    current_bucket_idx = b_idx;
                    all_future_buckets_empty = false;
                    break;
                }
            }
            if all_future_buckets_empty {
                break; // All done
            }
        }
    }
} 