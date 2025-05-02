//! Pathfinding algorithms (Delta-stepping, Yen's K-Shortest Paths).

use crate::graph::TokenGraph;
use petgraph::algo::{astar, dijkstra};
use petgraph::prelude::*;
use petgraph::visit::EdgeRef;
use tycho_simulation::tycho_common::Bytes;
use tracing::debug;
use crate::graph::csr::CsrGraph;

/// Main pathfinding struct, operates on TokenGraph.
pub struct Pathfinder<'a> {
    pub graph: &'a TokenGraph,
}

impl<'a> Pathfinder<'a> {
    pub fn new(graph: &'a TokenGraph) -> Self {
        Self { graph }
    }

    /// Find the best path (by weighted cost if available, else hops) from source to target token.
    pub fn best_path(&self, source: &Bytes, target: &Bytes) -> Option<Vec<NodeIndex>> {
        // Fast path: if source == target return single-node path
        if source == target {
            if let Some(&idx) = self.graph.token_indices.get(source) {
                return Some(vec![idx]);
            }
            return None;
        }

        // If CSR edges exist (after recent update), prefer Δ-Stepping SSSP
        #[cfg(feature = "delta_sssp")]
        {
            let csr = self.graph.to_csr();
            let source_idx = *self.graph.token_indices.get(source)? as usize;
            let target_idx = *self.graph.token_indices.get(target)? as usize;
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
        // Use Dijkstra's algorithm with edge weights if available
        let source_idx = *self.graph.token_indices.get(source)?;
        let target_idx = *self.graph.token_indices.get(target)?;
        let path_map = dijkstra(
            &self.graph.graph,
            source_idx,
            Some(target_idx),
            |e| e.weight().weight.unwrap_or(1.0),
        );
        if let Some(_cost) = path_map.get(&target_idx) {
            let mut path = vec![target_idx];
            let mut current = target_idx;
            while current != source_idx {
                let pred = self.graph.graph
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
        debug!(source = %hex::encode(source), target = %hex::encode(target), k=%k, max_depth=%max_depth, "Finding k-shortest paths");
        let (source_idx, target_idx) = match (self.graph.token_indices.get(source), self.graph.token_indices.get(target)) {
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

            // If path already ends at target we're done – record and continue (do not expand further)
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
            for edge in self.graph.graph.edges(last) {
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
                .map(|idx| self.graph.graph.node_weight(*idx).map_or("?".to_string(), |n| n.symbol.clone()))
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
        let (source_idx, target_idx) = match (self.graph.token_indices.get(source), self.graph.token_indices.get(target)) {
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
            for edge in self.graph.graph.edges(last) {
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
        let mut total_cost = 0.0;
        for w in path.windows(2) {
            let from = w[0];
            let to = w[1];
            let mut edge_cost = None;
            for edge in self.graph.graph.edges_connecting(from, to) {
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

    pub fn sssp_parallel(
        csr: &CsrGraph,
        source: usize,
        _dirty_nodes: &[usize],
        dist: &mut [f32],
        prev: &mut [u32],
    ) {
        let n = dist.len();
        dist.fill(f32::INFINITY);
        prev.fill(u32::MAX);
        dist[source] = 0.0;

        let mut buckets: Vec<Vec<usize>> = vec![vec![source]];
        let mut b = 0usize;
        let delta = DEFAULT_DELTA;

        while b < buckets.len() {
            if buckets[b].is_empty() {
                b += 1;
                continue;
            }
            let mut frontier = std::mem::take(&mut buckets[b]);
            // Process light edges
            let mut req: Vec<usize> = Vec::new();
            while let Some(u) = frontier.pop() {
                let du = dist[u];
                for idx in csr.indptr[u]..csr.indptr[u + 1] {
                    let v = csr.indices[idx] as usize;
                    let w = csr.weights[idx];
                    let alt = du + w;
                    if alt < dist[v] {
                        dist[v] = alt;
                        prev[v] = u as u32;
                        if w < delta {
                            frontier.push(v);
                        } else {
                            let bi = (alt / delta).floor() as usize;
                            if bi >= buckets.len() {
                                buckets.resize_with(bi + 1, Vec::new);
                            }
                            buckets[bi].push(v);
                        }
                    }
                }
            }
            // if frontier empty now, move to next bucket
            if buckets[b].is_empty() {
                b += 1;
            }
        }
    }
}