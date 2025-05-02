//! Graph and pathfinding tests for price-quoter.

use price_quoter::graph::TokenGraph;
use tycho_simulation::protocol::models::{ProtocolComponent, Token as ProtoToken};
use std::collections::HashMap;
use tycho_simulation::tycho_common::Bytes;

#[test]
fn test_graph_construction() {
    let mut graph = TokenGraph::new();
    let mut pools = HashMap::new();
    let pool_id = "pool1".to_string();
    let token_a = ProtoToken { address: Bytes::from([1u8; 20]), symbol: "A".to_string(), decimals: 18 };
    let token_b = ProtoToken { address: Bytes::from([2u8; 20]), symbol: "B".to_string(), decimals: 18 };
    let comp = ProtocolComponent { tokens: vec![token_a, token_b], protocol_system: "uniswap_v2".to_string(), ..Default::default() };
    pools.insert(pool_id.clone(), comp);
    graph.update_from_components(&pools);
    assert!(graph.graph.node_count() >= 2);
    assert!(graph.graph.edge_count() >= 2);
} 