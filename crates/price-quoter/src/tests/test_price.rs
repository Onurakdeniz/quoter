//! Price computation tests for price-quoter.

use price_quoter::price_engine::PriceEngine;
use price_quoter::component_tracker::ComponentTracker;
use price_quoter::graph::TokenGraph;
use tycho_simulation::tycho_common::Bytes;

#[test]
fn test_price_computation() {
    let tracker = ComponentTracker::new();
    let graph = TokenGraph::new();
    let engine = PriceEngine::new(tracker, &graph);
    let token_a = Bytes::from([1u8; 20]);
    let token_b = Bytes::from([2u8; 20]);
    let quote = engine.quote(&token_a, &token_b, 1000);
    assert!(quote.amount_out.is_none()); // No pools, so no route
} 