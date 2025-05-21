//! Price computation tests for price-quoter.

use price_quoter::config::AppConfig;
use price_quoter::data_management::component_tracker::{ComponentTracker, PoolComponent, TokenComponent};
use price_quoter::data_management::cache::QuoteCache;
use price_quoter::engine::graph::{TokenGraph, PoolEdge, TokenType, GraphNode};
use price_quoter::engine::PriceEngine;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tycho_simulation::tycho_common::{Bytes, Address};
use tycho_simulation::protocol::common::{SimInput, SimOutput, ProtocolSim, SimulationResult};
use num_bigint::BigUint;
use rust_decimal::Decimal;
use async_trait::async_trait;

// --- Mock ProtocolSim Implementation ---
#[derive(Clone, Debug)]
struct MockSim {
    expected_output_18_decimals: u128, // This is the value as if it had 18 decimals
}

#[async_trait]
impl ProtocolSim for MockSim {
    async fn get_amount_out(&self, _input: SimInput, _token_in_meta: Option<tycho_simulation::TokenMetadata>, _token_out_meta: Option<tycho_simulation::TokenMetadata>) -> SimulationResult {
        SimulationResult {
            amount: BigUint::from(self.expected_output_18_decimals), // Simulate output scaled to 18 decimals
            gas_estimate: Some(BigUint::from(0u64)), // No gas cost for simplicity
            error: None,
        }
    }

    fn box_clone(&self) -> Box<dyn ProtocolSim> {
        Box::new(self.clone())
    }

    fn get_type(&self) -> String {
        "MockSim".to_string()
    }
}

#[tokio::test]
async fn test_output_amount_scaling() {
    // 1. Setup
    let tracker = ComponentTracker::new();
    let graph_arc = Arc::new(RwLock::new(TokenGraph::new()));
    let cache_arc = Arc::new(RwLock::new(QuoteCache::new()));

    let mut config = AppConfig::default(); // Using default and overriding specifics
    config.gas_price_gwei = Some(0); // Zero gas price
    config.avg_gas_units_per_swap = Some(0); // Zero gas units per swap

    // Define WETH (18 decimals) and USDC (6 decimals)
    let weth_address_str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
    let usdc_address_str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";

    let weth_address_bytes = Bytes::from_hex(weth_address_str).unwrap();
    let usdc_address_bytes = Bytes::from_hex(usdc_address_str).unwrap();

    let weth_token = TokenComponent {
        address: weth_address_bytes.clone(),
        name: "Wrapped Ether".to_string(),
        symbol: "WETH".to_string(),
        decimals: 18,
        chain_id: 1,
        token_type: TokenType::ERC20,
        logo_url: None,
        tvl_usd: None,
        price_usd: None,
    };
    let usdc_token = TokenComponent {
        address: usdc_address_bytes.clone(),
        name: "USD Coin".to_string(),
        symbol: "USDC".to_string(),
        decimals: 6,
        chain_id: 1,
        token_type: TokenType::ERC20,
        logo_url: None,
        tvl_usd: None,
        price_usd: None,
    };

    tracker.all_tokens.write().unwrap().insert(weth_address_bytes.clone(), weth_token.clone());
    tracker.all_tokens.write().unwrap().insert(usdc_address_bytes.clone(), usdc_token.clone());

    // Add WETH and USDC nodes to the TokenGraph
    let weth_node_idx = graph_arc.write().unwrap().add_token_node(&weth_token);
    let usdc_node_idx = graph_arc.write().unwrap().add_token_node(&usdc_token);
    
    // Create a mock pool component for WETH -> USDC
    let mock_pool_id = "mock_pool_weth_usdc".to_string();
    let mock_pool_component = PoolComponent {
        id: mock_pool_id.clone(),
        address: Bytes::from_hex("0x1234567890123456789012345678901234567890").unwrap(), // Mock pool address
        protocol_id: "mock_protocol".to_string(),
        chain_id: 1,
        tokens: vec![weth_address_bytes.clone(), usdc_address_bytes.clone()],
        fee: Some(0.0), // Zero fee for simplicity
        tvl_usd: None,
        state_is_functional: true,
        last_update_block: 0,
    };
    tracker.all_pools.write().unwrap().insert(mock_pool_id.clone(), mock_pool_component.clone());

    // Create and register a mock ProtocolSim
    // Simulate that the protocol outputs 2500 units, but scaled as if it were an 18-decimal token
    let simulated_gross_output_18_decimals = 2500 * 10u128.pow(18); 
    let mock_sim = MockSim { expected_output_18_decimals: simulated_gross_output_18_decimals };
    tracker.pool_states.write().unwrap().insert(mock_pool_id.clone(), Box::new(mock_sim));

    // Add edge to TokenGraph
    let pool_edge = PoolEdge {
        pool_id: mock_pool_id.clone(),
        token_in_idx: weth_node_idx,
        token_out_idx: usdc_node_idx,
        fee: Some(0.0), // Zero fee
        protocol_type: "mock".to_string(),
        sim_input_props: None, // Not strictly needed for this test with mock sim
    };
    graph_arc.write().unwrap().graph.add_edge(weth_node_idx, usdc_node_idx, pool_edge);
    
    // Instantiate PriceEngine
    let engine = PriceEngine::from_config(
        tracker.clone(),
        graph_arc.clone(),
        cache_arc.clone(),
        &config,
    );
    // Manually update graph from tracker state as it's not done automatically in from_config
    engine.update_graph_from_tracker_state();


    // 2. Execution
    let input_weth_amount = 10u128.pow(18); // 1 WETH
    let quote = engine.quote_multi(&weth_address_bytes, &usdc_address_bytes, input_weth_amount, 1, None).await;

    // 3. Assertion
    assert!(quote.amount_out.is_some(), "Expected a quote, but got None");
    assert!(quote.gross_amount_out.is_some(), "Expected gross_amount_out, but got None");

    // Expected output: 2500 USDC. Since USDC has 6 decimals, this is 2500 * 10^6.
    let expected_output_usdc_units = 2500 * 10u128.pow(6);

    // Check gross_amount_out (before gas/protocol fees)
    // This should directly reflect the (rescaled) simulation output
    assert_eq!(
        quote.gross_amount_out.unwrap(),
        expected_output_usdc_units,
        "Gross amount out scaling is incorrect. Expected {}, got {}",
        expected_output_usdc_units,
        quote.gross_amount_out.unwrap()
    );

    // Check amount_out (net amount)
    // Since fees and gas are zero, net amount should equal gross amount
    assert_eq!(
        quote.amount_out.unwrap(),
        expected_output_usdc_units,
        "Net amount out scaling is incorrect. Expected {}, got {}",
        expected_output_usdc_units,
        quote.amount_out.unwrap()
    );

    // Optional: Verify path details if needed
    assert_eq!(quote.path_details.len(), 1, "Expected one path in details");
    let path_detail = &quote.path_details[0];
    assert_eq!(path_detail.gross_amount_out.unwrap(), expected_output_usdc_units, "Path detail gross amount out scaling incorrect");
    assert_eq!(path_detail.amount_out.unwrap(), expected_output_usdc_units, "Path detail net amount out scaling incorrect");
    assert_eq!(path_detail.route.len(), 2, "Route should have 2 tokens");
    assert_eq!(path_detail.route[0], weth_address_bytes, "Route token_in mismatch");
    assert_eq!(path_detail.route[1], usdc_address_bytes, "Route token_out mismatch");
    assert_eq!(path_detail.pools.len(), 1, "Path should have 1 pool");
    assert_eq!(path_detail.pools[0], mock_pool_id, "Pool ID mismatch in path");
}


#[test]
fn test_price_computation() {
    let tracker = ComponentTracker::new();
    let graph = Arc::new(RwLock::new(TokenGraph::new())); // Needs to be Arc<RwLock<>>
    let cache = Arc::new(RwLock::new(QuoteCache::new()));
    let config = AppConfig::default();
    let engine = PriceEngine::from_config(tracker, graph, cache, &config); // Updated to use from_config

    let token_a_addr = Bytes::from_hex("0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").unwrap();
    let token_b_addr = Bytes::from_hex("0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB").unwrap();
    
    // Minimal setup for tokens in tracker if needed for quote_multi to find paths, even if empty
    let token_a_comp = TokenComponent { address: token_a_addr.clone(), name:"TokenA".into(), symbol:"TA".into(), decimals:18, chain_id:1, token_type:TokenType::ERC20, logo_url:None, tvl_usd:None, price_usd:None };
    let token_b_comp = TokenComponent { address: token_b_addr.clone(), name:"TokenB".into(), symbol:"TB".into(), decimals:18, chain_id:1, token_type:TokenType::ERC20, logo_url:None, tvl_usd:None, price_usd:None };
    engine.tracker.all_tokens.write().unwrap().insert(token_a_addr.clone(), token_a_comp);
    engine.tracker.all_tokens.write().unwrap().insert(token_b_addr.clone(), token_b_comp);
    // Also add to graph
    let mut graph_w = engine.graph.write().unwrap();
    graph_w.add_token_node(&engine.tracker.all_tokens.read().unwrap().get(&token_a_addr).unwrap());
    graph_w.add_token_node(&engine.tracker.all_tokens.read().unwrap().get(&token_b_addr).unwrap());
    drop(graph_w);


    let quote = futures::executor::block_on(engine.quote(&token_a_addr, &token_b_addr, 1000, None)); // quote is async
    assert!(quote.amount_out.is_none()); // No pools, so no route
}