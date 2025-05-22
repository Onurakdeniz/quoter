//! Price computation tests for price-quoter.

use price_quoter::price_engine::PriceEngine;
use price_quoter::component_tracker::ComponentTracker;
use price_quoter::graph::{TokenGraph, PoolEdge, GraphEdgeType, TokenType, TokenData}; // Added TokenData
use price_quoter::config::AppConfig;
use price_quoter::engine::quoting::PriceQuote; // Added PriceQuote for direct use
use tycho_simulation::tycho_common::Bytes;
use tycho_simulation::tycho_components::pool::{PoolComponent, PoolType, ConstantProductState};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use rust_decimal::Decimal;
use std::str::FromStr;
use petgraph::graph::{NodeIndex}; // Removed EdgeIndex as it's not directly used in this setup for now

// Helper function to create Bytes from hex string
fn bytes_from_hex(s: &str) -> Bytes {
    Bytes::from_str(s).unwrap()
}

// Define some common token addresses for tests
lazy_static::lazy_static! {
    static ref TKA: Bytes = bytes_from_hex("0xA00000000000000000000000000000000000000A");
    static ref TKB: Bytes = bytes_from_hex("0xB00000000000000000000000000000000000000B");
    static ref TKC: Bytes = bytes_from_hex("0xC00000000000000000000000000000000000000C");
    static ref TKD: Bytes = bytes_from_hex("0xD00000000000000000000000000000000000000D");
    static ref NUM: Bytes = bytes_from_hex("0xNUME000000000000000000000000000000000NUME"); // Numeraire
}

const DEFAULT_DECIMALS: u32 = 18;
const TOKEN_UNIT: u128 = 1_000_000_000_000_000_000; // 1 token with 18 decimals

struct TestSetup {
    engine: PriceEngine,
    _graph: Arc<RwLock<TokenGraph>>, // Keep graph for potential inspection, mark unused if not directly used
    _tracker: Arc<RwLock<ComponentTracker>>, 
    _token_indices: HashMap<Bytes, NodeIndex>, // Keep for debugging or future assertions
}

// Define a type alias for pool definitions for clarity
type PoolDef = (Bytes, Bytes, String, f64, u128, u128);

fn setup_test_environment(
    mut main_pools_data: Vec<PoolDef>, 
    numeraire_pools_data_opt: Option<Vec<PoolDef>>,
    app_config_customizer: Option<fn(&mut AppConfig)>,
) -> TestSetup {
    if let Some(num_pools) = numeraire_pools_data_opt {
        main_pools_data.extend(num_pools);
    }

    let tracker_arc = Arc::new(RwLock::new(ComponentTracker::new()));
    let graph_arc = Arc::new(RwLock::new(TokenGraph::new()));

    let mut default_app_config = AppConfig {
        tycho_url: "".to_string(),
        chain: "".to_string(),
        tycho_api_key: "".to_string(),
        max_hops: Some(3),
        numeraire_token: Some(NUM.clone()),
        probe_depth: Some(TOKEN_UNIT), 
        gas_price_gwei: Some(30),
        avg_gas_units_per_swap: Some(150_000),
        protocol_fee_bps: Some(0.0), 
        native_token_address: Some(NUM.clone()), 
        tokens_file: None,
        price_history_file: None,
        infura_api_key: None,
        tvl_threshold: 0.0,
    };

    if let Some(customizer) = app_config_customizer {
        customizer(&mut default_app_config);
    }
    
    let mut token_indices_map = HashMap::new();

    {
        let mut tracker_w = tracker_arc.write().unwrap();
        let mut graph_w = graph_arc.write().unwrap();
        let mut all_tokens_cache = tracker_w.all_tokens.write().unwrap(); // Cache for all_tokens

        let mut tokens_to_process = pools_data.iter().flat_map(|pd| vec![pd.0.clone(), pd.1.clone()]).collect::<std::collections::HashSet<_>>();
        if let Some(num_token) = &default_app_config.numeraire_token {
            tokens_to_process.insert(num_token.clone());
        }
         if let Some(native_token) = &default_app_config.native_token_address {
            tokens_to_process.insert(native_token.clone());
        }


        for token_bytes in tokens_to_process {
            if !all_tokens_cache.contains_key(&token_bytes) {
                let token_data = TokenData {
                    address: token_bytes.clone(),
                    symbol: format!("TK{:X}", token_bytes.as_ref()[0]), 
                    name: format!("Token {:X}", token_bytes.as_ref()[0]),
                    decimals: DEFAULT_DECIMALS, 
                    total_supply: Some(Decimal::from(1_000_000_000) * Decimal::from(TOKEN_UNIT)),
                    token_type: TokenType::Erc20,
                    logo_uri: None,
                };
                all_tokens_cache.insert(token_bytes.clone(), Arc::new(token_data.clone()));
                let node_idx = graph_w.ensure_node(&token_bytes, Some(token_data.decimals), Some(token_data.symbol.clone()), Some(token_data.name.clone()));
                token_indices_map.insert(token_bytes.clone(), node_idx);
            }
        }

        let mut all_pools_cache = tracker_w.all_pools.write().unwrap(); // Cache for all_pools

        for (t1_addr, t2_addr, pool_id, fee_percent, res1, res2) in pools_data {
            let pool_component = PoolComponent {
                address: bytes_from_hex(&format!("0xPOOL{:0<36}", pool_id)), 
                pool_id: pool_id.clone(),
                pool_type: PoolType::ConstantProduct,
                tokens: vec![t1_addr.clone(), t2_addr.clone()],
                state: Some(serde_json::to_value(ConstantProductState {
                    reserve0: Decimal::from_u128(res1).unwrap_or_default(),
                    reserve1: Decimal::from_u128(res2).unwrap_or_default(),
                    k_last: None,
                }).unwrap()),
                fee: Some(fee_percent), 
                last_updated_block: 1, 
                tvl_usd: Some( (Decimal::from_u128(res1).unwrap_or_default() + Decimal::from_u128(res2).unwrap_or_default()) / Decimal::from(TOKEN_UNIT) * Decimal::from(1000) ), // Dummy TVL
                ..Default::default()
            };
            all_pools_cache.insert(pool_id.clone(), Arc::new(pool_component.clone()));
            graph_w.add_pool_edge(&t1_addr, &t2_addr, &pool_component, GraphEdgeType::Dex);
        }
    } 

    let engine = PriceEngine::from_config(
        tracker_arc.read().unwrap().clone(), 
        graph_arc.clone(),
        Arc::new(RwLock::new(price_quoter::data_management::cache::QuoteCache::new())),
        &default_app_config,
    );
    
    engine.update_graph_from_tracker_state();


    TestSetup { engine, _graph: graph_arc, _tracker: tracker_arc, _token_indices: token_indices_map }
}

#[allow(dead_code)] // Keep for potential future use or direct struct comparison if needed
fn empty_price_quote_for_assertions() -> PriceQuote {
    PriceQuote {
        amount_out: None, route: vec![], price_impact_bps: None, mid_price: None,
        slippage_bps: None, fee_bps: None, protocol_fee_in_token_out: None,
        gas_estimate: None, path_details: vec![], gross_amount_out: None,
        spread_bps: None, depth_metrics: None, cache_block: None,
        is_split_trade: false, num_split_paths: 0,
    }
}

#[tokio::test]
async fn test_basic_engine_init_no_paths() {
    let setup = setup_test_environment(vec![], None);
    let quote = setup.engine.quote_multi(&TKA, &TKB, TOKEN_UNIT, 1, None).await;
    assert!(quote.amount_out.is_none());
    assert_eq!(quote.is_split_trade, false);
    assert_eq!(quote.num_split_paths, 0);
    assert!(quote.path_details.is_empty());
}

#[tokio::test]
async fn test_quote_multi_successful_split_two_paths() {
    let large_reserve = TOKEN_UNIT * 1000; 
    let fee_val = 0.003; // 0.3%
    let pools = vec![
        (TKA.clone(), TKB.clone(), "P1".to_string(), fee_val, large_reserve, large_reserve),
        (TKA.clone(), TKB.clone(), "P2".to_string(), fee_val, large_reserve, large_reserve),
    ];
    let setup = setup_test_environment(pools, None);
    let amount_in_val: u128 = TOKEN_UNIT; // 1 TKA

    let quote = setup.engine.quote_multi(&TKA, &TKB, amount_in_val, 2, None).await;

    assert!(quote.amount_out.is_some(), "Should find a quote. Amount out: {:?}", quote.amount_out);
    assert_eq!(quote.is_split_trade, true, "Should be a split trade");
    assert_eq!(quote.num_split_paths, 2, "Should split across 2 paths");
    
    let input_per_split_dec = Decimal::from_u128(amount_in_val / 2).unwrap_or_default() / Decimal::from(TOKEN_UNIT);
    let fee_dec = Decimal::from_f64(fee_val).unwrap_or_default();
    let expected_out_one_split_dec = input_per_split_dec * (Decimal::ONE - fee_dec);
    let expected_total_out_dec = (expected_out_one_split_dec * Decimal::TWO).round_dp(DEFAULT_DECIMALS);
    
    let actual_total_out_dec = Decimal::from_i128_with_scale(quote.amount_out.unwrap() as i128, DEFAULT_DECIMALS);
    let tolerance = Decimal::from_str("0.0000000000000001").unwrap(); // For minor simulation vs direct math diffs
    assert!((actual_total_out_dec - expected_total_out_dec).abs() < tolerance,
            "Total amount out {} is not close to expected {}. Diff: {}", 
            actual_total_out_dec, expected_total_out_dec, (actual_total_out_dec - expected_total_out_dec).abs());

    assert_eq!(quote.path_details.len(), 2, "Should have 2 paths in details");
    for pd in &quote.path_details {
        assert!(pd.amount_out.is_some());
        assert_eq!(pd.input_amount.unwrap(), amount_in_val / 2, "Input amount for split path is incorrect");
    }
}

#[tokio::test]
async fn test_quote_multi_fallback_single_available_path() {
    let large_reserve = TOKEN_UNIT * 1000;
    let fee_val = 0.003;
    let pools = vec![
        (TKA.clone(), TKB.clone(), "P1".to_string(), fee_val, large_reserve, large_reserve),
    ];
    let setup = setup_test_environment(pools, None);
    let amount_in_val: u128 = TOKEN_UNIT; 

    let quote = setup.engine.quote_multi(&TKA, &TKB, amount_in_val, 2, None).await; 

    assert!(quote.amount_out.is_some(), "Should find a quote");
    assert_eq!(quote.is_split_trade, false, "Should not be a split trade (only 1 actual path)");
    assert_eq!(quote.num_split_paths, 1, "Should use 1 path"); 
    assert_eq!(quote.path_details.len(), 1, "Should have 1 path in details");
    assert_eq!(quote.path_details[0].input_amount.unwrap(), amount_in_val, "Input amount for single path should be full amount_in");

    let amount_in_dec = Decimal::from_u128(amount_in_val).unwrap_or_default() / Decimal::from(TOKEN_UNIT);
    let fee_dec = Decimal::from_f64(fee_val).unwrap_or_default();
    let expected_total_out_dec = (amount_in_dec * (Decimal::ONE - fee_dec)).round_dp(DEFAULT_DECIMALS);
    let actual_total_out_dec = Decimal::from_i128_with_scale(quote.amount_out.unwrap() as i128, DEFAULT_DECIMALS);
    let tolerance = Decimal::from_str("0.0000000000000001").unwrap();
    assert!((actual_total_out_dec - expected_total_out_dec).abs() < tolerance,
            "Total amount out {} is not close to expected {}. Diff: {}", 
            actual_total_out_dec, expected_total_out_dec, (actual_total_out_dec - expected_total_out_dec).abs());
}


#[tokio::test]
async fn test_quote_multi_zero_input_amount() {
    let large_reserve = TOKEN_UNIT * 1000;
    let pools = vec![
        (TKA.clone(), TKB.clone(), "P1".to_string(), 0.003, large_reserve, large_reserve),
    ];
    let setup = setup_test_environment(pools, None);
    let amount_in: u128 = 0;

    let quote = setup.engine.quote_multi(&TKA, &TKB, amount_in, 2, None).await;

    assert_eq!(quote.amount_out, Some(0), "Amount out should be Some(0) for zero input");
    assert_eq!(quote.gross_amount_out, Some(0));
    assert_eq!(quote.is_split_trade, false, "Should not be a split trade for zero input");
    
    if quote.path_details.is_empty() {
         assert_eq!(quote.num_split_paths, 0, "Num split paths should be 0 if no path details");
    } else {
        assert_eq!(quote.num_split_paths, 1, "Num split paths should be 1 if path detail exists for 0 input");
        assert_eq!(quote.path_details.len(), 1);
        assert_eq!(quote.path_details[0].amount_out, Some(0));
        assert_eq!(quote.path_details[0].input_amount, Some(0));
    }
}

#[tokio::test]
async fn test_quote_multi_fallback_ksplit_greater_than_available() {
    // k_split is hardcoded to 3 in quote_multi. We have 2 non-overlapping paths.
    let large_reserve = TOKEN_UNIT * 1000;
    let fee_val = 0.003;
    let pools = vec![
        (TKA.clone(), TKB.clone(), "P1".to_string(), fee_val, large_reserve, large_reserve),
        (TKA.clone(), TKB.clone(), "P2".to_string(), fee_val, large_reserve, large_reserve),
    ];
    let setup = setup_test_environment(pools, None);
    let amount_in_val: u128 = TOKEN_UNIT;

    // quote_multi's internal k_split is 3, but only 2 non-overlapping paths are available.
    // The `k` parameter to quote_multi (here 2) is for final truncation of path_details.
    let quote = setup.engine.quote_multi(&TKA, &TKB, amount_in_val, 2, None).await;

    assert!(quote.amount_out.is_some(), "Should find a quote");
    // Even if k_split (internal) is 3, if only 2 paths are found, it should use those 2.
    assert_eq!(quote.is_split_trade, true, "Should be a split trade as 2 paths were used");
    assert_eq!(quote.num_split_paths, 2, "Should split across the 2 available non-overlapping paths");
    // path_details is truncated by the `k` parameter of quote_multi.
    assert_eq!(quote.path_details.len(), 2, "Path details should contain the 2 paths used for splitting");
}

#[tokio::test]
async fn test_quote_multi_three_non_overlapping_paths_split() {
    let large_reserve = TOKEN_UNIT * 1000;
    let fee_val = 0.003;
    let pools = vec![
        (TKA.clone(), TKB.clone(), "P1".to_string(), fee_val, large_reserve, large_reserve),
        (TKA.clone(), TKB.clone(), "P2".to_string(), fee_val, large_reserve, large_reserve),
        (TKA.clone(), TKB.clone(), "P3".to_string(), fee_val, large_reserve, large_reserve),
    ];
    let main_pools = vec![
        (TKA.clone(), TKB.clone(), "P1".to_string(), fee_val, large_reserve, large_reserve),
        (TKA.clone(), TKB.clone(), "P2".to_string(), fee_val, large_reserve, large_reserve),
        (TKA.clone(), TKB.clone(), "P3".to_string(), fee_val, large_reserve, large_reserve),
    ];
    let numeraire_pools = Some(vec![
        (TKA.clone(), NUM.clone(), "NUM_TKA_POOL".to_string(), 0.0, TOKEN_UNIT * 1_000_000, TOKEN_UNIT * 1_000_000),
        (TKB.clone(), NUM.clone(), "NUM_TKB_POOL".to_string(), 0.0, TOKEN_UNIT * 1_000_000, TOKEN_UNIT * 1_000_000),
    ]);
    let setup = setup_test_environment(main_pools, numeraire_pools, None);
    let amount_in_val: u128 = TOKEN_UNIT * 3; 

    let quote = setup.engine.quote_multi(&TKA, &TKB, amount_in_val, 3, None).await;

    assert!(quote.amount_out.is_some(), "Should find a quote");
    assert_eq!(quote.is_split_trade, true, "Should be a split trade");
    assert_eq!(quote.num_split_paths, 3, "Should split across 3 paths");
    assert_eq!(quote.path_details.len(), 3, "Path details should contain 3 paths");

    let input_per_split_dec = Decimal::from_u128(amount_in_val / 3).unwrap_or_default() / Decimal::from(TOKEN_UNIT);
    let fee_dec = Decimal::from_f64(fee_val).unwrap_or_default();
    let expected_out_one_split_dec = input_per_split_dec * (Decimal::ONE - fee_dec);
    let expected_total_out_dec = (expected_out_one_split_dec * Decimal::from(3u8)).round_dp(DEFAULT_DECIMALS);
    
    let actual_total_out_dec = Decimal::from_i128_with_scale(quote.amount_out.unwrap() as i128, DEFAULT_DECIMALS);
    let tolerance = Decimal::from_str("0.0000000000000001").unwrap();
    assert!((actual_total_out_dec - expected_total_out_dec).abs() < tolerance,
            "Total amount out {} is not close to expected {}. Diff: {}", 
            actual_total_out_dec, expected_total_out_dec, (actual_total_out_dec - expected_total_out_dec).abs());
    
    for pd in &quote.path_details {
        assert_eq!(pd.input_amount.unwrap(), amount_in_val / 3, "Input amount for split path is incorrect");
    }
    assert!(quote.route.is_empty(), "Route should be empty for split trades");

    let amount_in_dec = Decimal::from_u128(amount_in_val).unwrap_or_default() / Decimal::from(TOKEN_UNIT);
    let actual_total_gross_out_dec = Decimal::from_i128_with_scale(quote.gross_amount_out.unwrap_or(0) as i128, DEFAULT_DECIMALS);

    let expected_gross_out_one_split_dec = input_per_split_dec * (Decimal::ONE - fee_dec);
    let expected_total_gross_out_dec = (expected_gross_out_one_split_dec * Decimal::from(3u8)).round_dp(DEFAULT_DECIMALS);
    assert!((actual_total_gross_out_dec - expected_total_gross_out_dec).abs() < tolerance,
            "Total gross amount out {} is not close to expected {}", actual_total_gross_out_dec, expected_total_gross_out_dec);

    let expected_gas_estimate_sum = setup.engine.avg_gas_units_per_swap * 3; 
    assert_eq!(quote.gas_estimate.unwrap_or(0), expected_gas_estimate_sum, "Gas estimate sum is incorrect");

    let expected_overall_mid_price = actual_total_out_dec / amount_in_dec; 
    assert!((quote.mid_price.unwrap_or_default() - expected_overall_mid_price).abs() < tolerance,
            "Overall mid price {} is not close to expected {}", quote.mid_price.unwrap_or_default(), expected_overall_mid_price);
    
    assert!(quote.fee_bps.unwrap_or_default() < Decimal::from_str("0.01").unwrap(), "Overall fee_bps should be near zero for 0% protocol fee"); 
    
    let gross_effective_price = actual_total_gross_out_dec / amount_in_dec;
    let net_effective_price = actual_total_out_dec / amount_in_dec; 
    let reference_mid_price = Decimal::ONE; // Established by TKA-NUM and TKB-NUM pools

    let expected_pi_bps = ((gross_effective_price / reference_mid_price) - Decimal::ONE).abs() * Decimal::new(10000,0);
    let expected_slippage_bps = ((net_effective_price / reference_mid_price) - Decimal::ONE).abs() * Decimal::new(10000,0);

    assert!((quote.price_impact_bps.unwrap_or_default() - expected_pi_bps.round_dp(DEFAULT_DECIMALS)).abs() < tolerance,
        "Overall Price Impact BPS {} is not close to expected {}. Expected based on RefPrice=1: {}", 
        quote.price_impact_bps.unwrap_or_default(), expected_pi_bps.round_dp(DEFAULT_DECIMALS), expected_pi_bps);
    assert!((quote.slippage_bps.unwrap_or_default() - expected_slippage_bps.round_dp(DEFAULT_DECIMALS)).abs() < tolerance,
        "Overall Slippage BPS {} is not close to expected {}. Expected based on RefPrice=1: {}", 
        quote.slippage_bps.unwrap_or_default(), expected_slippage_bps.round_dp(DEFAULT_DECIMALS), expected_slippage_bps);
}

#[tokio::test]
async fn test_quote_multi_overlapping_paths_selection() {
    let large_reserve = TOKEN_UNIT * 1000;
    let fee = 0.003; 
    let pools = vec![
        (TKA.clone(), TKB.clone(), "P1".to_string(), fee, large_reserve, large_reserve), 
        (TKA.clone(), TKC.clone(), "P2".to_string(), fee, large_reserve, large_reserve), 
        (TKC.clone(), TKB.clone(), "P3".to_string(), fee, large_reserve, large_reserve), 
        (TKA.clone(), TKD.clone(), "P4".to_string(), fee, large_reserve, large_reserve),
        (TKD.clone(), TKC.clone(), "P5".to_string(), fee, large_reserve, large_reserve), 
    ];
    let setup = setup_test_environment(pools, None);
    let amount_in: u128 = TOKEN_UNIT * 3; 

    let quote = setup.engine.quote_multi(&TKA, &TKB, amount_in, 3, None).await;

    assert!(quote.amount_out.is_some(), "Should find a quote");
    assert_eq!(quote.is_split_trade, true, "Should be a split trade");
    // Expected: Path A (P1) and Path B (P2+P3) are chosen. Path C (P4+P5+P3) is not chosen due to P3 overlap.
    assert_eq!(quote.num_split_paths, 2, "Should split across 2 non-overlapping paths (P1 and P2-P3)");
    assert_eq!(quote.path_details.len(), 2, "Path details should contain 2 paths");
    assert!(quote.route.is_empty(), "Route should be empty for split trades");

    let mut found_p1_path = false;
    let mut found_p2_p3_path = false;
    for pd in &quote.path_details {
        if pd.pools.len() == 1 && pd.pools.contains(&"P1".to_string()) { found_p1_path = true; }
        if pd.pools.len() == 2 && pd.pools.contains(&"P2".to_string()) && pd.pools.contains(&"P3".to_string()) { found_p2_p3_path = true; }
    }
    assert!(found_p1_path, "Path via P1 was not selected for split");
    assert!(found_p2_p3_path, "Path via P2 and P3 was not selected for split");
}

#[tokio::test]
async fn test_quote_multi_split_varied_paths() {
    let large_reserve = TOKEN_UNIT * 1000;
    let pools = vec![
        (TKA.clone(), TKB.clone(), "P1_varied".to_string(), 0.001, large_reserve, large_reserve), // Fee 0.1%
        (TKA.clone(), TKB.clone(), "P2_varied".to_string(), 0.003, large_reserve, large_reserve), // Fee 0.3%
        (TKA.clone(), TKB.clone(), "P3_varied".to_string(), 0.005, large_reserve, large_reserve), // Fee 0.5%
    ];
    let setup = setup_test_environment(pools, None);
    let amount_in_val: u128 = TOKEN_UNIT * 3; // 3 TKA

    // k=3 for path_details to include all 3 paths
    let quote = setup.engine.quote_multi(&TKA, &TKB, amount_in_val, 3, None).await;

    assert!(quote.amount_out.is_some(), "Should find a quote");
    assert_eq!(quote.is_split_trade, true);
    assert_eq!(quote.num_split_paths, 3);
    assert_eq!(quote.path_details.len(), 3);
    assert!(quote.route.is_empty(), "Route should be empty for split trades");

    let input_per_path_dec = Decimal::from_u128(TOKEN_UNIT).unwrap_or_default() / Decimal::from(TOKEN_UNIT); // 1 TKA in Decimal
    let fee_p1 = Decimal::from_f64(0.001).unwrap();
    let fee_p2 = Decimal::from_f64(0.003).unwrap();
    let fee_p3 = Decimal::from_f64(0.005).unwrap();

    let out_p1_dec = input_per_path_dec * (Decimal::ONE - fee_p1); // 0.999
    let out_p2_dec = input_per_path_dec * (Decimal::ONE - fee_p2); // 0.997
    let out_p3_dec = input_per_path_dec * (Decimal::ONE - fee_p3); // 0.995
    
    let expected_total_out_dec = (out_p1_dec + out_p2_dec + out_p3_dec).round_dp(DEFAULT_DECIMALS);
    let actual_total_out_dec = Decimal::from_i128_with_scale(quote.amount_out.unwrap() as i128, DEFAULT_DECIMALS);
    let tolerance = Decimal::from_str("0.0000000000000001").unwrap();
     assert!((actual_total_out_dec - expected_total_out_dec).abs() < tolerance,
            "Total amount out {} is not close to expected {}. Diff: {}", 
            actual_total_out_dec, expected_total_out_dec, (actual_total_out_dec - expected_total_out_dec).abs());

    // Check path_details are sorted by individual amount_out descending
    assert_eq!(quote.path_details[0].pools[0], "P1_varied"); // Best path (lowest fee)
    assert_eq!(quote.path_details[1].pools[0], "P2_varied");
    assert_eq!(quote.path_details[2].pools[0], "P3_varied"); // Worst path (highest fee)
}


// Original test, modified to be async and use the test setup
#[tokio::test]
async fn original_test_price_computation_no_pools() {
    let setup = setup_test_environment(vec![], None); 
    let quote = setup.engine.quote(&TKA, &TKB, TOKEN_UNIT, None).await; 
    assert!(quote.amount_out.is_none()); 
    assert_eq!(quote.is_split_trade, false);
    assert_eq!(quote.num_split_paths, 0);
} 