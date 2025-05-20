//! Price analytics, slippage, spread, and depth calculation.

use rust_decimal::Decimal;
use tycho_simulation::tycho_common::Bytes;
use petgraph::prelude::{NodeIndex, EdgeIndex};
use crate::engine::graph::TokenGraph; // Assuming TokenGraph is in engine module
use crate::data_management::component_tracker::ComponentTracker; // Assuming ComponentTracker is in data_management
use crate::engine::simulation; // For simulate_path_gross
use num_traits::FromPrimitive;


// Placeholder for find_depth_for_slippage, calculate_price_impact_bps, etc.
// from the original price_engine.rs file.

pub fn calculate_price_impact_bps(
    start_amount_dec: Decimal,
    gross_actual_amount_dec: Decimal,
    mid_price: Decimal,
) -> Option<Decimal> {
    if mid_price.is_zero() || start_amount_dec.is_zero() {
        return None;
    }
    // Expected amount out at mid_price
    let expected_amount_out_dec = start_amount_dec * mid_price;
    if expected_amount_out_dec.is_zero() {
        return None;
    }
    // Price impact: (expected_out - actual_gross_out) / expected_out * 10000
    let impact = (expected_amount_out_dec - gross_actual_amount_dec) / expected_amount_out_dec * Decimal::new(10000, 0);
    Some(impact)
}

pub fn calculate_slippage_bps(
    start_amount_dec: Decimal,       // Amount of token_in
    net_actual_amount_dec: Decimal, // Amount of token_out received, after all fees and gas
    final_mid_price: Decimal,       // Effective mid-price for this specific path execution (output/input at zero slippage for THIS path)
) -> Option<Decimal> {
    if final_mid_price.is_zero() || start_amount_dec.is_zero() {
        return None;
    }
    // Expected amount out if trading at the path's own ideal mid-price without any slippage
    let expected_out_at_path_mid_price = start_amount_dec * final_mid_price;
    if expected_out_at_path_mid_price.is_zero() {
        return None;
    }
    // Slippage: (expected_out_at_path_mid_price - net_actual_amount_out) / expected_out_at_path_mid_price * 10000
    let slippage = (expected_out_at_path_mid_price - net_actual_amount_dec) / expected_out_at_path_mid_price * Decimal::new(10000, 0);
    Some(slippage)
}

pub fn calculate_spread_bps(
    start_amount_dec: Decimal,       // Amount of token_in
    net_actual_amount_dec: Decimal, // Amount of token_out received, after all fees and gas
    final_mid_price: Decimal,       // Effective mid-price for this specific path execution (output/input at zero slippage for THIS path)
) -> Option<Decimal> {
    // Spread is often considered part of slippage in this context or calculated differently.
    // This implementation mirrors the original one which might be specific.
    // One common definition of spread in AMMs is related to the bid-ask difference derived from small trades.
    // Here, it seems to be calculated similarly to slippage but perhaps intended to capture a different aspect.
    calculate_slippage_bps(start_amount_dec, net_actual_amount_dec, final_mid_price)
}

// Simplified find_depth_for_slippage. Full version later.
pub fn find_depth_for_slippage(
    tracker: &ComponentTracker,
    graph: &TokenGraph,
    token_in: &Bytes,
    token_out: &Bytes,
    mid_price: Decimal, // The reference mid-price for the best path
    path_nodes: &[NodeIndex],
    path_edges: &[EdgeIndex],
    target_slippage_percent: f64, // e.g., 0.5, 1.0, 2.0
    block: Option<u64>,
) -> Option<u128> {
    if mid_price.is_zero() { return None; }

    let mut low = 0u128;
    // Estimate high based on a fraction of typical reserves or a large amount
    // This needs a better heuristic or access to reserve info.
    // Cap high to what Decimal::from(u128) can safely handle due to internal i64 conversion.
    let mut high = i64::MAX as u128;
    let mut result_amount = None;

    let target_slippage_bps = Decimal::from_f64(target_slippage_percent * 100.0).unwrap_or_else(|| Decimal::from_f64(0.0).unwrap());

    for _ in 0..20 { // Binary search for a limited number of iterations
        if high < low || high - low < 1000 { // Convergence or too small range
            break;
        }
        let test_amount_in = low + (high - low) / 2;
        if test_amount_in == 0 { low = 1; continue;}

        let gross_amount_out_opt = simulation::simulate_path_gross(
            tracker,
            graph,
            test_amount_in,
            path_nodes,
            path_edges,
            block,
        );

        if let Some(gross_amount_out) = gross_amount_out_opt {
            let gross_amount_out_dec = Decimal::from(gross_amount_out);
            let test_amount_in_dec = Decimal::from(test_amount_in);
            
            if test_amount_in_dec.is_zero() { low = test_amount_in +1; continue; }

            let effective_price = gross_amount_out_dec / test_amount_in_dec;
            let slippage_bps = (mid_price - effective_price) / mid_price * Decimal::new(10000,0);

            if slippage_bps <= target_slippage_bps {
                result_amount = Some(test_amount_in);
                low = test_amount_in + 1;
            } else {
                high = test_amount_in - 1;
            }
        } else {
            high = test_amount_in -1; // Simulation failed, try smaller amount
        }
         if high == 0 { break; }
    }
    result_amount
} 