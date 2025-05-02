//! CLI subcommand logic, output formatting, user interaction.

use clap::Parser;
use tycho_simulation::tycho_common::Bytes;
use price_quoter::price_engine::PriceEngine;
use std::str::FromStr;
use tracing::warn;
use hex;
use rustyline::Editor;
use rustyline::error::ReadlineError;
use rustyline::history::MemHistory;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;

/// CLI arguments for quoting a price.
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[arg(long)]
    pub sell_token: Option<String>,
    #[arg(long)]
    pub buy_token: Option<String>,
    #[arg(long)]
    pub sell_amount: Option<f64>,
    #[arg(long)]
    pub numeraire_token: Option<String>,
    #[arg(long)]
    pub probe_depth: Option<u128>,
    #[arg(long)]
    pub max_hops: Option<usize>,
    #[arg(long)]
    pub tokens_file: Option<String>,
}

/// Handles CLI commands and output.
pub struct CliHandler;

impl CliHandler {
    /// Handle a quote command: parse args, call PriceEngine, print result.
    pub fn handle_quote(cli: &Cli, engine: &PriceEngine, block: Option<u64>) {
        println!("[DEBUG] Enter handle_quote");
        if cli.buy_token.is_none() {
            eprintln!("Error: buy_token must be provided in single-quote mode");
            return;
        }
        // Use CLI or fallback to engine config
        let sell_token_str = if cli.sell_token.is_none() {
            engine.numeraire_token.as_ref().map(|b| format!("0x{}", hex::encode(b))).unwrap_or_default()
        } else {
            cli.sell_token.as_ref().unwrap().clone()
        };
        let sell_token = match Bytes::from_str(&sell_token_str) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Error: Invalid sell_token address '{}': {}", sell_token_str, e);
                return;
            }
        };
        let buy_token_str = cli.buy_token.as_ref().unwrap();
        let buy_token = match Bytes::from_str(buy_token_str) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Error: Invalid buy_token address '{}': {}", buy_token_str, e);
                return;
            }
        };
        // Fetch token info from tracker to get decimals
        let tokens_r = engine.tracker.all_tokens.read().unwrap();
        let sell_token_info = tokens_r.get(&sell_token);
        let buy_token_info = tokens_r.get(&buy_token);
        let sell_decimals = match sell_token_info {
            Some(info) => info.decimals,
            None => {
                warn!("Sell token {} not found in tracker, assuming 18 decimals.", sell_token_str);
                18 // Default or fallback
            }
        };
        let buy_decimals = match buy_token_info {
            Some(info) => info.decimals,
            None => {
                warn!("Buy token {} not found in tracker, assuming 18 decimals.", buy_token_str);
                18 // Default or fallback
            }
        };
        // Debug: Print decimals for both tokens
        let sell_symbol = sell_token_info.map_or(&sell_token_str, |t| &t.symbol);
        let buy_symbol = buy_token_info.map_or(buy_token_str, |t| &t.symbol);
        println!("[DEBUG] Sell token decimals: {} ({}), Buy token decimals: {} ({})", sell_decimals, sell_symbol, buy_decimals, buy_symbol);
        // Use CLI provided sell_amount or default to 1.0 token
        let sell_amount = cli.sell_amount.unwrap_or(1.0);
        if cli.sell_token.is_none() && engine.numeraire_token.is_some() {
            println!("[INFO] Using numeraire_token from config: {}", sell_token_str);
        }

        // Explore up to 5 paths, 3 hops max
        const MAX_K_PATHS: usize = 5;
        let quote_result = engine.quote_multi(&sell_token, &buy_token, (sell_amount * 10f64.powi(sell_decimals as i32)) as u128, MAX_K_PATHS, block);

        let sell_symbol = sell_token_info.map_or(&sell_token_str, |t| &t.symbol);
        let buy_symbol = buy_token_info.map_or(buy_token_str, |t| &t.symbol);

        // Debug: Print details of all paths found
        // NOTE: path_details is expected to be sorted by net output descending (best path first)
        debug_assert!(quote_result.path_details.windows(2).all(|w| {
            let a = w[0].amount_out.unwrap_or(0);
            let b = w[1].amount_out.unwrap_or(0);
            a >= b
        }), "path_details is not sorted by net output descending");
        println!("[DEBUG] Paths found by quote_multi (k={}, depth={}):", MAX_K_PATHS, engine.max_hops);
        for (i, path_detail) in quote_result.path_details.iter().enumerate() {
            let path_route_str = path_detail.route.iter()
                .map(|addr| tokens_r.get(addr).map_or_else(|| format!("0x{}..?", hex::encode(&addr[..4])), |t| t.symbol.clone()))
                .collect::<Vec<_>>().join(" -> ");
            let path_amount_out_float = path_detail.amount_out.map_or("N/A".to_string(), |a| (a as f64 / 10f64.powi(buy_decimals as i32)).to_string());
            println!("  Path {}: Route=[{}], Net Output Amount={}, MidPrice={:?}, Slippage={:?}, Spread={:?}",
                     i + 1, path_route_str, path_amount_out_float, path_detail.mid_price, path_detail.slippage_bps, path_detail.spread_bps);

            // Print the actual pool/protocol/fee for each hop in this path
            if path_detail.route.len() > 1 && !path_detail.pools.is_empty() {
                println!("    Hops:");
                for ((window, pool_id)) in path_detail.route.windows(2).zip(path_detail.pools.iter()) {
                    let from_addr = &window[0];
                    let to_addr = &window[1];
                    let from_idx = engine.graph.token_indices.get(from_addr);
                    let to_idx = engine.graph.token_indices.get(to_addr);
                    let pool_display = if let (Some(from_idx), Some(to_idx)) = (from_idx, to_idx) {
                        engine.graph.graph.edges_connecting(*from_idx, *to_idx)
                            .find_map(|edge_ref| {
                                let edge_data = edge_ref.weight();
                                if &edge_data.pool_id == pool_id {
                                    Some(format!("{} (protocol: {}, fee: {})", edge_data.pool_id, edge_data.protocol, edge_data.fee.map_or("?".to_string(), |f| format!("{:.4}", f))))
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| format!("{} (unknown pool)", pool_id))
                    } else {
                        format!("{} (unknown pool)", pool_id)
                    };
                    println!("      {} -> {} -> {}", 
                        tokens_r.get(from_addr).map_or_else(|| format!("0x{}..?", hex::encode(&from_addr[..4])), |t| t.symbol.clone()),
                        pool_display,
                        tokens_r.get(to_addr).map_or_else(|| format!("0x{}..?", hex::encode(&to_addr[..4])), |t| t.symbol.clone())
                    );
                }
            }
        }

        println!("Quote for {} {} -> {}:", sell_amount, sell_symbol, buy_symbol);

        if let Some(amount_out) = quote_result.amount_out {
            let amount_out_float = amount_out as f64 / 10f64.powi(buy_decimals as i32);
            let gross_amount_out_float = quote_result.gross_amount_out.map(|a| a as f64 / 10f64.powi(buy_decimals as i32));
            let gas_cost_token = quote_result.gas_estimate.map(|gas| {
                let gas_price_wei = *engine.gas_price_wei.read().unwrap();
                let gas_cost_wei = gas_price_wei * gas as u128;
                gas_cost_wei as f64 / 10f64.powi(buy_decimals as i32)
            });
            if let Some(gross) = gross_amount_out_float {
                println!("- Gross Output Amount: {} ({})", gross, buy_symbol);
            }
            if let Some(gas_cost) = gas_cost_token {
                println!("- Gas Cost: {} {}", gas_cost, buy_symbol);
            }
            println!("- Net Output Amount: {} ({})", amount_out_float, buy_symbol);
        } else {
            println!("- No route found or simulation failed.");
        }

        // Use the first path in path_details for summary display if available, including its pools
        let (summary_route, summary_pools, summary_mid_price, summary_slippage, summary_spread, summary_price_impact) = if let Some(best_path) = quote_result.path_details.first() {
            // Also extract price impact from the best path
            (best_path.route.clone(), best_path.pools.clone(), best_path.mid_price, best_path.slippage_bps, best_path.spread_bps, best_path.price_impact_bps)
        } else {
            // Fallback if no path details
            (quote_result.route.clone(), Vec::new(), quote_result.mid_price, quote_result.slippage_bps, quote_result.spread_bps, quote_result.price_impact_bps)
        };

        // Format the route for display, using the exact pools from the best path
        let mut route_parts = Vec::new();
        if !summary_route.is_empty() {
            // Start with the first token symbol
            let first_addr = &summary_route[0];
            route_parts.push(
                tokens_r.get(first_addr)
                    .map_or_else(
                        || format!("0x{}..?", hex::encode(&first_addr[..4])),
                        |token| token.symbol.clone()
                    )
            );
            // Zip token hops with the chosen pool ids for display
            for ((window, pool_id)) in summary_route.windows(2).zip(summary_pools.iter()) {
                let from_addr = &window[0];
                let to_addr = &window[1];
                // Display the pool id directly
                route_parts.push(format!("-> {} ->", pool_id));
                // Then the destination token symbol
                route_parts.push(
                    tokens_r.get(to_addr)
                        .map_or_else(
                            || format!("0x{}..?", hex::encode(&to_addr[..4])),
                            |token| token.symbol.clone()
                        )
                );
            }
        }
        println!("- Route: {}", route_parts.join(" "));

        // Summary metrics (Mid Price, Slippage, Spread)
        if let Some(mid) = summary_mid_price {
            println!("- Mid Price: {}", mid);
        } else {
            println!("- Mid Price: N/A");
        }
        if let Some(slip) = summary_slippage {
            println!("- Slippage: {:.2} bps", slip.round_dp(2));
        } else {
            println!("- Slippage: N/A");
        }
        if let Some(spread) = summary_spread {
            println!("- Spread: {:.2} bps", spread.round_dp(2));
        } else {
            println!("- Spread: N/A");
        }
        // Print Price Impact
        if let Some(impact) = summary_price_impact {
            println!("- Price Impact: {:.2} bps", impact.round_dp(2));
        } else {
            println!("- Price Impact: N/A");
        }

        // Print Depth Metrics (N-2)
        if let Some(ref metrics) = quote_result.depth_metrics {
            println!("- Depth for Slippage:");
            // Sort keys for consistent output (e.g., 0.5%, 1.0%, 2.0%)
            let mut sorted_keys: Vec<_> = metrics.keys().collect();
            sorted_keys.sort_by(|a, b| {
                let a_val = a.trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                let b_val = b.trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
            });

            for key in sorted_keys {
                if let Some(amount_u128) = metrics.get(key) {
                    // Convert amount back to readable float using sell token decimals
                    let depth_amount_float = *amount_u128 as f64 / 10f64.powi(sell_decimals as i32);
                    println!("    {}: {} {}", key, depth_amount_float, sell_symbol);
                }
            }
        } else {
             println!("- Depth for Slippage: N/A (Calculation failed or mid-price unavailable)");
        }

        println!("[DEBUG] Exit handle_quote");
        // Print cache metrics (hits/misses) for quotes and path enumeration (P-4)
        {
            let metrics = engine.cache.read().unwrap().metrics();
            let q_total = metrics.quote_hits + metrics.quote_misses;
            let p_total = metrics.path_hits + metrics.path_misses;
            println!(
                "[CACHE] quote hits={}/{} ({}%), path hits={}/{} ({}%)",
                metrics.quote_hits,
                q_total,
                if q_total > 0 { (metrics.quote_hits * 100) / q_total } else { 0 },
                metrics.path_hits,
                p_total,
                if p_total > 0 { (metrics.path_hits * 100) / p_total } else { 0 }
            );
        }

        // Persist price history to CSV after each quote
        let timestamp = Utc::now().to_rfc3339();
        let price_history_path = "price_history.csv";
        let file_exists = std::path::Path::new(price_history_path).exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(price_history_path)
            .expect("Unable to open or create price_history.csv");
        if !file_exists {
            writeln!(file, "timestamp,sell_token,buy_token,sell_amount,amount_out,mid_price,slippage_bps,fee_bps,gas_estimate,gross_amount_out,spread_bps").ok();
        }
        let amount_out = quote_result.amount_out.map(|a| a.to_string()).unwrap_or_default();
        let mid_price = quote_result.mid_price.map(|d| d.to_string()).unwrap_or_default();
        let slippage_bps = quote_result.slippage_bps.map(|d| d.to_string()).unwrap_or_default();
        let fee_bps = quote_result.fee_bps.map(|d| d.to_string()).unwrap_or_default();
        let gas_estimate = quote_result.gas_estimate.map(|g| g.to_string()).unwrap_or_default();
        let gross_amount_out = quote_result.gross_amount_out.map(|a| a.to_string()).unwrap_or_default();
        let spread_bps = quote_result.spread_bps.map(|d| d.to_string()).unwrap_or_default();
        writeln!(file, "{},{},{},{},{},{},{},{},{},{},{}",
            timestamp,
            sell_token_str,
            buy_token_str,
            sell_amount,
            amount_out,
            mid_price,
            slippage_bps,
            fee_bps,
            gas_estimate,
            gross_amount_out,
            spread_bps
        ).ok();
    }

    /// Interactive REPL mode for quoting
    pub fn handle_repl(_engine: &PriceEngine) {
        let mut rl = Editor::<(), MemHistory>::with_history(Default::default(), MemHistory::new()).unwrap();
        println!("Entering price-quoter REPL. Type 'help' for usage, 'exit' to quit.");
        loop {
            let readline = rl.readline("> ");
            match readline {
                Ok(line) => {
                    let _ = rl.add_history_entry(line.as_str());
                    // Process the line here
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D");
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }
    }
} 