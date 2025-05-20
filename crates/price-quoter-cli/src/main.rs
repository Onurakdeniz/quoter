use anyhow::Result;
use price_quoter::config::AppConfig;
use price_quoter::engine::{PriceEngine, graph::TokenGraph};
use price_quoter::data_management::component_tracker::ComponentTracker;
use price_quoter::Bytes;
use std::str::FromStr;
use futures_util::StreamExt;
use price_quoter::data_management::cache::QuoteCache;
use std::sync::{Arc, RwLock};
// Use prelude for Decimal and common traits like FromPrimitive
use rust_decimal::prelude::*;

// Helper to get token symbol and decimals
// Returns (Symbol, Decimals)
fn get_token_symbol_decimals(tracker: &ComponentTracker, token_address_bytes: &Bytes) -> (String, u8) {
    if let Some(data) = tracker.all_tokens.read().unwrap().get(token_address_bytes) {
        (data.symbol.clone(), data.decimals.try_into().unwrap_or(18_u8)) // Ensure u8, default 18
    } else {
        // Fallback if token not found in tracker
        (token_address_bytes.to_string(), 18_u8) // Show hex and assume 18 decimals
    }
}

// Helper to format raw u128 amount to a human-readable string
fn format_token_amount(raw_amount: u128, decimals: u8) -> String {
    if decimals == 0 {
        return format!("{}", raw_amount);
    }
    let divisor = Decimal::from(10u64.pow(decimals as u32));
    let amount_decimal = Decimal::from(raw_amount) / divisor;
    amount_decimal.round_dp(6).to_string() // Display with 6 decimal places
}

// Helper to format a route (Vec<Bytes>) into a String of symbols
fn format_route_symbols(tracker: &ComponentTracker, route_bytes: &[Bytes]) -> String {
    route_bytes.iter()
        .map(|addr_bytes| get_token_symbol_decimals(tracker, addr_bytes).0)
        .collect::<Vec<String>>()
        .join(" -> ")
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load AppConfig using load_with_cli to respect CLI args, file, and env vars
    let config = AppConfig::load_with_cli(); 

    // Get operational parameters from config, ensuring they are provided
    let sell_token_str = config.sell_token_address.clone().ok_or_else(|| anyhow::anyhow!("--sell-token is required"))?;
    let buy_token_str = config.buy_token_address.clone().ok_or_else(|| anyhow::anyhow!("--buy-token is required"))?;
    let sell_amount_f64 = config.sell_amount_value.ok_or_else(|| anyhow::anyhow!("--sell-amount is required"))?;
    // display_numeraire_token is optional for the quote display, not used by engine directly for core logic
    // let _display_numeraire_str = config.display_numeraire_token_address;

    let tracker = ComponentTracker::new();
    // Wrap TokenGraph in Arc<RwLock<>> for shared mutable access
    let graph_arc = Arc::new(RwLock::new(TokenGraph::new())); 

    // Ingest state from Tycho Indexer
    let updates = tracker
        .stream_updates(
            &config.tycho_url,
            config.chain,
            &config.tycho_api_key,
            config.tvl_threshold,
        )
        .await?;
    // Await at least one update to populate state
    tokio::pin!(updates);

    // Create a new cache instance for the engine
    let engine_cache = Arc::new(RwLock::new(QuoteCache::new()));

    // Initialize PriceEngine using from_config
    let engine = PriceEngine::from_config(
        tracker.clone(), 
        graph_arc.clone(), // Pass the Arc-wrapped graph
        engine_cache, 
        &config
    );

    let sell_token_bytes = Bytes::from_str(&sell_token_str).expect("Invalid sell token");
    let buy_token_bytes = Bytes::from_str(&buy_token_str).expect("Invalid buy token");
    
    // Get sell and buy token info once
    let (sell_token_symbol, sell_token_decimals) = get_token_symbol_decimals(&tracker, &sell_token_bytes);
    let (buy_token_symbol, buy_token_decimals) = get_token_symbol_decimals(&tracker, &buy_token_bytes);
    
    // Calculate raw sell amount based on its decimals
    let sell_amount_raw = Decimal::from_f64(sell_amount_f64).unwrap_or_default() * Decimal::from(10u64.pow(sell_token_decimals as u32));
    let sell_amount_u128 = sell_amount_raw.to_u128().unwrap_or(0);


    let mut update_count = 0;
    println!("Starting to listen for Tycho Indexer updates and quote continuously...");
    println!("Quoting for: {} {} -> {}", sell_amount_f64, sell_token_symbol, buy_token_symbol);


    loop {
        match updates.next().await {
            Some(_block_update) => {
                update_count += 1;
                println!("\n--- Update Cycle: {} ---", update_count);
                // You can inspect _block_update here if it contains useful info like block number
                // For example: println!("Received Block: {:?}\", _block_update.block_number);

                // Update the graph with the latest state from the tracker
                {
                    let mut graph_w = graph_arc.write().unwrap(); // Acquire write lock
                    graph_w.update_from_components_with_tracker(
                        &tracker.all_pools.read().unwrap(),
                        &tracker.pool_states.read().unwrap(),
                        &tracker.all_tokens.read().unwrap(),
                    );
                } // Write lock is released here
                
                // Check graph state after update (with a read lock)
                let (node_count, edge_count) = {
                    let graph_r = graph_arc.read().unwrap();
                    (graph_r.graph.node_count(), graph_r.graph.edge_count())
                };

                if node_count == 0 {
                    println!("Graph is empty after update. Skipping quote.");
                    continue;
                }

                println!("Quoting with updated graph ({} nodes, {} edges)...", node_count, edge_count);

                // Call the quote function
                let quote = engine.quote_multi(&sell_token_bytes, &buy_token_bytes, sell_amount_u128, 5, None);

                println!("\n--- Overall Best Quote Summary ---");
                println!("Selling: {} {}", sell_amount_f64, sell_token_symbol);

                if let Some(amount_out_raw) = quote.amount_out {
                    let formatted_amount_out = format_token_amount(amount_out_raw, buy_token_decimals);
                    println!("Receiving (Net): {} {}", formatted_amount_out, buy_token_symbol);

                    if sell_amount_f64 > 0.0 { // Initial check for non-zero float sell_amount
                        if let Some(sell_amount_decimal_val) = Decimal::from_f64(sell_amount_f64) {
                            if !sell_amount_decimal_val.is_zero() {
                                let buy_amount_decimal_val = Decimal::from(amount_out_raw) / Decimal::from(10u64.pow(buy_token_decimals as u32));
                                let effective_price = buy_amount_decimal_val / sell_amount_decimal_val;
                                println!("Effective Price: {} {} per {}", effective_price.round_dp(6), buy_token_symbol, sell_token_symbol);
                            } else {
                                println!("Effective Price: N/A (sell amount decimal is zero)");
                            }
                        } else {
                            println!("Effective Price: N/A (sell amount could not be converted to decimal)");
                        }
                    } else {
                        println!("Effective Price: N/A (sell amount is not positive)");
                    }
                } else {
                    println!("Receiving (Net): No quote available");
                }
                
                if let Some(gross_amount_out_raw) = quote.gross_amount_out {
                     let formatted_gross_amount_out = format_token_amount(gross_amount_out_raw, buy_token_decimals);
                     println!("Receiving (Gross): {} {}", formatted_gross_amount_out, buy_token_symbol);
                }

                println!("Route (Best Path): {}", format_route_symbols(&tracker, &quote.route));
                
                if let Some(mid_price_engine) = quote.mid_price {
                    println!("Mid Price (Engine): {} (Note: Internal engine rate, e.g., {} per {})", mid_price_engine.round_dp(8), sell_token_symbol, buy_token_symbol);
                }
                
                // Print other summary details if they exist
                quote.slippage_bps.map(|val| println!("Slippage (bps): {}", val));
                quote.fee_bps.map(|val| println!("Fee (bps): {}", val));
                quote.gas_estimate.map(|val| println!("Gas Estimate: {}", val));
                quote.spread_bps.map(|val| println!("Spread (bps): {}", val));
                quote.price_impact_bps.map(|val| println!("Price Impact (bps): {}", val));
                quote.cache_block.map(|val| println!("Cache Block: {}", val));


                if !quote.path_details.is_empty() {
                    println!("\n--- Top {} Path Details ---", quote.path_details.len());
                    for (i, detail) in quote.path_details.iter().enumerate() {
                        println!("  Path {}:", i + 1);
                        println!("    Route: {}", format_route_symbols(&tracker, &detail.route));
                        println!("    Pool IDs: {:?}", detail.pools);

                        let (path_input_symbol, path_input_decimals) = if let Some(first_token_in_route) = detail.route.first() {
                            get_token_symbol_decimals(&tracker, first_token_in_route)
                        } else {
                            (sell_token_symbol.clone(), sell_token_decimals) // Fallback
                        };
                        let (path_output_symbol, path_output_decimals) = if let Some(last_token_in_route) = detail.route.last() {
                            get_token_symbol_decimals(&tracker, last_token_in_route)
                        } else {
                            (buy_token_symbol.clone(), buy_token_decimals) // Fallback
                        };

                        if let Some(input_raw) = detail.input_amount {
                            let formatted_input = format_token_amount(input_raw, path_input_decimals);
                            println!("    Input Amount (Path Sim): {} {}", formatted_input, path_input_symbol);

                            if let Some(output_raw) = detail.amount_out {
                                let formatted_output = format_token_amount(output_raw, path_output_decimals);
                                println!("    Net Amount Out (Path Sim): {} {}", formatted_output, path_output_symbol);

                                // Calculate effective price for this path
                                let input_dec = Decimal::from(input_raw) / Decimal::from(10u64.pow(path_input_decimals as u32));
                                if !input_dec.is_zero() {
                                    let output_dec = Decimal::from(output_raw) / Decimal::from(10u64.pow(path_output_decimals as u32));
                                    let path_price = output_dec / input_dec;
                                    println!("    Effective Price (Path Sim): {} {} per {}", path_price.round_dp(6), path_output_symbol, path_input_symbol);
                                }
                            } else {
                                println!("    Net Amount Out (Path Sim): N/A");
                            }
                        } else {
                            println!("    Input Amount (Path Sim): N/A");
                        }
                         if let Some(gross_out_raw) = detail.gross_amount_out {
                            let formatted_gross_output = format_token_amount(gross_out_raw, path_output_decimals);
                            println!("    Gross Amount Out (Path Sim): {} {}", formatted_gross_output, path_output_symbol);
                        }

                        if let Some(mid_price_path_raw) = detail.mid_price {
                             // detail.mid_price is raw_buy_units / raw_sell_units for that specific simulation
                             // To make it BUY_TOKEN per SELL_TOKEN: mid_price_raw * (10^sell_decimals) / (10^buy_decimals)
                            let scaled_mid_price = mid_price_path_raw * Decimal::from(10u64.pow(path_input_decimals as u32)) / Decimal::from(10u64.pow(path_output_decimals as u32));
                            println!("    Mid Price (Path Sim): {} {} per {}", scaled_mid_price.round_dp(6), path_output_symbol, path_input_symbol);
                        }
                    }
                }

                if let Some(metrics) = &quote.depth_metrics {
                    println!("\n--- Depth Metrics ---");
                    for (slippage_target, depth_amount_raw) in metrics {
                        // Assuming depth_amount is in terms of the sell_token for now
                        let formatted_depth = format_token_amount(*depth_amount_raw, sell_token_decimals);
                        println!("  Slippage {}: Amount {} {}", slippage_target, formatted_depth, sell_token_symbol);
                    }
                }
            },
            None => {
                println!("Tycho Indexer update stream ended or an error occurred. Exiting.");
                break;
            }
        }
    }

    Ok(())
}

// Example Clap structure (if you add clap as a dependency)
/*
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    sell_token: String,
    #[arg(long)]
    buy_token: String,
    #[arg(long)]
    sell_amount: f64,
    #[arg(long)]
    numeraire_token: Option<String>,
    // Add other arguments from your run.md command
}
*/ 