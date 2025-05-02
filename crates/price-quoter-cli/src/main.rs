mod cli;
use crate::cli::{Cli, CliHandler};
use price_quoter::{component_tracker::ComponentTracker, graph::TokenGraph, price_engine::PriceEngine, config::AppConfig, utils::token_list, history};
use clap::Parser;
use tokio::time::{interval, Duration};
use futures::StreamExt;
use std::sync::{Arc, RwLock};
use tycho_simulation::tycho_common::Bytes;
use std::str::FromStr;
use num_traits::cast::ToPrimitive;
use std::time::Instant;
use tracing::info;
use reqwest::Client;
use serde_json::json;
use hex;

const REFRESH_SECS: u64 = 6;
const GAS_REFRESH_SECS: u64 = 10;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("price_quoter=info,price_quoter_cli=info")
        .init();
    let cli = Cli::parse();
    let mut config = AppConfig::load();

    // CLI overrides
    if let Some(ref n_token_str) = cli.numeraire_token {
        if let Ok(bytes) = Bytes::from_str(n_token_str) {
            config.numeraire_token = Some(bytes);
        }
    }
    if let Some(depth) = cli.probe_depth {
        config.probe_depth = Some(depth);
    }
    if let Some(max_hops_cli) = cli.max_hops {
        config.max_hops = Some(max_hops_cli);
    }
    if let Some(ref path) = cli.tokens_file {
        config.tokens_file = Some(path.clone());
    }

    if let Some(ref url) = config.rpc_url {
        std::env::set_var("RPC_URL", url);
    }

    let mut tracker = ComponentTracker::new();
    let updates_stream = tracker.stream_updates(&config.tycho_url, config.chain, &config.tycho_api_key, config.tvl_threshold).await?;

    // Wrap in Pin<Box<...>> so that it implements `Unpin` for StreamExt::next()
    let mut updates = Box::pin(updates_stream);

    // Wait first update
    if updates.next().await.is_none() {
        anyhow::bail!("Could not retrieve initial pool data.");
    }

    let mut graph = TokenGraph::new();
    {
        let pools_r = tracker.all_pools.read().unwrap();
        let states_r = tracker.pool_states.read().unwrap();
        let tokens_r = tracker.all_tokens.read().unwrap();
        graph.update_from_components_with_tracker(&pools_r, &states_r, &tokens_r);
    }

    // Load token list once (empty if single-quote mode)
    let tokens_to_track: Vec<Bytes> = if let Some(ref path) = config.tokens_file {
        match token_list::load_token_list(path) {
            Ok(list) => list,
            Err(e) => {
                eprintln!("[TOKEN_LIST_ERROR] {}", e);
                Vec::new()
            }
        }
    } else { Vec::new() };

    let shared_cache = Arc::new(RwLock::new(price_quoter::cache::QuoteCache::new()));
    let gas_price_shared = Arc::new(RwLock::new(config.gas_price_gwei.map(|g| g as u128 * 1_000_000_000u128).unwrap_or(30_000_000_000u128)));

    // === NEW: Spawn background task to keep gas price updated ===
    if let Some(rpc_url) = config.rpc_url.clone() {
        let gas_price_handle = gas_price_shared.clone();
        tokio::spawn(async move {
            let client = Client::new();
            let mut ticker = interval(Duration::from_secs(GAS_REFRESH_SECS));
            loop {
                ticker.tick().await; // wait first tick
                // Build JSON-RPC payload
                let payload = json!({
                    "jsonrpc": "2.0",
                    "method": "eth_gasPrice",
                    "params": [],
                    "id": 1
                });
                match client.post(&rpc_url).json(&payload).send().await {
                    Ok(resp) => {
                        if let Ok(json_resp) = resp.json::<serde_json::Value>().await {
                            if let Some(result_hex) = json_resp.get("result").and_then(|v| v.as_str()) {
                                if let Ok(wei) = u128::from_str_radix(result_hex.trim_start_matches("0x"), 16) {
                                    *gas_price_handle.write().unwrap() = wei;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[GAS_FETCH_ERROR] {}", e);
                    }
                }
            }
        });
    } else {
        eprintln!("[INFO] rpc_url not set, using static gas price");
    }

    // Initial engine usage for setup and initial quotes
    {
        let engine = PriceEngine::from_config(tracker.clone(), &graph, shared_cache.clone(), gas_price_shared.clone(), &config);
        // Precompute all quotes with no block filter
        engine.precompute_all_quotes(None);
        if !tokens_to_track.is_empty() {
            let eth_addr = config.numeraire_token.clone().expect("NUMERAIRE_TOKEN must be set for batch mode");
            // Use a temporary engine ignoring probe_depth for listing
            let list_engine = PriceEngine::with_cache(tracker.clone(), &graph, shared_cache.clone(), gas_price_shared.clone(), engine.max_hops);
            let prices = list_engine.quote_tokens_vs_eth(&tokens_to_track, &eth_addr, 0, None);
            for (tok, q) in prices {
                history::append_token_price(&chrono::Utc::now().to_rfc3339(), 0, &tok, q.amount_out.unwrap_or(0), q.mid_price, "eth_prices.csv").ok();
                let token_hex = format!("0x{}", hex::encode(&tok));
                let cache_info = q.cache_block.map_or(String::new(), |b| format!(" (cached @ block {})", b));
                if let Some(mid) = q.mid_price {
                    println!("{} => {:.18} ETH{}", token_hex, mid.to_f64().unwrap_or(0.0), cache_info);
                } else {
                    println!("{} => N/A{}", token_hex, cache_info);
                }
            }
        } else {
            CliHandler::handle_quote(&cli, &engine, None);
        }
    }

    // Periodic refresh (REMOVED - now only updates on new blocks)
    // let mut ticker = interval(Duration::from_secs(REFRESH_SECS));
    let mut last_block_number: Option<u64> = None;
    loop {
        tokio::select! {
            maybe_update = updates.next() => {
                if let Some(block_update) = maybe_update { 
                    // Extract block number (assuming BlockUpdate has a 'block_number' field)
                    let current_block_number = block_update.block_number; 
                    info!(block = current_block_number, "Received block update.");
                    let block_opt = Some(current_block_number);

                    // Invalidate cache for the previous block to force fresh quotes for each block
                    if let Some(prev_block) = last_block_number {
                        shared_cache.write().unwrap().invalidate_block(prev_block);
                    }
                    last_block_number = Some(current_block_number);

                    // Invalidate cache for pools/tokens impacted by this block update
                    {
                        let mut cache_w = shared_cache.write().unwrap();
                        // Collect all changed pool IDs
                        let changed_pools = block_update.new_pairs.keys()
                            .chain(block_update.removed_pairs.keys())
                            .chain(block_update.states.keys());
                        for pool_id in changed_pools {
                            cache_w.invalidate_pool(pool_id);
                            // Also invalidate per-token cache for all tokens in this pool
                            if let Some(component) = tracker.all_pools.read().unwrap().get(pool_id) {
                                for token in &component.tokens {
                                    cache_w.invalidate_token(&token.address);
                                }
                            }
                        }
                    }
                    // Update graph on pool additions, removals, or liquidity/state changes
                    if !block_update.new_pairs.is_empty() || !block_update.removed_pairs.is_empty() || !block_update.states.is_empty() {
                        let pools_r = tracker.all_pools.read().unwrap();
                        let states_r = tracker.pool_states.read().unwrap();
                        let tokens_r = tracker.all_tokens.read().unwrap();
                        graph.update_from_components_with_tracker(&pools_r, &states_r, &tokens_r);
                        // Release locks
                        drop(pools_r);
                        drop(states_r);
                        drop(tokens_r);
                    }
                    // Start timing quote generation separately
                    let t_start = Instant::now();
                    // Create a new engine instance for potentially updated graph/state
                    let engine = PriceEngine::from_config(tracker.clone(), &graph, shared_cache.clone(), gas_price_shared.clone(), &config);
                    
                    // In batch mode (tokens_to_track non-empty), skip full precompute to reduce latency
                    if !tokens_to_track.is_empty() {
                        let eth_addr = config.numeraire_token.clone().expect("NUMERAIRE_TOKEN must be set for batch mode");
                        // Pass actual block number down, using temporary engine to ignore probe_depth
                        let list_engine = PriceEngine::with_cache(tracker.clone(), &graph, shared_cache.clone(), gas_price_shared.clone(), engine.max_hops);
                        let prices = list_engine.quote_tokens_vs_eth(&tokens_to_track, &eth_addr, 0, block_opt);
                        let now_str = chrono::Utc::now().to_rfc3339();
                        // Use actual block number for history
                        let history_block = current_block_number; 
                        for (tok, q) in prices {
                            // Persist raw net amount and mid_price to CSV
                            history::append_token_price(
                                &now_str,
                                history_block,
                                &tok,
                                q.amount_out.unwrap_or(0),
                                q.mid_price,
                                "eth_prices.csv",
                            ).ok();
                            let token_hex = format!("0x{}", hex::encode(&tok));
                            let cache_info = q.cache_block.map_or(String::new(), |b| format!(" (cached @ block {})", b));
                            // Determine decimals for token and ETH
                            let tokens_r = list_engine.tracker.all_tokens.read().unwrap();
                            let token_dec = tokens_r.get(&tok).map(|t| t.decimals).unwrap_or(18) as i32;
                            let eth_dec = tokens_r.get(&eth_addr).map(|t| t.decimals).unwrap_or(18) as i32;
                            // Print net output amount and average price
                            if let Some(amount_out) = q.amount_out {
                                if amount_out > 0 {
                                    let out_f = amount_out as f64 / 10f64.powi(eth_dec);
                                    let avg_price = q.mid_price.unwrap_or_default().to_f64().unwrap_or(0.0);
                                    print!("{} => {:.18} ETH (net) | avg_price: {} ETH", token_hex, out_f, avg_price);
                                    // Print depth metrics if available
                                    if let Some(depths) = &q.depth_metrics {
                                        for (perc, depth_amt) in depths {
                                            // depth_amt is input amount causing slippage
                                            let depth_f = *depth_amt as f64 / 10f64.powi(token_dec);
                                            print!(" | depth {} slippage: {:.6}", perc, depth_f);
                                        }
                                    }
                                    println!("{}", cache_info);
                                } else {
                                    // Fallback: show spot price if available
                                    let spot_price = list_engine.list_unit_price_vs_eth(&[tok.clone()], &eth_addr).get(0).map(|(_, p)| *p).unwrap_or_default();
                                    let tokens_r = list_engine.tracker.all_tokens.read().unwrap();
                                    let symbol = tokens_r.get(&tok).map(|t| t.symbol.clone()).unwrap_or_else(|| token_hex.clone());
                                    if !spot_price.is_zero() {
                                        let spot_f = spot_price.to_f64().unwrap_or(0.0);
                                        if spot_f > 1e6 || (spot_f > 0.0 && spot_f < 1e-6) {
                                            println!("{} ({}): N/A | spot price: {:.18} ETH [WARNING: Unrealistic spot price, likely no liquidity/path]{}", token_hex, symbol, spot_f, cache_info);
                                        } else {
                                            println!("{} ({}): N/A | spot price (no liquidity/fee check): {:.18} ETH{}", token_hex, symbol, spot_f, cache_info);
                                        }
                                    } else {
                                        println!("{} ({}): N/A{}", token_hex, symbol, cache_info);
                                    }
                                }
                            } else {
                                // Fallback: show spot price if available
                                let spot_price = list_engine.list_unit_price_vs_eth(&[tok.clone()], &eth_addr).get(0).map(|(_, p)| *p).unwrap_or_default();
                                let tokens_r = list_engine.tracker.all_tokens.read().unwrap();
                                let symbol = tokens_r.get(&tok).map(|t| t.symbol.clone()).unwrap_or_else(|| token_hex.clone());
                                if !spot_price.is_zero() {
                                    let spot_f = spot_price.to_f64().unwrap_or(0.0);
                                    if spot_f > 1e6 || (spot_f > 0.0 && spot_f < 1e-6) {
                                        println!("{} ({}): N/A | spot price: {:.18} ETH [WARNING: Unrealistic spot price, likely no liquidity/path]{}", token_hex, symbol, spot_f, cache_info);
                                    } else {
                                        println!("{} ({}): N/A | spot price (no liquidity/fee check): {:.18} ETH{}", token_hex, symbol, spot_f, cache_info);
                                    }
                                } else {
                                    println!("{} ({}): N/A{}", token_hex, symbol, cache_info);
                                }
                            }
                        }
                    } else {
                        // In single-quote mode, run full precompute before quoting
                        engine.precompute_all_quotes(block_opt);
                        // Pass actual block number down
                        CliHandler::handle_quote(&cli, &engine, block_opt);
                    }
                    let elapsed = t_start.elapsed();
                    // Use actual block number in log
                    info!(block = current_block_number, latency_ms = ?elapsed.as_millis(), "block-to-quotes latency");
                } else {
                    info!("Tycho update stream ended.");
                    break; // Exit loop if the stream ends
                }
            }
            // Removed ticker branch to only update on new blocks
            /*
            _ = ticker.tick() => {
                let engine = PriceEngine::from_config(tracker.clone(), &graph, shared_cache.clone(), gas_price_shared.clone(), &config);
                engine.precompute_all_quotes();
                if !tokens_to_track.is_empty() {
                    let eth_addr = config.numeraire_token.clone().expect("NUMERAIRE_TOKEN must be set for batch mode");
                    let prices = engine.quote_tokens_vs_eth(&tokens_to_track, &eth_addr, 0);
                    for (tok, q) in prices {
                        if let Some(out_amt) = q.amount_out {
                            history::append_token_price(
                                &chrono::Utc::now().to_rfc3339(),
                                0,
                                &tok,
                                out_amt,
                                q.mid_price,
                                "eth_prices.csv",
                            ).ok();
                        }
                    }
                } else {
                    CliHandler::handle_quote(&cli, &engine);
                }
            }
            */
        }
    }

    Ok(())
}
