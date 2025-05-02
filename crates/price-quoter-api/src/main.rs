use std::sync::{Arc, RwLock};
use price_quoter::{component_tracker::ComponentTracker, graph::TokenGraph, price_engine::PriceEngine, api::ApiServer, config::AppConfig};
use tokio::signal;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;

const GAS_REFRESH_SECS: u64 = 10;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let config = AppConfig::load();
    let addr = "0.0.0.0:8080";
    let tracker = ComponentTracker::new();
    let mut updates_stream = tracker.stream_updates(&config.tycho_url, config.chain, &config.tycho_api_key, config.tvl_threshold).await?;

    // Wait first update
    if updates_stream.next().await.is_none() {
        anyhow::bail!("Could not retrieve initial pool data.");
    }

    let mut graph = TokenGraph::new();
    {
        let pools_r = tracker.all_pools.read().unwrap();
        let states_r = tracker.pool_states.read().unwrap();
        let tokens_r = tracker.all_tokens.read().unwrap();
        graph.update_from_components_with_tracker(&pools_r, &states_r, &tokens_r);
    }

    let shared_cache = Arc::new(RwLock::new(price_quoter::cache::QuoteCache::new()));
    let gas_price_shared = Arc::new(RwLock::new(config.gas_price_gwei.map(|g| g as u128 * 1_000_000_000u128).unwrap_or(30_000_000_000u128)));

    // === NEW: Spawn background task to keep gas price updated ===
    if let Some(rpc_url) = config.rpc_url.clone() {
        let gas_price_handle = gas_price_shared.clone();
        tokio::spawn(async move {
            let client = Client::new();
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(GAS_REFRESH_SECS));
            loop {
                ticker.tick().await;
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

    // We need PriceEngine with 'static lifetime. Create graph with 'static via leak.
    let graph_static: &'static TokenGraph = Box::leak(Box::new(graph));

    let engine = PriceEngine::from_config(tracker.clone(), graph_static, shared_cache.clone(), gas_price_shared.clone(), &config);

    let server = ApiServer::new(Arc::new(engine));

    tokio::spawn(async move {
        server.start(addr).await;
    });

    // Keep runtime alive until ctrl+c
    signal::ctrl_c().await?;
    Ok(())
}
