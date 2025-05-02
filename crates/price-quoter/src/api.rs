use std::sync::Arc;
use std::str::FromStr;

use axum::{Router, routing::get, extract::Query, Json, serve};
use serde::{Serialize, Deserialize};
use crate::price_engine::PriceEngine;
use tycho_simulation::tycho_common::Bytes;

/// Simple JSON schema returned from /health and error cases
#[derive(Serialize)]
struct HealthResp {
    status: &'static str,
}

#[derive(Deserialize)]
struct QuoteParams {
    sell_token: String,
    buy_token: String,
    amount: Option<u128>,
}

#[derive(Serialize)]
struct QuoteResp {
    amount_out: Option<String>,
    gas_estimate: Option<u64>,
}

pub struct ApiServer {
    engine: Arc<PriceEngine<'static>>,
}

impl ApiServer {
    pub fn new(engine: Arc<PriceEngine<'static>>) -> Self {
        Self { engine }
    }

    pub async fn start(self, addr: &str) {
        let engine = self.engine.clone();

        // /health endpoint
        let health_route = Router::new().route("/health", get(|| async { Json(HealthResp { status: "ok" }) }));

        // /quote endpoint
        let quote_route = Router::new().route("/quote", get(move |Query(params): Query<QuoteParams>| {
            let engine = engine.clone();
            async move {
                let sell = match Bytes::from_str(params.sell_token.as_str()) {
                    Ok(b) => b,
                    Err(_) => return Json(QuoteResp { amount_out: None, gas_estimate: None }),
                };
                let buy = match Bytes::from_str(params.buy_token.as_str()) {
                    Ok(b) => b,
                    Err(_) => return Json(QuoteResp { amount_out: None, gas_estimate: None }),
                };
                let amount = params.amount.unwrap_or(1_000_000_000_000_000_000u128);
                let quote = engine.quote(&sell, &buy, amount, None);
                Json(QuoteResp { amount_out: quote.amount_out.map(|x| x.to_string()), gas_estimate: quote.gas_estimate })
            }
        }));

        // Combine routes
        let app = health_route.merge(quote_route);

        let addr: std::net::SocketAddr = addr.parse().expect("invalid addr");
        println!("Starting API server on {}", addr);
        let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
        axum::serve(listener, app).await.expect("server failed");
    }
} 