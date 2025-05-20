use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3_asyncio::tokio::future_into_py;

use crate::config::AppConfig;
use crate::engine::PriceEngine;
use crate::data_management::component_tracker::ComponentTracker;
use crate::engine::graph::TokenGraph;
use crate::data_management::cache::QuoteCache;

use tycho_simulation::tycho_common::{Bytes, models::Chain};
use rust_decimal::Decimal;
use std::sync::{Arc, RwLock};
use std::str::FromStr;
use tokio_stream::StreamExt; // For stream.next().await
use crate::engine::quoting::{SinglePathQuote as RustSinglePathQuote, PriceQuote as RustPriceQuote}; // Alias Rust structs
use std::collections::HashMap; // For PyPriceQuote.depth_metrics

// Python representation for SinglePathQuote
#[pyclass(name = "SinglePathQuote")]
#[derive(Clone)]
struct PySinglePathQuote {
    #[pyo3(get)]
    amount_out: Option<u128>,
    #[pyo3(get)]
    route: Vec<String>, // Vec<Bytes> converted to Vec<String> (hex)
    #[pyo3(get)]
    mid_price: Option<String>, // Decimal to String
    #[pyo3(get)]
    slippage_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    fee_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    gas_estimate: Option<u64>, // Gas units
    #[pyo3(get)]
    gross_amount_out: Option<u128>,
    #[pyo3(get)]
    spread_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    price_impact_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    pools: Vec<String>,
    #[pyo3(get)]
    input_amount: Option<u128>,
    #[pyo3(get)]
    gas_cost_native: Option<String>, // Decimal to String
    #[pyo3(get)]
    gas_cost_in_token_out: Option<String>, // Decimal to String
}

impl From<RustSinglePathQuote> for PySinglePathQuote {
    fn from(rs_quote: RustSinglePathQuote) -> Self {
        PySinglePathQuote {
            amount_out: rs_quote.amount_out,
            route: rs_quote.route.into_iter().map(|b| format!("0x{}", hex::encode(b.as_ref()))).collect(),
            mid_price: rs_quote.mid_price.map(|d| d.to_string()),
            slippage_bps: rs_quote.slippage_bps.map(|d| d.to_string()),
            fee_bps: rs_quote.fee_bps.map(|d| d.to_string()),
            gas_estimate: rs_quote.gas_estimate,
            gross_amount_out: rs_quote.gross_amount_out,
            spread_bps: rs_quote.spread_bps.map(|d| d.to_string()),
            price_impact_bps: rs_quote.price_impact_bps.map(|d| d.to_string()),
            pools: rs_quote.pools,
            input_amount: rs_quote.input_amount,
            gas_cost_native: rs_quote.gas_cost_native.map(|d| d.to_string()),
            gas_cost_in_token_out: rs_quote.gas_cost_in_token_out.map(|d| d.to_string()),
        }
    }
}

// Python representation for PriceQuote
#[pyclass(name = "PriceQuote")]
#[derive(Clone)]
struct PyPriceQuote {
    #[pyo3(get)]
    amount_out: Option<u128>,
    #[pyo3(get)]
    route: Vec<String>, // Vec<Bytes> converted to Vec<String> (hex)
    #[pyo3(get)]
    price_impact_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    mid_price: Option<String>, // Decimal to String
    #[pyo3(get)]
    slippage_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    fee_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    gas_estimate: Option<u64>, // Gas units
    #[pyo3(get)]
    path_details: Vec<PySinglePathQuote>,
    #[pyo3(get)]
    gross_amount_out: Option<u128>,
    #[pyo3(get)]
    spread_bps: Option<String>, // Decimal to String
    #[pyo3(get)]
    depth_metrics: Option<HashMap<String, u128>>,
    #[pyo3(get)]
    cache_block: Option<u64>,
}

impl From<RustPriceQuote> for PyPriceQuote {
    fn from(rs_quote: RustPriceQuote) -> Self {
        PyPriceQuote {
            amount_out: rs_quote.amount_out,
            route: rs_quote.route.into_iter().map(|b| format!("0x{}", hex::encode(b.as_ref()))).collect(),
            price_impact_bps: rs_quote.price_impact_bps.map(|d| d.to_string()),
            mid_price: rs_quote.mid_price.map(|d| d.to_string()),
            slippage_bps: rs_quote.slippage_bps.map(|d| d.to_string()),
            fee_bps: rs_quote.fee_bps.map(|d| d.to_string()),
            gas_estimate: rs_quote.gas_estimate,
            path_details: rs_quote.path_details.into_iter().map(PySinglePathQuote::from).collect(),
            gross_amount_out: rs_quote.gross_amount_out,
            spread_bps: rs_quote.spread_bps.map(|d| d.to_string()),
            depth_metrics: rs_quote.depth_metrics,
            cache_block: rs_quote.cache_block,
        }
    }
}


#[pyclass(name = "PriceQuoter")]
struct PyPriceQuoter {
    engine: Arc<PriceEngine>,
    runtime: Arc<tokio::runtime::Runtime>, 
}

#[pymethods]
impl PyPriceQuoter {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tycho_url: String,
        tycho_api_key: String,
        chain_name: String,
        rpc_url: Option<String>,
        native_token_address_hex: Option<String>,
        numeraire_token_hex: Option<String>,
        probe_depth: Option<u128>,
        max_hops: Option<usize>,
        avg_gas_units_per_swap: Option<u64>,
        gas_price_gwei: Option<u64>,
        tvl_threshold: Option<f64>,
    ) -> PyResult<Self> {
        let config = AppConfig::for_python_bindings(
            tycho_url,
            tycho_api_key,
            chain_name,
            rpc_url,
            native_token_address_hex,
            numeraire_token_hex,
            probe_depth,
            max_hops,
            avg_gas_units_per_swap,
            gas_price_gwei,
            tvl_threshold,
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to create AppConfig: {}", e)))?;

        let tracker = ComponentTracker::new(config.chain.clone(), Some(config.tvl_threshold as u64), None);
        let graph = Arc::new(RwLock::new(TokenGraph::new()));
        let cache = Arc::new(RwLock::new(QuoteCache::new()));
        
        let price_engine = PriceEngine::from_config(tracker, graph.clone(), cache.clone(), &config);
        let engine_arc = Arc::new(price_engine);

        // Synchronous warm-up
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;
        
        let initial_tracker_ref = engine_arc.tracker.clone(); // Clone Arc for the async block
        let initial_engine_ref = engine_arc.clone();

        rt.block_on(async {
            let mut stream = initial_tracker_ref.stream_updates(
                config.tycho_url.clone(), // Use URL from config
                config.chain.clone(), 
                Some(config.tycho_api_key.clone()), // Use API key from config
                Some(config.tvl_threshold as u64)
            );
            // .await was removed from stream_updates as it returns the stream directly
            
            if let Some(update_result) = stream.next().await {
                 match update_result {
                    Ok(_block_data) => {
                        // Data is processed by the tracker internally when stream.next() is called.
                        // Now, update the graph based on the tracker's current state.
                        initial_engine_ref.update_graph_from_tracker_state();
                        Ok(())
                    }
                    Err(e) => {
                        Err(PyValueError::new_err(format!("Failed during initial data fetch: {}", e)))
                    }
                 }
            } else {
                Err(PyValueError::new_err("Failed to get any data from initial stream.".to_string()))
            }
        })?;


        Ok(PyPriceQuoter { 
            engine: engine_arc,
            runtime: Arc::new(rt), // Store runtime
        })
    }

    fn get_token_price_vs_numeraire(&self, token_address_hex: String, py: Python) -> PyResult<PyObject> {
        let engine = self.engine.clone(); 
        let token_bytes = Bytes::from_str(&token_address_hex)
            .map_err(|e| PyValueError::new_err(format!("Invalid token address: {}", e)))?;

        future_into_py(py, async move {
            match engine.get_token_price(&token_bytes, None).await {
                Some(price_decimal) => Ok(price_decimal.to_string()), 
                None => Ok(Python::with_gil(|py| py.None())),
            }
        })
    }

    fn get_quote(
        &self, 
        token_in_hex: String, 
        token_out_hex: String, 
        amount_in_str: String, 
        k_paths: usize, 
        py: Python
    ) -> PyResult<PyObject> {
        let engine = self.engine.clone();
        let token_in = Bytes::from_str(&token_in_hex)
            .map_err(|e| PyValueError::new_err(format!("Invalid token_in_hex: {}", e)))?;
        let token_out = Bytes::from_str(&token_out_hex)
            .map_err(|e| PyValueError::new_err(format!("Invalid token_out_hex: {}", e)))?;
        let amount_in = amount_in_str.parse::<u128>()
            .map_err(|e| PyValueError::new_err(format!("Invalid amount_in_str: {}", e)))?;

        future_into_py(py, async move {
            // Call the async quote_multi method
            let rust_price_quote = engine.quote_multi(&token_in, &token_out, amount_in, k_paths, None).await;
            // Convert to PyPriceQuote
            let py_price_quote = PyPriceQuote::from(rust_price_quote);
            Ok(py_price_quote)
        })
    }
}

// Module definition
#[pymodule]
fn price_quoter_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPriceQuoter>()?;
    m.add_class::<PySinglePathQuote>()?;
    m.add_class::<PyPriceQuote>()?;
    Ok(())
}
