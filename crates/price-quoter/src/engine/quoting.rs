//! Core quoting logic, including PriceQuote and SinglePathQuote structs.

use tycho_simulation::tycho_common::Bytes;
use rust_decimal::Decimal;
use std::collections::HashMap;
use petgraph::prelude::{NodeIndex, EdgeIndex};
use crate::types::{QuoteRequest, QuoteResult, PriceQuoterError, Result as PriceQuoterResult};
use crate::config::AppConfig;
use std::str::FromStr; // Required for Bytes::from_str

// Constants for gas calculation
const GWEI_TO_NATIVE_CONVERSION_FACTOR: f64 = 1e-9; // 1 Gwei = 10^-9 Native Token (e.g., ETH)
// TODO: This should ideally come from config or be chain-specific
const NATIVE_TOKEN_DECIMALS: u32 = 18;
// Placeholder for native token address (e.g., WETH on Ethereum)
// TODO: This should be dynamically configurable based on the chain.

/// Holds calculated gas costs and the net amount out.
#[derive(Debug, Clone, Default)]
pub struct GasCostDetails {
    pub gas_cost_native: Option<f64>,      // Gas cost in the native chain token (e.g., ETH)
    pub gas_cost_token_out: Option<f64>,   // Gas cost expressed in the 'to_token'
    pub net_amount_out: f64,               // The amount_out after deducting gas_cost_token_out
}

/// Calculates gas costs and the net amount out from a gross amount.
///
/// # Arguments
/// * `gross_amount_out` - The amount out from a swap simulation before gas deduction.
/// * `num_swaps` - The number of swaps in the path.
/// * `avg_gas_units_per_swap` - Average gas units estimated per swap.
/// * `gas_price_gwei` - The current gas price in Gwei.
/// * `to_token_address` - The address of the output token.
/// * `native_token_address` - The address of the chain's native token (e.g., WETH).
/// * `get_native_token_price_in_token_out` - A function/closure that can provide the price
///   of 1 unit of native token in terms of `to_token`. This is crucial for accurate conversion.
///   Signature: `Fn(native_token: &Bytes, to_token: &Bytes) -> Option<f64>`
///
/// # Returns
/// A `GasCostDetails` struct.
pub fn calculate_gas_details<F>(
    gross_amount_out: f64,
    num_swaps: usize,
    avg_gas_units_per_swap: Option<u64>,
    gas_price_gwei: Option<u64>,
    _to_token_address: &Bytes, // Will be used with the pricing function
    _native_token_address: &Bytes, // Will be used with the pricing function
    get_native_token_price_in_token_out: F, // Placeholder for actual price conversion
) -> GasCostDetails
where
    F: Fn(&Bytes, &Bytes) -> Option<f64>,
{
    let mut details = GasCostDetails {
        gas_cost_native: None,
        gas_cost_token_out: None,
        net_amount_out: gross_amount_out,
    };

    if num_swaps == 0 { // No swaps, no direct swap gas cost from this path
        return details;
    }

    match (avg_gas_units_per_swap, gas_price_gwei) {
        (Some(gas_units_per_swap), Some(price_gwei)) => {
            let total_gas_units = gas_units_per_swap * num_swaps as u64;
            let gas_cost_in_gwei = total_gas_units * price_gwei;
            
            let cost_native = gas_cost_in_gwei as f64 * GWEI_TO_NATIVE_CONVERSION_FACTOR;
            details.gas_cost_native = Some(cost_native);

            // Use the provided closure to get the price of native token in terms of to_token
            if let Some(price_of_native_in_token_out) = get_native_token_price_in_token_out(_native_token_address, _to_token_address) {
                if price_of_native_in_token_out > 0.0 {
                    let cost_token_out = cost_native * price_of_native_in_token_out;
                    details.gas_cost_token_out = Some(cost_token_out);
                    details.net_amount_out = gross_amount_out - cost_token_out;
                    if details.net_amount_out < 0.0 {
                         details.net_amount_out = 0.0; // Cannot be negative
                    }
                } else {
                    // Price is zero or negative, cannot meaningfully convert gas cost.
                    // net_amount_out remains gross_amount_out.
                }
            } else {
                // Could not get price for conversion, net_amount_out remains gross_amount_out.
            }
        }
        _ => {
            // Not enough gas information to calculate, net_amount_out remains gross_amount_out
        }
    }

    details
}

/// Result of a price quote computation.
pub struct PriceQuote {
    pub amount_out: Option<u128>,
    pub route: Vec<Bytes>,
    /// price impact over the whole route, in bps (10 000 bps = 1 %)
    pub price_impact_bps: Option<Decimal>,
    pub mid_price: Option<Decimal>,
    pub slippage_bps: Option<Decimal>,
    pub fee_bps: Option<Decimal>,
    pub gas_estimate: Option<u64>,
    pub path_details: Vec<SinglePathQuote>, // For multi-path
    pub gross_amount_out: Option<u128>,
    pub spread_bps: Option<Decimal>,
    /// Depth metrics: Input amount required to cause X% slippage. Key: "0.5%", "1.0%", etc. Value: input amount (u128)
    pub depth_metrics: Option<HashMap<String, u128>>,
    /// If this quote was returned from the cache, which block it was cached at
    pub cache_block: Option<u64>,
}

/// Per-path quote details for multi-path evaluation
#[derive(Clone)]
pub struct SinglePathQuote {
    pub amount_out: Option<u128>,
    pub route: Vec<Bytes>,
    pub mid_price: Option<Decimal>,
    pub slippage_bps: Option<Decimal>,
    pub fee_bps: Option<Decimal>,
    pub gas_estimate: Option<u64>,
    pub gross_amount_out: Option<u128>,
    pub spread_bps: Option<Decimal>,
    /// price impact for this path, in bps
    pub price_impact_bps: Option<Decimal>,
    pub pools: Vec<String>,
    pub input_amount: Option<u128>,
    pub node_path: Vec<NodeIndex>,
    pub edge_seq: Vec<EdgeIndex>,
    pub gas_cost_native: Option<Decimal>, // Added
    pub gas_cost_in_token_out: Option<Decimal>, // Added
}

// Placeholder for PriceEngine methods that will be moved or called from here
// For now, this file will primarily hold the struct definitions and potentially
// the quote, quote_multi, and quote_single_path_with_edges methods later.

pub fn invalid_path_quote(path: &[NodeIndex], edge_seq: &[EdgeIndex], amount_in: u128) -> SinglePathQuote {
    SinglePathQuote {
        amount_out: None,
        route: Vec::new(), // Or construct from path if possible
        mid_price: None,
        slippage_bps: None,
        fee_bps: None,
        gas_estimate: None,
        gross_amount_out: None,
        spread_bps: None,
        price_impact_bps: None,
        pools: Vec::new(), // Or construct from edge_seq if possible
        input_amount: Some(amount_in),
        node_path: path.to_vec(),
        edge_seq: edge_seq.to_vec(),
        gas_cost_native: None,
        gas_cost_in_token_out: None,
    }
}

// --- Illustrative Main Quoting Function with Gas Calculation ---

// Placeholder for a function that simulates a swap and returns gross amount and path
fn simulate_swap_path_mock(
    _from_token: &Bytes,
    _to_token: &Bytes,
    amount_in: f64,
    _max_hops: Option<usize>,
    // In a real system, this would take a graph, data sources, etc.
) -> PriceQuoterResult<(f64, Vec<Bytes>, f64, Option<f64>)> { // returns (gross_amount_out, path, total_fee, estimated_slippage)
    // Mock implementation:
    if amount_in <= 0.0 {
        return Err(PriceQuoterError::SimulationError("Input amount must be positive".to_string()));
    }
    let gross_out = amount_in * 0.98; // e.g., 2% total fee/slippage effect
    let total_fee_mock = amount_in * 0.003; // e.g. 0.3% fee
    let slippage_mock = Some(amount_in * 0.002); // e.g. 0.2% slippage

    // Construct a plausible path, ensuring from_token and to_token are part of it.
    let mut path_nodes = vec![_from_token.clone()];
    // Add intermediate if not direct
    if _from_token.to_string() != _to_token.to_string() {
        // Using a known address string that can be parsed into Bytes
        let intermediate_token_str = "0x0000000000000000000000000000000000000001";
        if let Ok(intermediate_bytes) = Bytes::from_str(intermediate_token_str) {
             if _from_token != &intermediate_bytes && _to_token != &intermediate_bytes {
                path_nodes.push(intermediate_bytes);
            }
        } else {
            // Fallback if intermediate token parsing fails, though ideally Bytes::from_str should work for valid hex
        }
    }
    path_nodes.push(_to_token.clone());
    
    Ok((gross_out, path_nodes, total_fee_mock, slippage_mock))
}

// Mock function to get the price of 1 native token in terms of output_token.
fn mock_get_native_price_in_output_token(
    native_token_address: &Bytes,
    output_token_address: &Bytes,
    config: &AppConfig, // May need config for chain-specific logic or RPCs
                         // In a real scenario, this would need access to the pricing engine/cache itself.
) -> Option<f64> {
    // If output token is the native token, price is 1.0 (after decimal adjustment if amounts were integers)
    if native_token_address == output_token_address {
        return Some(1.0);
    }

    // Example: 1 Native Token (e.g., ETH) = 3000 USDC.
    // This is a simplified placeholder. A real implementation would query the price quoter (carefully, to avoid recursion)
    // or use a trusted external/cached price for the native token.
    // Using string comparison for placeholder addresses:
    let usdc_address_placeholder = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48".to_lowercase();
    // We need to compare the config's native_token_address here
    if let Some(conf_native_addr) = &config.native_token_address {
        if output_token_address.to_string().to_lowercase() == usdc_address_placeholder && 
           native_token_address == conf_native_addr { // Compare against the dynamically passed native_token_address
            return Some(3000.0); // 1 Native (ETH) = 3000 Output Token (USDC)
        }
    }
    
    // Fallback: Unable to determine price for this mock.
    None
}

/// Illustrative function to generate a quote including gas cost considerations.
/// This demonstrates how `calculate_gas_details` would be integrated.
pub fn generate_quote_with_gas(
    request: &QuoteRequest,
    config: &AppConfig,
) -> PriceQuoterResult<QuoteResult> {
    // 1. Simulate the swap to get gross amount out and path
    // In a real system, this would involve pathfinding and detailed simulation.
    let (gross_amount_out, path, total_fee, estimated_slippage) = simulate_swap_path_mock(
        &request.from_token,
        &request.to_token,
        request.amount_in,
        request.max_hops,
    )?;

    let num_swaps = if path.len() > 0 { path.len() - 1 } else { 0 };

    // 2. Get native token address from config
    let native_token_addr = config.native_token_address.as_ref()
        .ok_or_else(|| PriceQuoterError::ConfigError("Native token address not configured in AppConfig".to_string()))?;

    // 3. Define the gas cost conversion function (closure)
    let price_conversion_fn = |native_addr: &Bytes, output_addr: &Bytes| {
        // In a real system, this would call an internal, gas-less pricing function or use a cache.
        // For this illustration, it calls the mock function.
        mock_get_native_price_in_output_token(native_addr, output_addr, config)
    };

    // 4. Calculate gas details
    let gas_details = calculate_gas_details(
        gross_amount_out,
        num_swaps,
        config.avg_gas_units_per_swap,
        config.gas_price_gwei,
        &request.to_token,      // to_token from the original request
        native_token_addr,     // The chain's native token address, already checked for Some
        price_conversion_fn,
    );

    // 5. Construct the final QuoteResult using the comprehensive details from gas_details
    Ok(QuoteResult {
        amount_out_gross: gross_amount_out,
        path,
        total_fee, // This was from the mock simulation, representing DEX fees
        estimated_slippage, // From mock simulation
        gas_cost_native: gas_details.gas_cost_native,
        gas_cost_token_out: gas_details.gas_cost_token_out,
        amount_out_net: gas_details.net_amount_out, // Directly use net_amount_out from gas_details
    })
}

// --- End of Illustrative Main Quoting Function ---

// --- Continuous Price Updater ---
// Placed here due to inability to create a new file like engine/price_updater.rs
// Ideally, this would be in its own module.

use crate::data_management::cache::{QuoteCache, CachedContinuousPrice};
use crate::data_management::component_tracker::ComponentTracker;
use crate::engine::PriceEngine; // Assuming PriceEngine is in crate::engine
use std::sync::{Arc, RwLock, Mutex};
use tokio_stream::StreamExt; // For updates.next().await
use tracing::{info, error, warn};
use rust_decimal::Decimal;
use std::fs::{File, OpenOptions};
use std::io::{Write, BufReader, BufRead};
use std::collections::HashSet;


// Represents the information needed to price a token against the chosen numeraire
// struct TrackedTokenPriceRequest { // This struct seems unused now
//     token_to_price: Bytes,
//     numeraire_token: Bytes,
//     probe_amount_for_numeraire: f64,
// }

pub struct ContinuousPriceUpdater {
    config: Arc<AppConfig>,
    price_cache: Arc<RwLock<QuoteCache>>,
    price_engine: Arc<PriceEngine>, 
    tracker: Arc<ComponentTracker>,   
    global_numeraire: Bytes,
    tokens_for_history: HashSet<Bytes>,
    price_history_writer: Option<Arc<Mutex<csv::Writer<File>>>>,
}

impl ContinuousPriceUpdater {
    pub fn new(
        config: Arc<AppConfig>,
        price_cache: Arc<RwLock<QuoteCache>>,
        price_engine: Arc<PriceEngine>,
        tracker: Arc<ComponentTracker>,
    ) -> PriceQuoterResult<Self> {
        let global_numeraire = config.numeraire_token.clone().ok_or_else(|| {
            PriceQuoterError::ConfigError(
                "Global numeraire_token must be configured for ContinuousPriceUpdater.".to_string(),
            )
        })?;

        let mut tokens_for_history = HashSet::new();
        if let Some(tokens_file_path) = &config.tokens_file {
            info!("Loading tokens for price history from: {}", tokens_file_path);
            match File::open(tokens_file_path) {
                Ok(file) => {
                    let reader = BufReader::new(file);
                    for line in reader.lines() {
                        match line {
                            Ok(addr_str) => {
                                if addr_str.starts_with('#') || addr_str.trim().is_empty() { continue; }
                                match Bytes::from_str(addr_str.trim()) {
                                    Ok(token_bytes) => {
                                        tokens_for_history.insert(token_bytes);
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse token address '{}' from tokens file for history: {}", addr_str, e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to read line from tokens file for history: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to open tokens_file '{}' for history: {}. No specific tokens will be tracked for history.", tokens_file_path, e);
                }
            }
        }

        let price_history_writer = if let Some(history_file_path) = &config.price_history_file {
            let is_new_file = !std::path::Path::new(history_file_path).exists();
            match OpenOptions::new().append(true).create(true).open(history_file_path) {
                Ok(file) => {
                    let mut writer = csv::Writer::from_writer(file);
                    if is_new_file {
                        info!("Price history file {} created. Writing header.", history_file_path);
                        if let Err(e) = writer.write_record(&["block_number", "token_address", "price", "numeraire_address"]) {
                            error!("Failed to write header to price history file {}: {}", history_file_path, e);
                            None // Disable writer if header fails
                        } else {
                            if let Err(e) = writer.flush() {
                                error!("Failed to flush header to price history file {}: {}", history_file_path, e);
                            }
                            Some(Arc::new(Mutex::new(writer)))
                        }
                    } else {
                        info!("Appending to existing price history file: {}", history_file_path);
                        Some(Arc::new(Mutex::new(writer)))
                    }
                }
                Err(e) => {
                    error!("Failed to open or create price history file {}: {}. Price history will be disabled.", history_file_path, e);
                    None
                }
            }
        } else {
            None
        };

        if price_history_writer.is_some() {
            if config.tokens_file.is_some() {
                info!("Price history enabled. Logging specific tokens from: {:?}. Total: {}", config.tokens_file.as_ref().unwrap(), tokens_for_history.len());
            } else {
                info!("Price history enabled. Logging all calculated token prices to: {:?}", config.price_history_file.as_ref().unwrap());
            }
        } else {
            info!("Price history saving is disabled.");
        }
        
        Ok(Self {
            config: config.clone(), // Clone config for ownership if needed later, or ensure all uses are through Arc
            price_cache,
            price_engine,
            tracker,
            global_numeraire,
            tokens_for_history,
            price_history_writer,
        })
    }

    // Main loop for the updater
    pub async fn run(&self) {
        info!(
            "ContinuousPriceUpdater started. Numeraire: {:?}, Tycho URL: {:?}",
            self.global_numeraire, self.config.tycho_rpc_url
        );

        let tycho_url = self.config.tycho_rpc_url.clone().unwrap_or_default();
        let tycho_api_key = self.config.tycho_api_key.clone();

        // Create the Tycho update stream
        // stream_updates takes &self, so tracker needs to be Arc'd or PriceUpdater needs to own it.
        // Assuming tracker is Arc<ComponentTracker> as passed in new()
        let mut updates = self.tracker.stream_updates(
            tycho_url,
            self.config.chain.clone(), // Assuming AppConfig.chain is suitable
            tycho_api_key,
            Some(self.config.tvl_update_threshold_usd.unwrap_or(1000.0) as u64), // Example threshold
        );

        info!("Tycho update stream initiated.");

        while let Some(block_update_result) = updates.next().await {
            match block_update_result {
                Ok(block_data) => {
                    let block_for_this_update = block_data.block_number;
                    info!("Received block update for block: {}", block_for_this_update);

                    self.price_engine.update_graph_from_tracker_state();
                    info!("PriceEngine graph updated for block: {}", block_for_this_update);
                    
                    let tokens_to_update = match self.price_engine.graph.read() { // Renamed for clarity
                        Ok(graph_guard) => graph_guard.get_all_token_addresses(),
                        Err(e) => {
                            error!("Failed to get read lock on graph: {}. Skipping update for block {}", e, block_for_this_update); // Corrected variable name
                            continue;
                        }
                    };

                    if tokens_to_update.is_empty() {
                        info!("No tokens found in the graph to update for block {}.", block_for_this_update);
                        continue;
                    }
                    info!("Found {} tokens in graph to update prices for block {}.", tokens_to_update.len(), block_for_this_update);

                    for token_addr in tokens_to_update {
                        let price_option: Option<Decimal>;
                        if token_addr == self.global_numeraire {
                            price_option = Some(Decimal::ONE);
                            // info!("Price for numeraire token {} (self-price) is 1.0", token_addr);
                        } else {
                            // PriceEngine's get_token_price uses its configured numeraire and probe depth
                            price_option = self.price_engine.get_token_price(&token_addr, Some(block_for_this_update));
                            // match price_option {
                            //     Some(price) => info!("Calculated price for token {}: {} (vs numeraire {}) at block {}", token_addr, price, self.global_numeraire, block_for_this_update),
                            //     None => warn!("Could not calculate price for token {} at block {}", token_addr, block_for_this_update),
                            // }
                        }

                        let cached_price = CachedContinuousPrice {
                            price: price_option,
                            block: block_for_this_update,
                        };
                        
                        match self.price_cache.write() {
                            Ok(mut cache_guard) => {
                                cache_guard.update_continuous_price(token_addr.clone(), cached_price.clone());
                                if price_option.is_some() {
                                   // info!("Updated price for token {}: {} at block {}", token_addr, price_option.unwrap(), block_for_this_update);
                                } else {
                                   // warn!("Stored None price for token {} at block {}", token_addr, block_for_this_update);
                                }
                            }
                            Err(e) => {
                                error!("Failed to get write lock on price_cache for token {}: {}. Skipping cache update.", token_addr, e);
                            }
                        }
                    }
                    info!("Finished price updates for block: {}", block_for_this_update);
                }
                Err(e) => {
                    error!("Error receiving block update from Tycho stream: {}", e);
                    // Depending on the error, might need to decide whether to break or continue
                }
            }
        }
        warn!("Tycho update stream ended.");
    }

    // get_tokens_to_update, add_tracked_token, remove_tracked_token, list_tracked_tokens
    // are now obsolete as we price all tokens from the graph.
    // get_current_block is also obsolete as block number comes from Tycho.
}