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
use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use tokio::time::{interval, Duration};
use tracing::{info, error, warn};
use std::fs::File;
use std::io::{BufReader, BufRead};


// Represents the information needed to price a token against the chosen numeraire
struct TrackedTokenPriceRequest {
    token_to_price: Bytes,
    numeraire_token: Bytes,
    probe_amount_for_numeraire: f64,
}

pub struct ContinuousPriceUpdater {
    config: Arc<AppConfig>,
    // The QuoteCache needs to be Arc<RwLock<>> because the updater runs in a separate async task
    // and will mutate it, while other parts of the application might read it.
    price_cache: Arc<RwLock<QuoteCache>>,
    tracked_tokens: Arc<RwLock<HashSet<Bytes>>>,
    global_numeraire: Bytes,
    // tycho_listener: TychoListener, // Placeholder for actual Tycho integration
    current_block: Arc<RwLock<u64>>, // To associate prices with a block number
}

impl ContinuousPriceUpdater {
    pub fn new(config: Arc<AppConfig>, price_cache: Arc<RwLock<QuoteCache>>) -> PriceQuoterResult<Self> {
        let global_numeraire = config.numeraire_token.clone()
            .ok_or_else(|| PriceQuoterError::ConfigError("Global numeraire_token must be configured for ContinuousPriceUpdater.".to_string()))?;

        let mut initial_tracked_tokens = HashSet::new();
        if let Some(tokens_file_path) = &config.tokens_file {
            info!("Loading initial tracked tokens from: {}", tokens_file_path);
            match File::open(tokens_file_path) {
                Ok(file) => {
                    let reader = BufReader::new(file);
                    for line in reader.lines() {
                        match line {
                            Ok(addr_str) => {
                                if addr_str.starts_with("#") || addr_str.trim().is_empty() { continue; }
                                match Bytes::from_str(addr_str.trim()) {
                                    Ok(token_bytes) => {
                                        initial_tracked_tokens.insert(token_bytes);
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse token address '{}' from tokens file: {}", addr_str, e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to read line from tokens file: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to open tokens_file '{}': {}. No initial tokens loaded.", tokens_file_path, e);
                }
            }
        }
        if initial_tracked_tokens.is_empty() {
            info!("No initial tokens loaded for continuous tracking. Add tokens via API or tokens_file.");
        }

        Ok(Self {
            config,
            price_cache,
            tracked_tokens: Arc::new(RwLock::new(initial_tracked_tokens)),
            global_numeraire,
            current_block: Arc::new(RwLock::new(0)), // Initialize block number
        })
    }

    // Main loop for the updater
    pub async fn run(&self) {
        info!("ContinuousPriceUpdater started. Numeraire: {:?}", self.global_numeraire);
        // TODO: Replace interval with actual Tycho block event listener
        // For now, simulate block ticks and increment block number

        // Determine block time based on the chain
        let block_time_duration = match self.config.chain {
            // Assuming tycho_simulation::tycho_common::models::Chain is the type of self.config.chain
            // And assuming variants like Ethereum, Base. Add other chains as needed.
            // tycho_simulation::tycho_common::models::Chain::Ethereum => Duration::from_secs(12),
            // tycho_simulation::tycho_common::models::Chain::Base => Duration::from_secs(2),
            // tycho_simulation::tycho_common::models::Chain::Arbitrum => Duration::from_millis(250),
            _ => Duration::from_secs(12), // Default block time if chain not matched or variants unknown
        };
        info!("Using block time duration for updates: {:?}", block_time_duration);
        let mut tick_interval = interval(block_time_duration);

        loop {
            tick_interval.tick().await; // Wait for new block/tick
            
            let mut current_block_guard = self.current_block.write().unwrap();
            *current_block_guard += 1; // Increment block number
            let block_for_this_update = *current_block_guard;
            drop(current_block_guard);

            info!("Updating prices for block: {}", block_for_this_update);

            let tokens_to_update = self.get_tokens_to_update();
            if tokens_to_update.is_empty() {
                info!("No tokens currently tracked for continuous price updates.");
                continue;
            }
            
            // Use probe_depth from config for the amount_in of the numeraire token
            // Default to 1.0 unit of numeraire if probe_depth is not set or zero
            let probe_amount = self.config.probe_depth.map_or(1.0, |d| if d == 0 { 1.0 } else { d as f64 });

            for token_addr in tokens_to_update {
                if token_addr == self.global_numeraire {
                    // Price of numeraire in itself is 1, handle explicitly or skip
                    // For now, let's construct a simple QuoteResult for it.
                    let numeraire_price = QuoteResult {
                        amount_out_gross: 1.0,
                        path: vec![self.global_numeraire.clone(), self.global_numeraire.clone()],
                        total_fee: 0.0,
                        estimated_slippage: Some(0.0),
                        gas_cost_native: Some(0.0),
                        gas_cost_token_out: Some(0.0),
                        amount_out_net: 1.0,
                    };
                    let cached_price = CachedContinuousPrice::from((numeraire_price, block_for_this_update));
                    self.price_cache.write().unwrap().update_continuous_price(token_addr.clone(), cached_price);
                    info!("Updated price for numeraire token {} (self-price): 1.0", token_addr);
                    continue;
                }

                let request = QuoteRequest {
                    from_token: self.global_numeraire.clone(), // Price from Global Numeraire
                    to_token: token_addr.clone(),             // To the Token we are tracking
                    amount_in: probe_amount,                  // Probe with a standard amount of the numeraire
                    max_hops: self.config.max_hops,           // Use configured max_hops
                };

                match generate_quote_with_gas(&request, &self.config) {
                    Ok(quote_result) => {
                        let cached_price = CachedContinuousPrice::from((quote_result, block_for_this_update));
                        // Lock cache for write
                        let mut cache_guard = self.price_cache.write().unwrap();
                        cache_guard.update_continuous_price(token_addr.clone(), cached_price);
                        drop(cache_guard);
                        // info!("Updated price for token {}: Net Amount {:.6}", token_addr, cached_price.amount_out_net);
                    }
                    Err(e) => {
                        error!("Failed to update price for token {}: {}", token_addr, e);
                    }
                }
            }
            info!("Finished price updates for block: {}", block_for_this_update);
        }
    }

    fn get_tokens_to_update(&self) -> Vec<Bytes> {
        // For now, returns all tracked tokens. 
        // TODO: Could be enhanced to select tokens based on recent Tycho updates if available.
        let tracked_tokens_guard = self.tracked_tokens.read().unwrap();
        let tokens: Vec<Bytes> = tracked_tokens_guard.iter().cloned().collect();
        drop(tracked_tokens_guard);
        tokens
    }

    pub fn add_tracked_token(&self, token: Bytes) {
        let mut tracked_tokens_guard = self.tracked_tokens.write().unwrap();
        if tracked_tokens_guard.insert(token.clone()) {
            info!("Added token to continuous tracking: {}", token);
        } else {
            info!("Token {} is already being tracked.", token);
        }
    }

    pub fn remove_tracked_token(&self, token: &Bytes) {
        let mut tracked_tokens_guard = self.tracked_tokens.write().unwrap();
        if tracked_tokens_guard.remove(token) {
            info!("Removed token from continuous tracking: {}", token);
            // Also remove it from the continuous price cache explicitly
            self.price_cache.write().unwrap().continuous_prices.pop(token);
        } else {
            info!("Token {} was not in the tracking list.", token);
        }
    }
    
    pub fn list_tracked_tokens(&self) -> Vec<Bytes> {
        self.tracked_tokens.read().unwrap().iter().cloned().collect()
    }

    // Method to get the current block number used by the updater
    pub fn get_current_block(&self) -> u64 {
        *self.current_block.read().unwrap()
    }
} 