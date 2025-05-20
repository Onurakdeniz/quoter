//! Configuration loading, env vars, CLI flags.

use std::env;
use std::str::FromStr;
use tycho_simulation::tycho_common::models::Chain;
use tycho_simulation::tycho_common::Bytes;
use tracing::info;
use serde::Deserialize;

#[cfg(feature = "cli")]
use clap::Parser;

#[derive(Clone)]
pub struct AppConfig {
    pub tycho_url: String,
    pub tycho_api_key: String,
    pub chain: Chain,
    pub tvl_threshold: f64,
    pub rpc_url: Option<String>,
    pub gas_price_gwei: Option<u64>,
    pub avg_gas_units_per_swap: Option<u64>,
    pub native_token_address: Option<Bytes>,
    pub max_hops: Option<usize>,
    pub numeraire_token: Option<Bytes>,
    pub probe_depth: Option<u128>,
    pub tokens_file: Option<String>,
    pub sell_token_address: Option<String>,
    pub buy_token_address: Option<String>,
    pub sell_amount_value: Option<f64>,
    pub display_numeraire_token_address: Option<String>,
    pub price_history_file: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FileConfig {
    pub tycho_url: Option<String>,
    pub tycho_api_key: Option<String>,
    pub chain: Option<String>,
    pub tvl_threshold: Option<f64>,
    pub rpc_url: Option<String>,
    pub gas_price_gwei: Option<u64>,
    pub avg_gas_units_per_swap: Option<u64>,
    pub native_token_address: Option<String>,
    pub max_hops: Option<usize>,
    pub numeraire_token: Option<String>,
    pub probe_depth: Option<u128>,
    pub tokens_file: Option<String>,
    pub sell_token_address: Option<String>,
    pub buy_token_address: Option<String>,
    pub sell_amount_value: Option<f64>,
    pub display_numeraire_token_address: Option<String>,
}

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct CliConfig {
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long)]
    pub tycho_url: Option<String>,
    #[arg(long)]
    pub tycho_api_key: Option<String>,
    #[arg(long)]
    pub chain: Option<String>,
    #[arg(long)]
    pub tvl_threshold: Option<f64>,
    #[arg(long)]
    pub rpc_url: Option<String>,
    #[arg(long)]
    pub gas_price_gwei: Option<u64>,
    #[arg(long)]
    pub avg_gas_units_per_swap: Option<u64>,
    #[arg(long)]
    pub native_token_address: Option<String>,
    #[arg(long)]
    pub max_hops: Option<usize>,
    #[arg(long)]
    pub numeraire_token: Option<String>,
    #[arg(long)]
    pub probe_depth: Option<u128>,
    #[arg(long)]
    pub tokens_file: Option<String>,
    #[arg(long)]
    pub sell_token: Option<String>,
    #[arg(long)]
    pub buy_token: Option<String>,
    #[arg(long)]
    pub sell_amount: Option<f64>,
    #[arg(long)]
    pub display_numeraire_token: Option<String>,
    #[arg(long)]
    pub price_history_file: Option<String>,
}

impl AppConfig {
    pub fn load() -> Self {
        let chain = env::var("CHAIN").unwrap_or_else(|_| "ethereum".to_string());
        let chain_enum = Chain::from_str(&chain).unwrap_or(Chain::Ethereum);
        let tycho_url = env::var("TYCHO_URL").unwrap_or_else(|_| {
            match chain_enum {
                Chain::Ethereum => "tycho-beta.propellerheads.xyz".to_string(),
                Chain::Base => "tycho-base-beta.propellerheads.xyz".to_string(),
                Chain::Unichain => "tycho-unichain-beta.propellerheads.xyz".to_string(),
                _ => panic!("Unknown chain for default URL"),
            }
        });
        let tycho_api_key = env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());
        let tvl_threshold = env::var("TVL_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(100.0);

        let rpc_url = env::var("RPC_URL").ok();
        let gas_price_gwei = env::var("GAS_PRICE_GWEI").ok().and_then(|s| s.parse().ok());
        let avg_gas_units_per_swap = env::var("AVG_GAS_UNITS_PER_SWAP").ok().and_then(|s| s.parse().ok());
        let max_hops: Option<usize> = env::var("MAX_HOPS").ok().and_then(|s| s.parse().ok());
        if rpc_url.is_none() {
            info!("RPC_URL environment variable not set. Balancer/Curve simulations may be limited.");
        }

        // New optional env vars for P-2 feature
        let numeraire_token = env::var("NUMERAIRE_TOKEN").ok().and_then(|s| Bytes::from_str(&s).ok());
        let probe_depth = env::var("PROBE_DEPTH").ok().and_then(|s| s.parse().ok());
        let native_token_address = env::var("NATIVE_TOKEN_ADDRESS").ok().and_then(|s| Bytes::from_str(&s).ok());

        Self {
            tycho_url,
            tycho_api_key,
            chain: chain_enum,
            tvl_threshold,
            rpc_url,
            gas_price_gwei,
            avg_gas_units_per_swap,
            native_token_address,
            max_hops,
            numeraire_token,
            probe_depth,
            tokens_file: env::var("TOKENS_FILE").ok(),
            sell_token_address: None,
            buy_token_address: None,
            sell_amount_value: None,
            display_numeraire_token_address: None,
            price_history_file: env::var("PRICE_HISTORY_FILE").ok(),
        }
    }

    #[cfg(feature = "cli")]
    pub fn load_with_cli() -> Self {
        let cli = CliConfig::parse();
        let mut file_config = FileConfig {
            tycho_url: None,
            tycho_api_key: None,
            chain: None,
            tvl_threshold: None,
            rpc_url: None,
            gas_price_gwei: None,
            avg_gas_units_per_swap: None,
            max_hops: None,
            numeraire_token: None,
            probe_depth: None,
            tokens_file: None,
            sell_token_address: None,
            buy_token_address: None,
            sell_amount_value: None,
            display_numeraire_token_address: None,
            native_token_address: None,
            price_history_file: None,
        };
        if let Some(ref path) = cli.config {
            if let Ok(contents) = std::fs::read_to_string(path) {
                if let Ok(cfg) = toml::from_str::<FileConfig>(&contents) {
                    file_config = cfg;
                }
            }
        }
        let chain = cli.chain
            .or(file_config.chain)
            .or(env::var("CHAIN").ok())
            .unwrap_or_else(|| "ethereum".to_string());
        let chain_enum = Chain::from_str(&chain).unwrap_or(Chain::Ethereum);
        let tycho_url = cli.tycho_url
            .or(file_config.tycho_url)
            .or(env::var("TYCHO_URL").ok())
            .unwrap_or_else(|| match chain_enum {
                Chain::Ethereum => "tycho-beta.propellerheads.xyz".to_string(),
                Chain::Base => "tycho-base-beta.propellerheads.xyz".to_string(),
                Chain::Unichain => "tycho-unichain-beta.propellerheads.xyz".to_string(),
                _ => panic!("Unknown chain for default URL"),
            });
        let tycho_api_key = cli.tycho_api_key
            .or(file_config.tycho_api_key)
            .or(env::var("TYCHO_API_KEY").ok())
            .unwrap_or_else(|| "sampletoken".to_string());
        let tvl_threshold = cli.tvl_threshold
            .or(file_config.tvl_threshold)
            .or(env::var("TVL_THRESHOLD").ok().and_then(|s| s.parse().ok()))
            .unwrap_or(100.0);
        let rpc_url = cli.rpc_url
            .or(file_config.rpc_url)
            .or(env::var("RPC_URL").ok());
        let gas_price_gwei = cli.gas_price_gwei
            .or(file_config.gas_price_gwei)
            .or(env::var("GAS_PRICE_GWEI").ok().and_then(|s| s.parse().ok()));
        let avg_gas_units_per_swap = cli.avg_gas_units_per_swap
            .or(file_config.avg_gas_units_per_swap)
            .or(env::var("AVG_GAS_UNITS_PER_SWAP").ok().and_then(|s| s.parse().ok()));
        let native_token_address = cli.native_token_address
            .or(file_config.native_token_address)
            .or(env::var("NATIVE_TOKEN_ADDRESS").ok())
            .and_then(|s| Bytes::from_str(&s).ok());
        let max_hops = cli.max_hops
            .or(file_config.max_hops)
            .or(env::var("MAX_HOPS").ok().and_then(|s| s.parse().ok()));
        let numeraire_token = cli.numeraire_token
            .or(file_config.numeraire_token)
            .or(env::var("NUMERAIRE_TOKEN").ok())
            .and_then(|s| Bytes::from_str(&s).ok());
        let probe_depth = cli.probe_depth
            .or(file_config.probe_depth)
            .or(env::var("PROBE_DEPTH").ok().and_then(|s| s.parse().ok()));
        let tokens_file = cli.tokens_file
            .or(file_config.tokens_file)
            .or(env::var("TOKENS_FILE").ok());
        
        // Load direct quote operation params
        let sell_token_address = cli.sell_token
            .or(file_config.sell_token_address)
            .or(env::var("SELL_TOKEN").ok());
        let buy_token_address = cli.buy_token
            .or(file_config.buy_token_address)
            .or(env::var("BUY_TOKEN").ok());
        let sell_amount_value = cli.sell_amount
            .or(file_config.sell_amount_value)
            .or(env::var("SELL_AMOUNT").ok().and_then(|s| s.parse().ok()));
        let display_numeraire_token_address = cli.display_numeraire_token
            .or(file_config.display_numeraire_token_address)
            .or(env::var("DISPLAY_NUMERAIRE_TOKEN").ok());
        let price_history_file = cli.price_history_file
            .or(file_config.price_history_file)
            .or(env::var("PRICE_HISTORY_FILE").ok());

        Self {
            tycho_url,
            tycho_api_key,
            chain: chain_enum,
            tvl_threshold,
            rpc_url,
            gas_price_gwei,
            avg_gas_units_per_swap,
            native_token_address,
            max_hops,
            numeraire_token,
            probe_depth,
            tokens_file,
            sell_token_address,
            buy_token_address,
            sell_amount_value,
            display_numeraire_token_address,
            price_history_file,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn for_python_bindings(
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
        // tokens_file and price_history_file are omitted for Python bindings for now
        // as they are less likely to be configured this way.
    ) -> Result<Self, String> {
        let chain_enum = Chain::from_str(&chain_name)
            .map_err(|_| format!("Invalid chain name: {}", chain_name))?;

        let native_token_address = native_token_address_hex
            .map(|hex| Bytes::from_str(&hex))
            .transpose()
            .map_err(|e| format!("Invalid native_token_address: {}", e))?;

        let numeraire_token = numeraire_token_hex
            .map(|hex| Bytes::from_str(&hex))
            .transpose()
            .map_err(|e| format!("Invalid numeraire_token: {}", e))?;

        Ok(Self {
            tycho_url,
            tycho_api_key,
            chain: chain_enum,
            tvl_threshold: tvl_threshold.unwrap_or(100.0),
            rpc_url,
            gas_price_gwei,
            avg_gas_units_per_swap,
            native_token_address,
            max_hops,
            numeraire_token,
            probe_depth,
            tokens_file: None, // Not configured via Python for now
            sell_token_address: None, // Not relevant for general quoter init
            buy_token_address: None, // Not relevant for general quoter init
            sell_amount_value: None, // Not relevant for general quoter init
            display_numeraire_token_address: None, // Not relevant for general quoter init
            price_history_file: None, // Not configured via Python for now
        })
    }
}