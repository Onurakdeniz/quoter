//! Configuration loading, env vars, CLI flags.

use std::env;
use std::str::FromStr;
use tycho_simulation::tycho_common::models::Chain;
use tycho_simulation::tycho_common::Bytes;
use tracing::info;
use std::fs;
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
    pub max_hops: Option<usize>,
    pub numeraire_token: Option<Bytes>,
    pub probe_depth: Option<u128>,
    pub tokens_file: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FileConfig {
    pub tycho_url: Option<String>,
    pub tycho_api_key: Option<String>,
    pub chain: Option<String>,
    pub tvl_threshold: Option<f64>,
    pub rpc_url: Option<String>,
    pub gas_price_gwei: Option<u64>,
    pub max_hops: Option<usize>,
    pub numeraire_token: Option<String>,
    pub probe_depth: Option<u128>,
    pub tokens_file: Option<String>,
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
    pub max_hops: Option<usize>,
    #[arg(long)]
    pub numeraire_token: Option<String>,
    #[arg(long)]
    pub probe_depth: Option<u128>,
    #[arg(long)]
    pub tokens_file: Option<String>,
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
        let max_hops: Option<usize> = env::var("MAX_HOPS").ok().and_then(|s| s.parse().ok());
        if rpc_url.is_none() {
            info!("RPC_URL environment variable not set. Balancer/Curve simulations may be limited.");
        }

        // New optional env vars for P-2 feature
        let numeraire_token = env::var("NUMERAIRE_TOKEN").ok().and_then(|s| Bytes::from_str(&s).ok());
        let probe_depth = env::var("PROBE_DEPTH").ok().and_then(|s| s.parse().ok());

        Self {
            tycho_url,
            tycho_api_key,
            chain: chain_enum,
            tvl_threshold,
            rpc_url,
            gas_price_gwei,
            max_hops,
            numeraire_token,
            probe_depth,
            tokens_file: env::var("TOKENS_FILE").ok(),
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
            max_hops: None,
            numeraire_token: None,
            probe_depth: None,
            tokens_file: None,
        };
        if let Some(ref path) = cli.config {
            if let Ok(contents) = fs::read_to_string(path) {
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
        Self {
            tycho_url,
            tycho_api_key,
            chain: chain_enum,
            tvl_threshold,
            rpc_url,
            gas_price_gwei,
            max_hops,
            numeraire_token,
            probe_depth,
            tokens_file,
        }
    }
} 