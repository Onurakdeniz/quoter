//! Common types, enums, error handling, data models.

use std::fmt;
use serde::{Serialize, Deserialize};
use tycho_simulation::tycho_common::Bytes;

/// Common error type for the price-quoter system.
#[derive(Debug)]
pub enum PriceQuoterError {
    IngestionError(String),
    GraphError(String),
    PathfindingError(String),
    SimulationError(String),
    ConfigError(String),
    Other(String),
}

impl fmt::Display for PriceQuoterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PriceQuoterError::IngestionError(e) => write!(f, "Ingestion error: {}", e),
            PriceQuoterError::GraphError(e) => write!(f, "Graph error: {}", e),
            PriceQuoterError::PathfindingError(e) => write!(f, "Pathfinding error: {}", e),
            PriceQuoterError::SimulationError(e) => write!(f, "Simulation error: {}", e),
            PriceQuoterError::ConfigError(e) => write!(f, "Config error: {}", e),
            PriceQuoterError::Other(e) => write!(f, "Other error: {}", e),
        }
    }
}

impl std::error::Error for PriceQuoterError {}

// Shared data models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteRequest {
    pub from_token: Bytes,
    pub to_token: Bytes,
    pub amount_in: f64,
    pub max_hops: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteResult {
    pub amount_out_gross: f64,
    pub path: Vec<Bytes>,
    pub total_fee: f64,
    pub estimated_slippage: Option<f64>,
    pub gas_cost_native: Option<f64>,
    pub gas_cost_token_out: Option<f64>,
    pub amount_out_net: f64,
}

pub type Result<T> = std::result::Result<T, PriceQuoterError>;

// TODO: Add shared types, error enums, and data models used across modules 