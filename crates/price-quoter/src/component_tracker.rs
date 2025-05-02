//! Handles Tycho Indexer stream, pool/token state, and ingestion.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use tycho_simulation::{
    evm::{
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            ekubo::state::EkuboState,
            filters::{balancer_pool_filter, curve_pool_filter, uniswap_v4_pool_with_hook_filter},
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            uniswap_v4::state::UniswapV4State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    protocol::models::{BlockUpdate, ProtocolComponent},
    tycho_client::feed::component_tracker::ComponentFilter,
    models::Token,
};
use tycho_simulation::tycho_common::models::Chain;
use futures::Stream;
use async_stream::stream;
use futures::StreamExt;
use futures::stream::BoxStream;

pub type UpdateCallback = Box<dyn Fn(&BlockUpdate) + Send + Sync>;

/// Tracks all pools and tokens by ingesting Tycho Indexer stream.
#[derive(Clone)]
pub struct ComponentTracker {
    pub all_pools: Arc<RwLock<HashMap<String, ProtocolComponent>>>,
    pub pool_states: Arc<RwLock<HashMap<String, Box<dyn tycho_simulation::protocol::state::ProtocolSim + Send + Sync>>>>,
    pub all_tokens: Arc<RwLock<HashMap<tycho_simulation::tycho_common::Bytes, Token>>>,
    callbacks: Arc<Mutex<Vec<UpdateCallback>>>,
}

impl ComponentTracker {
    /// Create a new, empty tracker.
    pub fn new() -> Self {
        Self {
            all_pools: Arc::new(RwLock::new(HashMap::new())),
            pool_states: Arc::new(RwLock::new(HashMap::new())),
            all_tokens: Arc::new(RwLock::new(HashMap::new())),
            callbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a callback to be called on every BlockUpdate.
    pub fn register_callback<F>(&self, cb: F)
    where
        F: Fn(&BlockUpdate) + Send + Sync + 'static,
    {
        self.callbacks.lock().unwrap().push(Box::new(cb));
    }

    /// Notify all registered callbacks.
    fn notify_callbacks(&self, update: &BlockUpdate) {
        for cb in self.callbacks.lock().unwrap().iter() {
            cb(update);
        }
    }

    /// Connect to Tycho Indexer and yield BlockUpdate events.
    /// This is the main ingestion loop. Callers can use the updates to drive graph/state.
    pub async fn stream_updates(
        &self,
        tycho_url: &str,
        chain: Chain,
        api_key: &str,
        tvl_threshold: f64,
    ) -> anyhow::Result<BoxStream<'_, BlockUpdate>> {
        use tycho_simulation::utils::load_all_tokens;
        let all_tokens = load_all_tokens(
            tycho_url,
            false,
            Some(api_key),
            chain,
            None,
            None,
        )
        .await;
        *self.all_tokens.write().unwrap() = all_tokens.clone();
        let tvl_filter = ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold);
        let mut builder = ProtocolStreamBuilder::new(tycho_url, chain);
        // Register only Uniswap V2 and V3 exchanges that are supported
        builder = builder
            .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
            .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None);

        // Note: Additional protocols (Balancer, Curve, UniV4, etc.) currently disabled
        // The remote Tycho Indexer API doesn't support these protocols
        let mut protocol_stream = builder
            .auth_key(Some(api_key.to_string()))
            .skip_state_decode_failures(true)
            .set_tokens(all_tokens)
            .await
            .build()
            .await?;
        // Return a stream of BlockUpdate events
        // (In practice, you may want to spawn a task to process and update self.all_pools/pool_states)
        let stream = stream! {
            while let Some(msg) = protocol_stream.next().await {
                if let Ok(update) = msg {
                    {
                        let mut pools_w = self.all_pools.write().unwrap();
                        let mut states_w = self.pool_states.write().unwrap();
                        for (id, comp) in update.new_pairs.iter() {
                            pools_w.insert(id.clone(), comp.clone());
                        }
                        for id in update.removed_pairs.keys() {
                            pools_w.remove(id);
                            states_w.remove(id);
                        }
                        for (id, state) in update.states.iter() {
                            states_w.insert(id.clone(), state.clone());
                        }
                    }
                    self.notify_callbacks(&update);
                    yield update;
                }
            }
        };
        Ok(stream.boxed())
    }
    // TODO: Add callback registration, graph update hooks, etc.
} 