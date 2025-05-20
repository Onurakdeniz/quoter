//! Caching for quotes and pathfinding.

use tycho_simulation::tycho_common::Bytes;
use std::time::{Duration, Instant};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
// use crate::types::QuoteResult; // Import the main QuoteResult for continuous pricing
use rust_decimal::Decimal;

/// Cache key for a quote: (sell_token, buy_token, amount, block number)
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct QuoteCacheKey {
    pub sell_token: Bytes,
    pub buy_token: Bytes,
    pub amount: u128,
    pub block: u64,
}

/// Cached quote result (can be expanded as needed)
#[derive(Clone, Debug)]
pub struct CachedQuote {
    pub amount_out: u128,
    pub route: Vec<Bytes>,
    pub route_pools: Vec<String>,
    pub mid_price: Option<rust_decimal::Decimal>,
    pub slippage_bps: Option<rust_decimal::Decimal>,
    pub spread_bps: Option<rust_decimal::Decimal>,
    pub block: u64,
    pub gross_amount_out: Option<u128>,
    pub fee_bps: Option<rust_decimal::Decimal>,
    pub gas_estimate: Option<u64>,
    pub price_impact_bps: Option<rust_decimal::Decimal>,
}

/// Cache key for a path: (sell_token, buy_token, block number, k)
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct PathCacheKey {
    pub sell_token: Bytes,
    pub buy_token: Bytes,
    pub block: u64,
    pub k: usize,
}

/// Cached K-path result
#[derive(Clone, Debug)]
pub struct CachedPaths {
    pub paths: Vec<Vec<Bytes>>,
    pub block: u64,
    pub timestamp: Instant,
}

/// Snapshot of cache hit/miss counters
pub struct CacheMetrics {
    pub quote_hits: usize,
    pub quote_misses: usize,
    pub path_hits: usize,
    pub path_misses: usize,
}

/// Cached continuous price result, derived from types::QuoteResult
#[derive(Clone, Debug)]
pub struct CachedContinuousPrice {
    pub price: Option<Decimal>, // Price in terms of the global numeraire
    pub block: u64,             // Block at which this price was calculated
    // Numeraire is implicit (the one configured globally for the continuous updater)
}

// impl From<(QuoteResult, u64)> for CachedContinuousPrice {
//     fn from(data: (QuoteResult, u64)) -> Self {
//         let (qr, block_num) = data;
//         CachedContinuousPrice {
//             amount_out_gross: qr.amount_out_gross,
//             path: qr.path,
//             total_fee: qr.total_fee,
//             estimated_slippage: qr.estimated_slippage,
//             gas_cost_native: qr.gas_cost_native,
//             gas_cost_token_out: qr.gas_cost_token_out,
//             amount_out_net: qr.amount_out_net,
//             block: block_num,
//         }
//     }
// }

/// Main quote cache structure with LRU and path cache
pub struct QuoteCache {
    pub quotes: LruCache<QuoteCacheKey, (CachedQuote, Instant)>,
    pub path_cache: LruCache<PathCacheKey, CachedPaths>,
    // Cache for continuously updated prices (TokenAddress -> Price in terms of global numeraire)
    // Key: Token address being priced. Value: (CachedContinuousPrice, Timestamp)
    pub continuous_prices: LruCache<Bytes, (CachedContinuousPrice, Instant)>,
    pub max_age: Duration, // General max age for reactive quotes and paths
    pub continuous_price_max_age: Duration, // Specific max age for continuously updated prices
    pub quote_hits: AtomicUsize,
    pub quote_misses: AtomicUsize,
    pub path_hits: AtomicUsize,
    pub path_misses: AtomicUsize,
}

impl QuoteCache {
    pub fn new() -> Self {
        Self {
            quotes: LruCache::new(NonZeroUsize::new(1000).unwrap()),
            path_cache: LruCache::new(NonZeroUsize::new(100).unwrap()),
            continuous_prices: LruCache::new(NonZeroUsize::new(5000).unwrap()), // Capacity for many tokens
            max_age: Duration::from_secs(60),
            continuous_price_max_age: Duration::from_secs(120), // e.g., ~10 blocks
            quote_hits: AtomicUsize::new(0),
            quote_misses: AtomicUsize::new(0),
            path_hits: AtomicUsize::new(0),
            path_misses: AtomicUsize::new(0),
        }
    }

    /// Get a cached quote if present and not expired.
    pub fn get(&mut self, key: &QuoteCacheKey) -> Option<&CachedQuote> {
        // First, check if entry exists and is still fresh **without** creating a long-lived
        // borrow so we can safely remove an expired entry afterwards.
        let is_expired = match self.quotes.peek(key) {
            Some((_, ts)) => ts.elapsed() >= self.max_age,
            None => false,
        };

        if is_expired {
            // Safe to mutate again – the previous peek borrow ended at the end of the match.
            self.quotes.pop(key);
            self.quote_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if let Some((quote, _)) = self.quotes.get(key) {
            self.quote_hits.fetch_add(1, Ordering::Relaxed);
            Some(quote)
        } else {
            self.quote_misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert or update a cached quote.
    pub fn insert(&mut self, key: QuoteCacheKey, value: CachedQuote) {
        self.quotes.put(key, (value, Instant::now()));
    }

    /// Invalidate all cache entries for a given block (or implement more granular invalidation).
    pub fn invalidate_block(&mut self, block: u64) {
        let keys_to_remove: Vec<_> = self.quotes.iter().filter(|(k, _)| k.block == block).map(|(k, _)| k.clone()).collect();
        for k in keys_to_remove { self.quotes.pop(&k); }
        let keys_to_remove_paths: Vec<_> = self.path_cache.iter().filter(|(k, _)| k.block == block).map(|(k, _)| k.clone()).collect();
        for k in keys_to_remove_paths { self.path_cache.pop(&k); }
        // Also invalidate continuous prices if they are older than the new block logic dictates
        // Or simply rely on TTL for continuous_prices for now, or a more specific invalidation if needed
        let keys_to_remove_continuous: Vec<_> = self.continuous_prices.iter().filter(|(_, (cp, _))| cp.block < block).map(|(k,_)| k.clone()).collect();
        for k in keys_to_remove_continuous { self.continuous_prices.pop(&k); }
    }

    /// Invalidate all cache entries that touch a given pool (per-pool/edge granularity)
    pub fn invalidate_pool(&mut self, pool_id: &str) {
        let keys_to_remove: Vec<_> = self.quotes.iter()
            .filter(|(_k, v)| v.0.route_pools.iter().any(|pid| pid == pool_id))
            .map(|(k, _)| k.clone()).collect();
        for k in keys_to_remove { self.quotes.pop(&k); }
        let keys_to_remove: Vec<_> = self.path_cache.iter()
            .filter(|(_k, v)| v.paths.iter().flatten().any(|addr| format!("{:x}", addr) == pool_id))
            .map(|(k, _)| k.clone()).collect();
        for k in keys_to_remove { self.path_cache.pop(&k); }
    }

    /// Remove expired entries
    pub fn purge_expired(&mut self) {
        let max_age = self.max_age;
        let keys_to_remove: Vec<_> = self.quotes.iter().filter(|(_, v)| v.1.elapsed() >= max_age).map(|(k, _)| k.clone()).collect();
        for k in keys_to_remove { self.quotes.pop(&k); }
        let path_keys_to_remove: Vec<_> = self.path_cache.iter().filter(|(_, v)| v.timestamp.elapsed() >= max_age).map(|(k, _)| k.clone()).collect();
        for k in path_keys_to_remove { self.path_cache.pop(&k); }
        // Purge expired continuous prices
        let continuous_max_age = self.continuous_price_max_age;
        let continuous_keys_to_remove: Vec<_> = self.continuous_prices.iter().filter(|(_,(_,ts))| ts.elapsed() >= continuous_max_age).map(|(k,_)| k.clone()).collect();
        for k in continuous_keys_to_remove { self.continuous_prices.pop(&k); }
    }

    /// Get cached K-paths if present and not expired
    pub fn get_paths(&mut self, key: &PathCacheKey) -> Option<&CachedPaths> {
        let is_expired = match self.path_cache.peek(key) {
            Some(paths) => paths.timestamp.elapsed() >= self.max_age,
            None => false,
        };

        if is_expired {
            self.path_cache.pop(key);
            self.path_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if let Some(paths) = self.path_cache.get(key) {
            self.path_hits.fetch_add(1, Ordering::Relaxed);
            Some(paths)
        } else {
            self.path_misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert K-paths into cache
    pub fn insert_paths(&mut self, key: PathCacheKey, value: CachedPaths) {
        self.path_cache.put(key, value);
    }

    /// Retrieve current cache metrics snapshot.
    pub fn metrics(&self) -> CacheMetrics {
        CacheMetrics {
            quote_hits: self.quote_hits.load(Ordering::Relaxed),
            quote_misses: self.quote_misses.load(Ordering::Relaxed),
            path_hits: self.path_hits.load(Ordering::Relaxed),
            path_misses: self.path_misses.load(Ordering::Relaxed),
        }
    }

    /// Clear all cached paths (use when underlying pool graph changed drastically)
    pub fn clear_paths(&mut self) {
        self.path_cache.clear();
    }

    /// Invalidate all cache entries for a given token (sell or buy)
    pub fn invalidate_token(&mut self, token: &Bytes) {
        // Remove quotes where token is either sell_token or buy_token
        let keys_to_remove: Vec<_> = self.quotes.iter()
            .filter(|(k, _)| &k.sell_token == token || &k.buy_token == token)
            .map(|(k, _)| k.clone())
            .collect();
        for k in keys_to_remove { self.quotes.pop(&k); }
        // Remove path entries where token is involved
        let path_keys_to_remove: Vec<_> = self.path_cache.iter()
            .filter(|(k, _)| &k.sell_token == token || &k.buy_token == token)
            .map(|(k, _)| k.clone())
            .collect();
        for k in path_keys_to_remove { self.path_cache.pop(&k); }
        // Invalidate continuous price for this token
        self.continuous_prices.pop(token);
    }

    // --- Methods for Continuous Price Cache ---

    /// Get a cached continuous price if present and not expired.
    /// Key is the token address whose price (vs global numeraire) is stored.
    pub fn get_continuous_price(&mut self, token_address: &Bytes) -> Option<&CachedContinuousPrice> {
        let max_age = self.continuous_price_max_age;
        let is_expired = match self.continuous_prices.peek(token_address) {
            Some((_, ts)) => ts.elapsed() >= max_age,
            None => false,
        };

        if is_expired {
            self.continuous_prices.pop(token_address);
            // Consider adding specific hit/miss counters for this cache if needed
            return None;
        }

        if let Some((price_info, _)) = self.continuous_prices.get(token_address) {
            Some(price_info)
        } else {
            None
        }
    }

    /// Insert or update a cached continuous price.
    /// Key is the token address being priced.
    pub fn update_continuous_price(&mut self, token_address: Bytes, price_info: CachedContinuousPrice) {
        self.continuous_prices.put(token_address, (price_info, Instant::now()));
    }

    /// Clear all continuously cached prices.
    pub fn clear_continuous_prices(&mut self) {
        self.continuous_prices.clear();
    }
} 