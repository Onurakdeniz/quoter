//! Cache logic tests for price-quoter.

use price_quoter::cache::{QuoteCache, QuoteCacheKey, CachedQuote};
use tycho_simulation::tycho_common::Bytes;
use std::thread::sleep;
use std::time::Duration;

#[test]
fn test_cache_logic() {
    let mut cache = QuoteCache::new();
    let key = QuoteCacheKey {
        sell_token: Bytes::from([1u8; 20]),
        buy_token: Bytes::from([2u8; 20]),
        amount: 1000,
        block: 1,
    };
    let quote = CachedQuote {
        amount_out: 2000,
        route: vec![Bytes::from([1u8; 20]), Bytes::from([2u8; 20])],
        route_pools: vec![],
        mid_price: None,
        slippage_bps: None,
        spread_bps: None,
        block: 1,
        gross_amount_out: None,
        fee_bps: None,
        gas_estimate: None,
        price_impact_bps: None,
    };
    cache.insert(key.clone(), quote.clone());
    assert!(cache.get(&key).is_some());
    cache.invalidate_block(1);
    assert!(cache.get(&key).is_none());
    cache.insert(key.clone(), quote.clone());
    sleep(Duration::from_secs(2));
    cache.max_age = Duration::from_secs(1);
    cache.purge_expired();
    assert!(cache.get(&key).is_none());
} 