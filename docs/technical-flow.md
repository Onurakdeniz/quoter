# Technical Flow

The price-quoter system processes token swap price calculations through a series of well-defined steps and modules:

1. **Data Ingestion:**
   - `ComponentTracker` ingests on-chain data and maintains the state of pools and tokens.
2. **Graph Construction:**
   - `TokenGraph` builds a directed graph of tokens and pools, updating on every state change.
3. **Pathfinding:**
   - `Pathfinder` enumerates all possible swap paths between tokens, up to a configurable hop limit.
4. **Price Calculation:**
   - `PriceEngine` simulates swaps along each path, accounting for fees, slippage, and gas, and computes all relevant metrics.
5. **Caching:**
   - `QuoteCache` stores results for efficiency and invalidates them on state changes.

## Metrics Computed
- Net and gross output amounts
- Gas cost
- Mid price, slippage, spread, price impact
- Depth metrics (input size for slippage thresholds)

This flow ensures accurate, real-time price quotes for both single and batch operations. 