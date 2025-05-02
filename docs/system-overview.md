# System Overview

The price-quoter system is designed to compute token swap prices (quotes) in real time, using on-chain liquidity data. It supports both single-quote and batch (list) price calculation, and can be used as a CLI or as a library/API.

- **Core crate:** `crates/price-quoter` (library)
- **CLI crate:** `crates/price-quoter-cli` (command-line interface)

## Key Components

- **ComponentTracker:** Maintains up-to-date state of all pools, tokens, and their on-chain states by ingesting a stream of `BlockUpdate` events.
- **TokenGraph:** Represents the swap ecosystem as a directed graph of tokens and pools.
- **Pathfinder:** Enumerates all possible swap paths between tokens.
- **PriceEngine:** Central logic for price computation, simulation, and quote generation.
- **QuoteCache:** LRU cache for quote results and path enumerations.

For more details, see the Technical Flow and CLI Usage pages. 