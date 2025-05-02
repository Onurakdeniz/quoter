# CLI Usage

The `price-quoter-cli` provides a command-line interface for interacting with the price-quoter system. It supports both single-quote and batch/list price calculation modes.

## Main Flow
1. Loads configuration (from environment, CLI args, or config file)
2. Starts the `ComponentTracker` and syncs state
3. Builds the `TokenGraph`
4. Instantiates the `PriceEngine`
5. Optionally updates gas price in the background

## Modes
- **Single-Quote Mode:**
  - Specify `--sell_token`, `--buy_token`, and optionally `--sell_amount`
  - CLI prints all found paths, best quote, and metrics
- **Batch/List Mode:**
  - Provide a token list file (`--tokens_file`)
  - Computes quotes for all tokens vs a numeraire (e.g., ETH)
  - Results are printed and saved to CSV

## Example Command
```sh
price-quoter-cli --tokens_file tokens.json --numeraire_token 0x...ETH
```

For each token, the CLI computes the best price/route, simulates all paths, and outputs the results. 