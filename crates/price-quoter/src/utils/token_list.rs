use std::str::FromStr;
use tycho_simulation::tycho_common::Bytes;
use anyhow::anyhow;
use serde::Deserialize;

/// Load a list of token addresses from a JSON `["0x..."]` or TOML `tokens=[...]` file.
/// Accepts absolute or relative path.
pub fn load_token_list<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Vec<Bytes>> {
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow!("unable to read token list {}: {}", path.as_ref().display(), e))?;

    // 1. Try JSON array
    if let Ok(vec) = serde_json::from_str::<Vec<String>>(&text) {
        return parse_addresses(vec);
    }

    // 2. Try TOML with wrapper
    #[derive(Deserialize)]
    struct Wrapper { tokens: Vec<String> }
    let wrapper: Wrapper = toml::from_str(&text)
        .map_err(|e| anyhow!("token list {} is not valid JSON nor TOML: {}", path.as_ref().display(), e))?;
    parse_addresses(wrapper.tokens)
}

fn parse_addresses(list: Vec<String>) -> anyhow::Result<Vec<Bytes>> {
    list.into_iter()
        .map(|s| Bytes::from_str(&s).map_err(|e| anyhow!("invalid address {}: {}", s, e)))
        .collect()
} 