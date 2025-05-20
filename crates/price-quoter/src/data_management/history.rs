use tycho_simulation::tycho_common::Bytes;
use rust_decimal::Decimal;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// Append a token vs ETH price row to CSV.
/// If the file does not yet exist it writes a header first.
pub fn append_token_price(
    timestamp: &str,
    block: u64,
    token: &Bytes,
    eth_amount_out: u128,
    mid_price: Option<Decimal>,
    out_path: &str,
) -> anyhow::Result<()> {
    let exists = Path::new(out_path).exists();
    let mut file = OpenOptions::new().create(true).append(true).open(out_path)?;
    if !exists {
        writeln!(file, "timestamp,block,token,amount_out,mid_price")?;
    }
    writeln!(
        file,
        "{},{},0x{},{},{}",
        timestamp,
        block,
        hex::encode(token),
        eth_amount_out,
        mid_price.map(|d| d.to_string()).unwrap_or_default()
    )?;
    Ok(())
} 