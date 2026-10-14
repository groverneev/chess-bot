mod attacks;
mod bench;
mod board;
mod data;
mod datagen;
mod eval;
mod hce;
mod history;
mod movegen;
mod movepick;
mod moves;
mod nnue;
mod search;
mod timeman;
mod trainer;
mod tt;
mod types;
mod uci;
mod zobrist;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize| args.get(i).map(|s| s.as_str());
    match arg(1) {
        Some("bench") => bench::run(arg(2).and_then(|s| s.parse().ok()).unwrap_or(bench::DEFAULT_DEPTH)),
        Some("datagen") => {
            let threads = arg(2).and_then(|s| s.parse().ok()).unwrap_or(1);
            let positions = arg(3).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
            let out = arg(4).unwrap_or("data.bin");
            let nodes = arg(5).and_then(|s| s.parse().ok()).unwrap_or(5000);
            datagen::run(threads, positions, out, nodes);
        }
        Some("train") => {
            if let Err(error) = trainer::run(trainer::Config {
                data: arg(2).expect("usage: train <data> <out-dir> [epochs] [lr] [wdl] [threads]").to_string(),
                out_dir: arg(3).unwrap_or("nets/run").to_string(),
                epochs: arg(4).and_then(|s| s.parse().ok()).unwrap_or(30),
                batch: 16384,
                lr: arg(5).and_then(|s| s.parse().ok()).unwrap_or(0.001),
                final_lr: 0.00001,
                wdl: arg(6).and_then(|s| s.parse().ok()).unwrap_or(0.25),
                threads: arg(7).and_then(|s| s.parse().ok()).unwrap_or(8),
            }) {
                eprintln!("training error: {error}");
                std::process::exit(1);
            }
        }
        _ => uci::Engine::new().run(),
    }
}
