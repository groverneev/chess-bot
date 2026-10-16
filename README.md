# Ferrum

A UCI chess engine written in Rust with no external crates.

Ferrum has a legal-move bitboard generator with build-time magic bitboards, a
principal variation search with modern pruning and Lazy SMP, a hand-crafted
evaluation, and an NNUE pipeline (self-play data generation, a CPU trainer, and
SIMD inference). Strength has not yet been measured against external engines.

## Building

```bash
cargo build --release          # binary at target/release/ferrum
cargo test --release
```

`.cargo/config.toml` builds with `-C target-cpu=native`, so a binary is tuned for
(and may only run on) the machine that built it.

The build embeds `nets/default.nnue` if present, or the file named by
`FERRUM_NET`. Without a network, Ferrum uses its hand-crafted evaluation.

```bash
FERRUM_NET=path/to/net.nnue cargo build --release
```

## Using it

Ferrum speaks UCI, so it works with any UCI GUI (Cute Chess, Arena, En Croissant,
BanksiaGUI) or match runner (fastchess, cutechess-cli).

| Option | Default | Range |
|---|---|---|
| `Hash` (MB) | 16 | 1–65536 |
| `Threads` | 1 | 1–512 |
| `Move Overhead` (ms) | 20 | 0–5000 |
| `Clear Hash` | button | |

`go` accepts `wtime`, `btime`, `winc`, `binc`, `movestogo`, `movetime`, `depth`,
`nodes` and `infinite`. Pondering, MultiPV, Chess960 and tablebases are not
supported.

Extra commands at the UCI prompt: `d` (print the board), `eval`, `perft <depth>`,
and `bench [depth]`.

## Command-line tools

```bash
ferrum bench [depth]                                        # fixed node-count signature + nps
ferrum datagen <threads> <positions> <out.bin> [soft_nodes]  # self-play training data (appends)
ferrum train <data.bin> <out-dir> [epochs] [lr] [wdl] [threads] [mirror]
```

`scripts/sprt.sh <new> <old>` runs an SPRT between two binaries with fastchess,
and `scripts/pvcheck.py <engine>` checks that each `bestmove` matches the last
reported PV. See `PROGRESS.md` for the tool setup, data formats, network
architecture, experiment log and roadmap.

## Layout

| Path | Contents |
|---|---|
| `build.rs` | Magic bitboard generation; network embedding |
| `src/board.rs`, `movegen.rs`, `moves.rs`, `attacks.rs` | Board representation, move generation, perft |
| `src/search.rs`, `movepick.rs`, `history.rs`, `tt.rs`, `timeman.rs` | Search |
| `src/hce.rs`, `nnue.rs`, `eval.rs` | Evaluation |
| `src/datagen.rs`, `data.rs`, `trainer.rs` | NNUE data generation and training |
| `src/uci.rs`, `bench.rs`, `main.rs` | Front end |
