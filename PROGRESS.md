# Ferrum: progress notes and handoff

Goal: build the strongest UCI chess engine we can, in Rust (std only, no crates).
Machine: Apple M3 (aarch64, 8 cores, 16 GB). Rust 1.92. Work paused 2026-10-12.

## Current state (all working and verified)

| Area | Status |
|---|---|
| Move generation | Legal-only bitboard movegen with check/pin masks. Magic sliders generated at **build time** (`build.rs`). Perft passes all 6 standard positions (~540 Mnps bulk). |
| Search | PVS + iterative deepening + aspiration windows, lock-free shared TT (4-way buckets), Lazy SMP. Pruning: RFP, razoring, NMP, LMP, futility, SEE pruning, history pruning, IIR. Extensions: in-check, singular (incl. double ext + multicut + negative ext). LMR with history/killer/check/cutnode adjustments. Quiescence with SEE + delta pruning. |
| Ordering / history | Staged move picker (TT → good captures → quiets → bad captures); main, continuation (1 & 2 ply), capture histories; killer; countermove; pawn + non-pawn correction history. |
| Eval | `hce.rs` = PeSTO tapered PSTs + bishop pair + drawish scaling. `nnue.rs` = NNUE inference, used automatically when a net is embedded at build time. |
| UCI | `uci, isready, ucinewgame, position, go (wtime/btime/winc/binc/movestogo/movetime/depth/nodes/infinite), stop, quit, setoption (Hash, Threads, Move Overhead, Clear Hash)`, plus `d`, `eval`, `perft N`, `bench [depth]`. |
| Tests | `cargo test --release`: perft suite, staged-gen partition, incremental hash, SEE. |
| Sanity | 100-game self-play at 2+0.02: no crashes, illegal moves, or time losses. Finds KQK/KRK mates. |

**No NNUE has been trained on real data yet.** The released binary currently uses HCE.
Strength has **not** been measured against any external engine yet.

## File map

- `build.rs`: finds magics, writes `$OUT_DIR/magics.rs` + `slider_attacks.bin`; embeds `nets/default.nnue` (or `$FERRUM_NET`) as `$OUT_DIR/net.bin`. An empty or missing net means HCE fallback.
- `src/types.rs`, `attacks.rs`, `zobrist.rs`, `moves.rs` (16-bit move: from|to|4-bit flag), `board.rs` (copy-make `Board`, FEN, SEE), `movegen.rs` (+ perft, tests).
- `src/search.rs`: `Worker` (per thread), `Shared` (TT, stop flag, node counters). `src/movepick.rs`, `history.rs`, `tt.rs`, `timeman.rs`.
- `src/eval.rs`: `Evaluator` front-end. Lazy accumulator stack: `push_move` / `push_null` record deltas, `evaluate` catches up. Has a `debug_assert` comparing against a full refresh. 50-move fade applied.
- `src/nnue.rs`: arch constants, `Network` layout, NEON SCReLU kernel (scalar fallback for non-aarch64).
- `src/data.rs`: 32-byte `PackedPos` training record.
- `src/datagen.rs`: self-play data generator. `src/trainer.rs`: multi-threaded CPU NNUE trainer.
- `src/bench.rs`: 35 bench FENs. `src/uci.rs`: engine front-end (search runs on its own thread; helpers via `thread::scope`, 64 MB stacks).
- `tools/` (gitignored): `fastchess/fastchess` (built from source), `books/UHO_Lichess_4852_v1.epd` + `8moves_v3.pgn`, `sprt.sh`.
- `bins/` (gitignored): `ferrum-v0` (first HCE build), `ferrum-datagen0` (binary used for gen0 data).
- `data/` (gitignored): `gen0.bin` (808,960 HCE self-play positions, record-aligned), `gen0.log`, `small.bin` (disposable test snapshot).
- `nets/test/` = toy net trained on 347K positions for 4 epochs, only to validate the pipeline. **Do not use it as a real net.**

## Commands

```bash
cargo build --release                         # HCE (no nets/default.nnue present)
FERRUM_NET=path/to.nnue cargo build --release  # embed a specific net
cargo test --release
./target/release/ferrum bench 12              # deterministic node count + nps

# Self-play data (appends to file; target counts only this run's positions)
./target/release/ferrum datagen <threads> <positions> <out.bin> [soft_nodes=5000]

# Train: writes <out>/epochN.nnue and final.nnue, then prints float-vs-quantized checks
./target/release/ferrum train <data.bin> <out-dir> [epochs=30] [lr=0.001] [wdl=0.25] [threads=8]

# SPRT: new vs old binary (default 8+0.08, elo0=0 elo1=5, concurrency 7)
tools/sprt.sh bins/new bins/old [tc] [elo0] [elo1] [concurrency]
```

## Formats

- **Training record (32 bytes, LE):** `occ u64` | 16 bytes of 4-bit pieces in ascending-square order (`color<<3 | type`) | `score i16` (cp, **White POV**) | `result u8` (0 black win, 1 draw, 2 white win) | `stm u8` | 4 bytes padding.
- **Datagen policy:** 8–9 random opening plies; reject if a 20K-node search gives |score| > 1000. Then 5000 soft nodes per move (hard cap 100K). Positions are recorded only when the side to move is not in check, the best move is quiet, and the score is not a mate score. Games are adjudicated as a win after |score| ≥ 2000 for 6 plies or any mate score, and as a draw after ply ≥ 80 with |score| ≤ 10 for 12 plies, or by 3-fold, 50-move, insufficient material, or 600 plies.
- **NNUE:** 768 inputs → 256 hidden (shared weights, two perspectives) → SCReLU → 1 output, with 8 output buckets by piece count, `(n-2)/4`. Feature = `(color^persp)*384 + type*64 + (persp==WHITE ? sq : sq^56)`. Quantization QA=255, QB=64, SCALE=400. W1 and W2 are clipped to ±1.98 during training; the NEON kernel relies on |W2_q| ≤ 127. File layout: `w1[768][H] i16, b1[H] i16, w2[8][2H] i16, b2[32] i32` (only 8 used), 402,048 bytes for H=256.
- **Trainer:** AdamW (β 0.9/0.999, wd 0.01), batch 16384, cosine LR 1e-3 → 1e-5. Target = `(1-wdl)*sigmoid(score/400) + wdl*result`. 1% (≤200K) held out for validation. Throughput is about 230K samples/s per thread.

## Measured numbers

- Bench (depth 13, HCE, idle machine): 3.65M nodes, ~6.1M nps single thread.
- NNUE (H=256) costs about 25% nps versus HCE.
- Datagen throughput: ~1,900 positions/s with 7 threads on HCE. NNUE will be slower.

## Next steps (planned order)

1. **Generate gen0 data.** Resume with `bins/ferrum-datagen0 datagen 7 <N> data/gen0.bin 5000` (appends). Plan was ~12–15M positions total, about 2 hours, then stop. Already have 0.81M.
2. **Train net0:** `ferrum train data/gen0.bin nets/net0 30`. Copy `final.nnue` to `nets/default.nnue` and rebuild. SPRT it against `bins/ferrum-v0` (HCE); expect a large gain (+200 Elo or more).
3. **gen1 data:** self-play with the net0 engine (30M+ positions). Train net1 on gen1 (optionally + gen0). Consider H=512, horizontal mirroring, and later king input buckets; SPRT each change.
4. **Search improvements** (SPRT each, after NNUE is in, since tuning depends on eval): ProbCut; continuation history at ply 4; deeper/shallower re-search after LMR; using eval differences to update the previous move's quiet history; check-extension variants; pruning margins and LMR constants (or SPSA); quiet check-evasion pruning in qsearch.
5. **Calibrate strength** against Stockfish with `UCI_LimitStrength` / `UCI_Elo` (would need building Stockfish locally) to give the user an Elo estimate. Then write the README.

## Caveats / known gaps

- Repetition is treated as a draw on its first recurrence inside search (standard). Draw score is plain 0, with no contempt or jitter.
- `see_ge` returns `0 >= threshold` for castling, en passant and promotions (Stockfish-style simplification).
- TT stores eval-only entries (`BOUND_NONE`) at nodes without a TT hit, so `hashfull` fills quickly; this is intended.
- No pondering, MultiPV, Chess960, Syzygy, or `EvalFile` option yet.
- The git repo was created by `cargo init`. **Nothing has been committed**; the user has not asked for commits.
