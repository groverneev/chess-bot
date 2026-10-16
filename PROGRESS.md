# Ferrum: progress notes and handoff

Goal: build the strongest UCI chess engine we can, in Rust (std only, no crates).
Machines: Apple M3 (aarch64, 8 cores, 16 GB, Rust 1.92) through October 14; October 16 work ran in a 4-core x86_64 Xeon cloud container (AVX-512, Rust 1.97). Last updated 2026-10-16.

## Current state (all working and verified)

| Area | Status |
|---|---|
| Move generation | Legal-only bitboard movegen with check/pin masks. Magic sliders generated at **build time** (`build.rs`). Perft passes all 6 standard positions (~540 Mnps bulk). |
| Search | PVS + iterative deepening + aspiration windows, lock-free shared TT (4-way buckets), Lazy SMP. Pruning: RFP, razoring, NMP, LMP, futility, SEE pruning, history pruning, IIR. Extensions: in-check, singular (incl. double ext + multicut + negative ext). LMR with history/killer/check/cutnode adjustments. Quiescence with SEE + delta pruning. |
| Ordering / history | Staged move picker (TT → good captures → quiets → bad captures); main, continuation (1 & 2 ply), capture histories; killer; countermove; pawn + non-pawn correction history. |
| Eval | **Default: NNUE `nets/default.nnue` (net1m-e8, Oct 16), +157 Elo over HCE.** `hce.rs` = PeSTO tapered PSTs + bishop pair + drawish scaling, used when no net is embedded. |
| UCI | `uci, isready, ucinewgame, position, go (wtime/btime/winc/binc/movestogo/movetime/depth/nodes/infinite), stop, quit, setoption (Hash, Threads, Move Overhead, Clear Hash)`, plus `d`, `eval`, `perft N`, `bench [depth]`. |
| Tests | `cargo test --release`: 9 tests covering perft, staged-gen partition, incremental hash, SEE, trainer input validation, partial-batch gradients, small-dataset training, and mirror augmentation. |
| Sanity | 100-game self-play at 2+0.02: no crashes, illegal moves, or time losses. Finds KQK/KRK mates. Bench signature (depth 12, default net): **2,494,005 nodes**; HCE build (`FERRUM_NET=none cargo build --release`): 2,430,733. |

**NNUE is now the default** (`nets/default.nnue`, committed): SPRT vs HCE at 8+0.08 accepted H1 with **+156.6 ± 35.0 Elo**. See the October 16 section.
Strength has **not** been measured against any external engine yet.

## File map

- `build.rs`: finds magics, writes `$OUT_DIR/magics.rs` + `slider_attacks.bin`; embeds `nets/default.nnue` (or `$FERRUM_NET`) as `$OUT_DIR/net.bin`. An empty or missing net means HCE fallback.
- `src/types.rs`, `attacks.rs`, `zobrist.rs`, `moves.rs` (16-bit move: from|to|4-bit flag), `board.rs` (copy-make `Board`, FEN, SEE), `movegen.rs` (+ perft, tests).
- `src/search.rs`: `Worker` (per thread), `Shared` (TT, stop flag, node counters). `src/movepick.rs`, `history.rs`, `tt.rs`, `timeman.rs`.
- `src/eval.rs`: `Evaluator` front-end. Lazy accumulator stack: `push_move` / `push_null` record deltas, `evaluate` catches up. Has a `debug_assert` comparing against a full refresh. 50-move fade applied.
- `src/nnue.rs`: arch constants, `Network` layout, NEON and AVX2 SCReLU kernels (scalar fallback otherwise).
- `src/data.rs`: 32-byte `PackedPos` training record.
- `src/datagen.rs`: self-play data generator. `src/trainer.rs`: multi-threaded CPU NNUE trainer.
- `src/bench.rs`: 35 bench FENs. `src/uci.rs`: engine front-end (search runs on its own thread; helpers via `thread::scope`, 64 MB stacks).
- `scripts/sprt.sh` (fastchess SPRT wrapper) and `scripts/pvcheck.py` (bestmove/PV consistency check).
- `tools/` (gitignored): `fastchess/fastchess` (built from source: `git clone https://github.com/Disservin/fastchess && make -j`), `books/UHO_Lichess_4852_v1.epd` + `8moves_v3.pgn` (unzipped from `official-stockfish/books`).
- `bins/` (gitignored): `ferrum-v0` (first HCE build), `ferrum-datagen0` (binary used for gen0 data).
- `data/` (gitignored): `gen0.bin` (808,960 HCE self-play positions, record-aligned), `gen0.log`, `small.bin` (disposable test snapshot).
- `nets/test/` = toy net trained on 347K positions for 4 epochs, only to validate the pipeline. **Do not use it as a real net.**
- `nets/net0-pilot/` (gitignored): 30 epoch checkpoints and `final.nnue` trained on all 808,960 gen0 positions. **Not a replacement for HCE:** lost the pilot match.
- `bins/ferrum-hce-20261014`, `bins/ferrum-net0-pilot` (gitignored): baseline and embedded pilot binaries from the October 14 experiment.
- `data/net0-pilot-train.log`, `data/net0-pilot-vs-hce.log`, `data/net0-pilot-vs-hce.pgn`, `data/net0-pilot-match-config.json` (gitignored): experiment evidence.

## Commands

```bash
cargo build --release                         # embeds nets/default.nnue
FERRUM_NET=none cargo build --release         # HCE only (any unreadable path = no net)
FERRUM_NET=path/to.nnue cargo build --release  # embed a specific net
cargo test --release
./target/release/ferrum bench 12              # deterministic node count + nps

# Self-play data (appends to file; target counts only this run's positions)
./target/release/ferrum datagen <threads> <positions> <out.bin> [soft_nodes=5000]

# Train: writes <out>/epochN.nnue and final.nnue, then prints float-vs-quantized checks.
# Validation = last 1% of the file (whole games), taken before shuffling. `mirror` = file-flip augmentation.
./target/release/ferrum train <data.bin> <out-dir> [epochs=30] [lr=0.001] [wdl=0.25] [threads=8] [mirror]

# SPRT: new vs old binary (default 8+0.08, elo0=0 elo1=5, concurrency nproc-1)
scripts/sprt.sh bins/new bins/old [tc] [elo0] [elo1] [concurrency]
scripts/pvcheck.py target/release/ferrum      # bestmove must match the last reported PV
```

## Formats

- **Training record (32 bytes, LE):** `occ u64` | 16 bytes of 4-bit pieces in ascending-square order (`color<<3 | type`) | `score i16` (cp, **White POV**) | `result u8` (0 black win, 1 draw, 2 white win) | `stm u8` | 4 bytes padding.
- **Datagen policy:** 8–9 random opening plies; reject if a 20K-node search gives |score| > 1000. Then 5000 soft nodes per move (hard cap 100K). Positions are recorded only when the side to move is not in check, the best move is quiet, and the score is not a mate score. Games are adjudicated as a win after |score| ≥ 2000 for 6 plies or any mate score, and as a draw after ply ≥ 80 with |score| ≤ 10 for 12 plies, or by 3-fold, 50-move, insufficient material, or 600 plies.
- **NNUE:** 768 inputs → 256 hidden (shared weights, two perspectives) → SCReLU → 1 output, with 8 output buckets by piece count, `(n-2)/4`. Feature = `(color^persp)*384 + type*64 + (persp==WHITE ? sq : sq^56)`. Quantization QA=255, QB=64, SCALE=400. W1 and W2 are clipped to ±1.98 during training; the NEON kernel relies on |W2_q| ≤ 127. File layout: `w1[768][H] i16, b1[H] i16, w2[8][2H] i16, b2[32] i32` (only 8 used), 402,048 bytes for H=256.
- **Trainer:** AdamW (β 0.9/0.999, wd 0.01), batch 16384, cosine LR 1e-3 → 1e-5. Target = `(1-wdl)*sigmoid(score/400) + wdl*result`. The last 1% (≤200K) of the file is held out for validation (game-independent since Oct 16; before that it was a random split). Throughput is about 230K samples/s per thread on the M3.

## October 14 continuation

- Fixed the trainer's dropped final batch: every training position now contributes each epoch, including datasets smaller than one batch. Gradient reduction includes only active workers, avoiding stale gradients when the last batch uses fewer threads. Loss normalization uses the actual training-set size.
- Added configuration and record validation: reject zero epochs/batch/threads, nonfinite or invalid learning rates/WDL, truncated data, fewer than two positions, invalid piece encodings, and missing/duplicate kings. Small datasets now retain at least one validation position. CLI reports these errors with a nonzero exit status.
- All 8 release tests pass. Existing dead-code warnings remain.
- Pilot command: `bins/ferrum-hce-20261014 train data/gen0.bin nets/net0-pilot 30 0.001 0.25 7`. Deterministic shuffle seed `0xC0FFEE`; 800,871 training / 8,089 validation positions. Final training MSE 0.002649, validation MSE 0.003394, elapsed approximately 14 seconds. This random position split is not a game-independent holdout; validation loss alone does not establish playing strength.
- Float versus quantized checks on 8 held-out positions differed by up to approximately 26.5 cp. This is a sample check, not exhaustive quantization validation.
- Fixed-size comparison: **28 wins / 54 losses / 18 draws**, 37/100 points. Fastchess reported relative Elo **−92.46 ± 58.38** at `2+0.02`, one thread, 16 MB hash. This is a short-time-control pilot, not an SPRT or an absolute Elo rating. 50 paired UHO openings, seed `20261014`, concurrency 4; match took 2m31s. No recorded game ended by crash, illegal move, or time forfeit.
- Both builds produced `Bestmove does not match beginning of last PV` warnings (125 total). Investigate interrupted-iteration reporting in `src/search.rs` separately before claiming clean UCI diagnostics.
- Depth-12 bench: HCE 2,430,733 nodes / ~5.00M nps; pilot 2,390,715 nodes / ~4.25M nps. Different evaluation changes the search tree, so this is not a pure inference-speed comparison.
- **Decision: keep HCE as default.** Preserve the pilot only for reproducibility; expand gen0 toward the planned 12–15M positions, retrain, and test again before promotion. (Superseded October 16.)
- SHA-256 of gen0 data: `9c7395b3271906edba8870aa3b51bcb16350a7ddb8bc9b611eec24ac5bc94f49`.
- SHA-256 of pilot final network: `6406cfc5d51243bf4609d13464a5c49d3cd892a4cf4950da8371bcd85313a69c`.

## October 16 continuation

Work is on branch `ferrum-continue`. The October 14 `data/`, `bins/`, `tools/` and `nets/net0-pilot` did not carry over (fresh container), so gen0 below is a **new** dataset.

- **Fixed the bestmove/PV mismatch.** The root set `best_move` on aspiration fail-highs, but the re-search reset `pv_len[0]`; a stop before any move raised alpha again left `best_move` unreported. The root now keeps `root_pv`, updated together with `best_move`. `scripts/pvcheck.py`: 9/600 mismatches before, 0/600 after; 0 warnings in the 232-game SPRT below.
- **AVX2 SCReLU kernel** (`nnue.rs`): bit-identical to scalar (same bench nodes), about 16% higher bench nps on the Xeon.
- **Trainer validation was leaking.** The random 1% split shared games with training data, so validation loss tracked training loss. With a game-independent holdout (last 1% of the file), a 30-epoch run on 3M positions has its best validation loss at epoch 2–3 and then rises steadily (0.00663 → 0.00728) while training loss keeps falling (0.0061 → 0.0037). That overfitting likely contributed to the pilot's loss.
- **Mirror augmentation** (`train ... mirror`): flips each position across the d/e boundary with probability 1/2 per batch. Lowers held-out loss and delays overfitting.
- **gen0 data:** `ferrum datagen 4 3000000 data/gen0.bin 5000` with the HCE build → 3,000,361 positions from 30,940 games in 4,530 s (~660 pos/s on 4 Xeon cores). SHA-256 `3a26e4f0b76069d06daeb7e6f5fa3821390baa8510eb3b66e7371c899934d4bb`. 2,970,358 train / 30,003 validation positions.
- Held-out validation loss (final epoch; lower is better), all lr 1e-3 → 1e-5, wdl 0.25:

  | Run | Epochs | Mirror | Val loss |
  |---|---|---|---|
  | net1 | 30 | no | 0.007283 |
  | net1-e4 | 4 | no | 0.006495 |
  | net1m-e4 | 4 | yes | 0.006452 |
  | **net1m-e8** | 8 | yes | **0.006371** |
  | net1m-e12 | 12 | yes | 0.006397 |

- **SPRT net1m-e8 vs HCE**, 8+0.08, 1 thread, 16 MB, UHO book, concurrency 4, bounds [0, 10]: **H1 accepted** after 232 games. +127 −29 =76, **Elo +156.56 ± 35.04**, nElo 229.1 ± 44.7, pentanomial [0, 11, 25, 51, 29]. No crashes, time losses, illegal moves, or PV warnings. 25 minutes.
- **Promoted** `nets/net1m-e8/final.nnue` to `nets/default.nnue` (SHA-256 `4912a4fc4bfa370ab366e1418cf3e73f71c3c63f28c8d5164001b2b74307a948`). A plain `cargo build --release` produces a binary byte-identical to the tested candidate.
- Reproduce: `ferrum train data/gen0.bin nets/net1m-e8 8 0.001 0.25 4 mirror` (deterministic given the same data and thread count).

## Measured numbers

- Bench (depth 13, HCE, idle machine): 3.65M nodes, ~6.1M nps single thread.
- NNUE (H=256) costs about 25% nps versus HCE.
- Datagen throughput: ~1,900 positions/s with 7 threads on HCE. NNUE will be slower.

## Next steps (planned order)

1. ~~Generate gen0 data and train a net that beats HCE.~~ Done October 16 (3M positions, +157 Elo).
2. **gen1 data:** self-play with the default (NNUE) engine; as many positions as time allows (10M+ ideally; on the M3 at ~1,500 pos/s that is ~2 hours). Train net2 on gen1 (+ gen0 if kept), choosing epochs by held-out validation loss, then SPRT against the current default. Note that the trainer's 1-epoch-on-3M overfitting curve means more data will matter more than more epochs.
3. **Architecture:** H=512, king input buckets; SPRT each change. Mirror augmentation is already in.
4. **Search improvements** (SPRT each, after NNUE is in, since tuning depends on eval): ProbCut; continuation history at ply 4; deeper/shallower re-search after LMR; using eval differences to update the previous move's quiet history; check-extension variants; pruning margins and LMR constants (or SPSA); quiet check-evasion pruning in qsearch.
5. **Calibrate strength** against Stockfish with `UCI_LimitStrength` / `UCI_Elo` (would need building Stockfish locally) to give the user an Elo estimate. Then write the README.

## Caveats / known gaps

- Repetition is treated as a draw on its first recurrence inside search (standard). Draw score is plain 0, with no contempt or jitter.
- `see_ge` returns `0 >= threshold` for castling, en passant and promotions (Stockfish-style simplification).
- TT stores eval-only entries (`BOUND_NONE`) at nodes without a TT hit, so `hashfull` fills quickly; this is intended.
- No pondering, MultiPV, Chess960, Syzygy, or `EvalFile` option yet.
- October 16 commits are on branch `ferrum-continue` (not merged into `main`, no PR).
