//! Self-play data generation for NNUE training.
//!
//! Usage: ferrum datagen <threads> <positions> <out-file> [soft-nodes]

use crate::board::*;
use crate::data::*;
use crate::movegen::*;
use crate::moves::*;
use crate::search::*;
use crate::timeman::*;
use crate::types::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x2545_F491_4F6C_DD1D | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 32) * n as u64 >> 32) as usize
    }
}

struct Ctx {
    out: Mutex<BufWriter<std::fs::File>>,
    positions: AtomicU64,
    games: AtomicU64,
    target: u64,
    done: AtomicBool,
    soft_nodes: u64,
}

fn search(worker: &mut Worker, b: &Board, hashes: &[u64], soft_nodes: u64) -> (Move, i32) {
    worker.shared.stop.store(false, Relaxed);
    let limits = Limits { soft_nodes: Some(soft_nodes), nodes: Some(soft_nodes * 20), ..Default::default() };
    let tm = TimeManager::new(&limits, b.stm, 0);
    worker.search(b, hashes, limits, tm)
}

fn is_threefold(b: &Board, hashes: &[u64]) -> bool {
    let n = hashes.len();
    let back = (b.halfmove as usize).min(n);
    hashes[n - back..].iter().filter(|&&h| h == b.hash).count() >= 2
}

/// Plays one game and returns its recorded positions with results filled in.
fn play_game(worker: &mut Worker, rng: &mut Rng, soft_nodes: u64) -> Vec<PackedPos> {
    let (mut b, mut hashes) = 'opening: loop {
        let mut b = Board::startpos();
        let mut hashes = Vec::new();
        let plies = 8 + rng.below(2);
        for _ in 0..plies {
            let mut list = MoveList::new();
            generate_all(&b, &mut list);
            if list.len == 0 {
                continue 'opening;
            }
            hashes.push(b.hash);
            b.make_move(list.moves[rng.below(list.len)]);
        }
        let mut list = MoveList::new();
        generate_all(&b, &mut list);
        if list.len == 0 {
            continue;
        }
        worker.shared.tt.clear();
        worker.clear();
        let (_, score) = search(worker, &b, &hashes, 20_000);
        if score.abs() <= 1000 {
            break (b, hashes);
        }
    };

    let mut positions = Vec::with_capacity(256);
    let mut win_plies = 0;
    let mut draw_plies = 0;
    let mut game_plies = 0;
    // From White's point of view: 2 win, 1 draw, 0 loss.
    let result: u8 = loop {
        if b.halfmove >= 100 || b.is_insufficient_material() || is_threefold(&b, &hashes) || game_plies >= 600 {
            break 1;
        }
        let (m, score) = search(worker, &b, &hashes, soft_nodes);
        if m.is_none() {
            break if b.in_check() { if b.stm == WHITE { 0 } else { 2 } } else { 1 };
        }
        let white_score = if b.stm == WHITE { score } else { -score };
        let white_wins = if white_score > 0 { 2 } else { 0 };
        if score.abs() >= MATE_BOUND {
            break white_wins;
        }
        win_plies = if score.abs() >= 2000 { win_plies + 1 } else { 0 };
        if win_plies >= 6 {
            break white_wins;
        }
        draw_plies = if game_plies >= 80 && score.abs() <= 10 { draw_plies + 1 } else { 0 };
        if draw_plies >= 12 {
            break 1;
        }
        if !b.in_check() && !m.is_noisy() {
            positions.push(PackedPos::pack(&b, white_score as i16));
        }
        hashes.push(b.hash);
        b.make_move(m);
        game_plies += 1;
    };
    for p in &mut positions {
        p.result = result;
    }
    positions
}

pub fn run(threads: usize, target: u64, path: &str, soft_nodes: u64) {
    let file = OpenOptions::new().create(true).append(true).open(path).expect("cannot open output file");
    let ctx = Arc::new(Ctx {
        out: Mutex::new(BufWriter::new(file)),
        positions: AtomicU64::new(0),
        games: AtomicU64::new(0),
        target,
        done: AtomicBool::new(false),
        soft_nodes,
    });
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    println!("datagen: {threads} threads, target {target} positions, {soft_nodes} soft nodes -> {path}");

    let start = Instant::now();
    std::thread::scope(|s| {
        for t in 0..threads {
            let ctx = ctx.clone();
            std::thread::Builder::new()
                .stack_size(64 << 20)
                .spawn_scoped(s, move || {
                    let shared = Arc::new(Shared::new(16, 1));
                    let mut worker = Worker::new(0, shared);
                    worker.silent = true;
                    let mut rng = Rng::new(seed.wrapping_add(t as u64 * 0x9E37_79B9_7F4A_7C15));
                    while !ctx.done.load(Relaxed) {
                        let positions = play_game(&mut worker, &mut rng, ctx.soft_nodes);
                        let mut out = ctx.out.lock().unwrap();
                        for p in &positions {
                            out.write_all(&p.to_bytes()).unwrap();
                        }
                        drop(out);
                        ctx.games.fetch_add(1, Relaxed);
                        let total = ctx.positions.fetch_add(positions.len() as u64, Relaxed) + positions.len() as u64;
                        if total >= ctx.target {
                            ctx.done.store(true, Relaxed);
                        }
                    }
                })
                .unwrap();
        }
        let mut last_report = Instant::now();
        while !ctx.done.load(Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
            if last_report.elapsed() >= Duration::from_secs(30) {
                last_report = Instant::now();
                let pos = ctx.positions.load(Relaxed);
                let secs = start.elapsed().as_secs_f64();
                println!(
                    "{:>8.0}s  games {:>8}  positions {:>10}  ({:.0} pos/s)",
                    secs,
                    ctx.games.load(Relaxed),
                    pos,
                    pos as f64 / secs
                );
            }
        }
    });
    ctx.out.lock().unwrap().flush().unwrap();
    println!(
        "done: {} games, {} positions in {:.0}s",
        ctx.games.load(Relaxed),
        ctx.positions.load(Relaxed),
        start.elapsed().as_secs_f64()
    );
}
