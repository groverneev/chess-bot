use crate::bench;
use crate::board::*;
use crate::movegen::*;
use crate::search::*;
use crate::timeman::*;
use std::io::BufRead;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::thread::JoinHandle;

pub const NAME: &str = "Ferrum";
const STACK_SIZE: usize = 64 * 1024 * 1024;

pub struct Engine {
    board: Board,
    hashes: Vec<u64>,
    shared: Arc<Shared>,
    workers: Vec<Worker>,
    handle: Option<JoinHandle<Vec<Worker>>>,
    hash_mb: usize,
    threads: usize,
    move_overhead: u64,
}

impl Engine {
    pub fn new() -> Engine {
        let mut e = Engine {
            board: Board::startpos(),
            hashes: Vec::new(),
            shared: Arc::new(Shared::new(16, 1)),
            workers: Vec::new(),
            handle: None,
            hash_mb: 16,
            threads: 1,
            move_overhead: 20,
        };
        e.rebuild();
        e
    }

    /// Blocks until a running search has finished and reclaims its workers.
    fn wait(&mut self) {
        if let Some(h) = self.handle.take() {
            self.workers = h.join().expect("search thread panicked");
        }
    }

    fn rebuild(&mut self) {
        self.wait();
        self.workers.clear();
        self.shared = Arc::new(Shared::new(self.hash_mb, self.threads));
        self.workers = (0..self.threads).map(|i| Worker::new(i, self.shared.clone())).collect();
    }

    fn new_game(&mut self) {
        self.wait();
        self.shared.tt.clear();
        for w in &mut self.workers {
            w.clear();
        }
    }

    fn set_position(&mut self, tokens: &[&str]) {
        let (mut board, mut idx) = match tokens.get(1) {
            Some(&"startpos") => (Board::startpos(), 2),
            Some(&"fen") => {
                let end = tokens.iter().position(|&t| t == "moves").unwrap_or(tokens.len());
                match Board::from_fen(&tokens[2..end].join(" ")) {
                    Ok(b) => (b, end),
                    Err(e) => {
                        println!("info string invalid position: {e}");
                        return;
                    }
                }
            }
            _ => return,
        };
        let mut hashes = Vec::new();
        if tokens.get(idx) == Some(&"moves") {
            idx += 1;
            for s in &tokens[idx..] {
                match parse_uci_move(&board, s) {
                    Some(m) => {
                        hashes.push(board.hash);
                        board.make_move(m);
                    }
                    None => {
                        println!("info string illegal move {s}");
                        break;
                    }
                }
            }
        }
        self.board = board;
        self.hashes = hashes;
    }

    fn go(&mut self, tokens: &[&str]) {
        self.wait();
        let mut limits = Limits::default();
        let mut i = 1;
        while i < tokens.len() {
            let num = tokens.get(i + 1).and_then(|s| s.parse::<i64>().ok()).map(|v| v.max(0) as u64);
            match tokens[i] {
                "depth" => limits.depth = num.map(|v| v as i32),
                "nodes" => limits.nodes = num,
                "movetime" => limits.movetime = num,
                "wtime" => limits.wtime = num,
                "btime" => limits.btime = num,
                "winc" => limits.winc = num,
                "binc" => limits.binc = num,
                "movestogo" => limits.movestogo = num,
                "infinite" => limits.infinite = true,
                _ => {}
            }
            i += 1;
        }
        let tm = TimeManager::new(&limits, self.board.stm, self.move_overhead);
        self.shared.stop.store(false, Relaxed);
        self.shared.tt.new_search();
        for n in &self.shared.nodes {
            n.0.store(0, Relaxed);
        }

        let mut workers = std::mem::take(&mut self.workers);
        let board = self.board;
        let hashes = self.hashes.clone();
        let shared = self.shared.clone();
        let handle = std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(move || {
                std::thread::scope(|s| {
                    let (main, helpers) = workers.split_at_mut(1);
                    for w in helpers.iter_mut() {
                        let (hashes, limits, tm) = (&hashes, limits.clone(), tm.clone());
                        std::thread::Builder::new()
                            .stack_size(STACK_SIZE)
                            .spawn_scoped(s, move || {
                                w.silent = true;
                                w.search(&board, hashes, limits, tm);
                            })
                            .expect("failed to spawn helper thread");
                    }
                    main[0].silent = false;
                    main[0].search(&board, &hashes, limits, tm);
                    shared.stop.store(true, Relaxed);
                });
                println!("bestmove {}", workers[0].best_move.to_uci());
                workers
            })
            .expect("failed to spawn search thread");
        self.handle = Some(handle);
    }

    fn set_option(&mut self, tokens: &[&str]) {
        let name_at = tokens.iter().position(|&t| t == "name");
        let value_at = tokens.iter().position(|&t| t == "value");
        let Some(n) = name_at else { return };
        let name = tokens[n + 1..value_at.unwrap_or(tokens.len())].join(" ").to_lowercase();
        let value = value_at.map(|v| tokens[v + 1..].join(" ")).unwrap_or_default();
        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.hash_mb = mb.clamp(1, 65536);
                    self.rebuild();
                }
            }
            "threads" => {
                if let Ok(t) = value.parse::<usize>() {
                    self.threads = t.clamp(1, 512);
                    self.rebuild();
                }
            }
            "move overhead" => {
                if let Ok(v) = value.parse::<u64>() {
                    self.move_overhead = v.min(5000);
                }
            }
            "clear hash" => self.new_game(),
            _ => println!("info string unknown option '{name}'"),
        }
    }

    pub fn run(&mut self) {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let Some(&cmd) = tokens.first() else { continue };
            match cmd {
                "uci" => {
                    println!("id name {NAME} {}", env!("CARGO_PKG_VERSION"));
                    println!("id author the {NAME} authors");
                    println!("option name Hash type spin default 16 min 1 max 65536");
                    println!("option name Threads type spin default 1 min 1 max 512");
                    println!("option name Move Overhead type spin default 20 min 0 max 5000");
                    println!("option name Clear Hash type button");
                    println!("uciok");
                }
                "isready" => println!("readyok"),
                "ucinewgame" => self.new_game(),
                "position" => self.set_position(&tokens),
                "go" => self.go(&tokens),
                "stop" => self.shared.stop.store(true, Relaxed),
                "ponderhit" => {}
                "setoption" => self.set_option(&tokens),
                "quit" => {
                    self.shared.stop.store(true, Relaxed);
                    break;
                }
                "d" => self.board.print(),
                "eval" => {
                    let mut ev = crate::eval::Evaluator::new();
                    ev.reset(&self.board);
                    println!("eval: {} cp (side to move)", ev.evaluate(&self.board, 0));
                }
                "perft" => {
                    let depth = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
                    let t = std::time::Instant::now();
                    let n = perft(&self.board, depth);
                    println!("nodes {n} time {} ms", t.elapsed().as_millis());
                }
                "bench" => {
                    self.wait();
                    bench::run(tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(bench::DEFAULT_DEPTH));
                }
                _ => println!("info string unknown command '{cmd}'"),
            }
        }
        self.wait();
    }
}
