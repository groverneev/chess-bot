use crate::types::*;
use std::time::Instant;

#[derive(Clone, Default, Debug)]
pub struct Limits {
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    /// Stop after the iteration in which this many nodes were exceeded (used by datagen).
    pub soft_nodes: Option<u64>,
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u64>,
    pub infinite: bool,
}

#[derive(Clone)]
pub struct TimeManager {
    start: Instant,
    soft_ms: f64,
    hard_ms: f64,
    /// Fixed-time searches use the full budget instead of stopping early on stable moves.
    fixed: bool,
}

impl TimeManager {
    pub fn new(limits: &Limits, stm: usize, overhead: u64) -> TimeManager {
        let start = Instant::now();
        if let Some(mt) = limits.movetime {
            let t = mt.saturating_sub(overhead).max(1) as f64;
            return TimeManager { start, soft_ms: t, hard_ms: t, fixed: true };
        }
        let (time, inc) = if stm == WHITE { (limits.wtime, limits.winc) } else { (limits.btime, limits.binc) };
        let Some(time) = time else {
            return TimeManager { start, soft_ms: f64::INFINITY, hard_ms: f64::INFINITY, fixed: true };
        };
        let t = time.saturating_sub(overhead).max(1) as f64;
        let inc = inc.unwrap_or(0) as f64;
        let max_time = t * 0.75;
        let base = match limits.movestogo {
            Some(mtg) if mtg > 0 => t / (mtg.min(50) as f64 + 1.0) + inc * 0.8,
            _ => t * 0.05 + inc * 0.8,
        };
        TimeManager { start, soft_ms: (base * 0.76).min(max_time), hard_ms: (base * 3.0).min(max_time), fixed: false }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    pub fn hard_exceeded(&self) -> bool {
        self.start.elapsed().as_secs_f64() * 1000.0 >= self.hard_ms
    }

    /// Decide whether to start another iteration. `best_frac` is the share of root nodes spent
    /// on the current best move; `stability` counts iterations the best move has not changed.
    pub fn soft_exceeded(&self, best_frac: f64, stability: usize) -> bool {
        if self.fixed {
            return false;
        }
        const STABILITY: [f64; 5] = [2.0, 1.35, 1.1, 0.9, 0.8];
        let node_factor = (1.5 - best_frac) * 1.35;
        let limit = self.soft_ms * node_factor * STABILITY[stability.min(4)];
        self.start.elapsed().as_secs_f64() * 1000.0 >= limit.min(self.hard_ms)
    }
}
