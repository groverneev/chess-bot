//! Principal variation search with iterative deepening, aspiration windows and Lazy SMP.

use crate::board::*;
use crate::eval::Evaluator;
use crate::history::*;
use crate::movegen::*;
use crate::movepick::*;
use crate::moves::*;
use crate::timeman::{Limits, TimeManager};
use crate::tt::*;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

pub const MAX_PLY: usize = 128;
pub const INF: i32 = 32001;
pub const MATE: i32 = 32000;
pub const MATE_BOUND: i32 = MATE - 2 * MAX_PLY as i32;
pub const EVAL_NONE: i32 = 32500;

const SS_OFFSET: usize = 4;

#[repr(align(128))]
#[derive(Default)]
pub struct PaddedCounter(pub AtomicU64);

pub struct Shared {
    pub tt: TT,
    pub stop: AtomicBool,
    pub nodes: Vec<PaddedCounter>,
}

impl Shared {
    pub fn new(hash_mb: usize, threads: usize) -> Shared {
        Shared {
            tt: TT::new(hash_mb),
            stop: AtomicBool::new(false),
            nodes: (0..threads).map(|_| PaddedCounter::default()).collect(),
        }
    }

    pub fn total_nodes(&self) -> u64 {
        self.nodes.iter().map(|n| n.0.load(Relaxed)).sum()
    }
}

#[derive(Clone, Copy)]
struct StackEntry {
    static_eval: i32,
    raw_eval: i32,
    mv: Move,
    piece: u8,
    excluded: Move,
    killer: Move,
    double_exts: i32,
}

impl Default for StackEntry {
    fn default() -> Self {
        StackEntry {
            static_eval: EVAL_NONE,
            raw_eval: EVAL_NONE,
            mv: Move::NONE,
            piece: NO_PIECE,
            excluded: Move::NONE,
            killer: Move::NONE,
            double_exts: 0,
        }
    }
}

#[inline(always)]
fn score_to_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_BOUND {
        s + ply as i32
    } else if s <= -MATE_BOUND {
        s - ply as i32
    } else {
        s
    }
}

#[inline(always)]
fn score_from_tt(s: i32, ply: usize) -> i32 {
    if s == EVAL_NONE {
        s
    } else if s >= MATE_BOUND {
        s - ply as i32
    } else if s <= -MATE_BOUND {
        s + ply as i32
    } else {
        s
    }
}

#[inline(always)]
fn stat_bonus(depth: i32) -> i32 {
    (280 * depth - 250).clamp(0, 2400)
}

pub fn format_score(score: i32) -> String {
    if score >= MATE_BOUND {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_BOUND {
        format!("mate -{}", (MATE + score) / 2)
    } else {
        format!("cp {score}")
    }
}

pub struct Worker {
    pub id: usize,
    pub shared: Arc<Shared>,
    pub hist: Box<History>,
    pub eval: Evaluator,
    ss: Vec<StackEntry>,
    pv: Box<[[Move; MAX_PLY + 1]; MAX_PLY + 1]>,
    pv_len: [usize; MAX_PLY + 1],
    pub nodes: u64,
    seldepth: usize,
    hashes: Vec<u64>,
    root_depth: i32,
    lmr: Box<[[[i32; 64]; 64]; 2]>,
    tm: TimeManager,
    limits: Limits,
    root_nodes: Box<[[u64; 64]; 64]>,
    pub best_move: Move,
    pub best_score: i32,
    root_score: i32,
    pub completed_depth: i32,
    pub silent: bool,
}

impl Worker {
    pub fn new(id: usize, shared: Arc<Shared>) -> Worker {
        let mut lmr = Box::new([[[0i32; 64]; 64]; 2]);
        for d in 1..64 {
            for m in 1..64 {
                let l = (d as f64).ln() * (m as f64).ln();
                lmr[0][d][m] = (0.20 + l / 3.35) as i32;
                lmr[1][d][m] = (0.80 + l / 2.20) as i32;
            }
        }
        Worker {
            id,
            shared,
            hist: History::new(),
            eval: Evaluator::new(),
            ss: vec![StackEntry::default(); MAX_PLY + SS_OFFSET + 4],
            pv: Box::new([[Move::NONE; MAX_PLY + 1]; MAX_PLY + 1]),
            pv_len: [0; MAX_PLY + 1],
            nodes: 0,
            seldepth: 0,
            hashes: Vec::with_capacity(1024),
            root_depth: 0,
            lmr,
            tm: TimeManager::new(&Limits::default(), WHITE, 0),
            limits: Limits::default(),
            root_nodes: Box::new([[0; 64]; 64]),
            best_move: Move::NONE,
            best_score: 0,
            root_score: 0,
            completed_depth: 0,
            silent: false,
        }
    }

    pub fn clear(&mut self) {
        self.hist.clear();
    }

    #[inline(always)]
    fn ss(&self, ply: usize) -> &StackEntry {
        &self.ss[ply + SS_OFFSET]
    }

    #[inline(always)]
    fn ss_mut(&mut self, ply: usize) -> &mut StackEntry {
        &mut self.ss[ply + SS_OFFSET]
    }

    /// Entry `back` plies before `ply` (may land in the zeroed padding before the root).
    #[inline(always)]
    fn ss_back(&self, ply: usize, back: usize) -> &StackEntry {
        &self.ss[ply + SS_OFFSET - back]
    }

    #[inline(always)]
    fn stopped(&self) -> bool {
        self.shared.stop.load(Relaxed)
    }

    fn check_limits(&mut self) {
        self.shared.nodes[self.id].0.store(self.nodes, Relaxed);
        if self.id != 0 {
            return;
        }
        if self.tm.hard_exceeded() {
            self.shared.stop.store(true, Relaxed);
        }
        if let Some(n) = self.limits.nodes {
            if self.shared.total_nodes() >= n {
                self.shared.stop.store(true, Relaxed);
            }
        }
    }

    fn is_draw(&self, b: &Board) -> bool {
        if b.halfmove >= 100 || b.is_insufficient_material() {
            return true;
        }
        let n = self.hashes.len();
        let max_back = (b.halfmove as usize).min(n);
        let mut i = 4;
        while i <= max_back {
            if self.hashes[n - i] == b.hash {
                return true;
            }
            i += 2;
        }
        false
    }

    fn corrected_eval(&self, b: &Board, raw: i32) -> i32 {
        let stm = b.stm;
        let idx = |h: u64| (h as usize) & (CORR_SIZE - 1);
        let p = self.hist.pawn_corr[stm][idx(b.pawn_hash)] as i32;
        let w = self.hist.nonpawn_corr[stm][WHITE][idx(b.nonpawn_hash[WHITE])] as i32;
        let k = self.hist.nonpawn_corr[stm][BLACK][idx(b.nonpawn_hash[BLACK])] as i32;
        let corr = (p * 2 + w + k) / (2 * CORR_GRAIN);
        (raw + corr).clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    fn update_correction(&mut self, b: &Board, depth: i32, diff: i32) {
        let stm = b.stm;
        let idx = |h: u64| (h as usize) & (CORR_SIZE - 1);
        corr_update(&mut self.hist.pawn_corr[stm][idx(b.pawn_hash)], diff, depth);
        corr_update(&mut self.hist.nonpawn_corr[stm][WHITE][idx(b.nonpawn_hash[WHITE])], diff, depth);
        corr_update(&mut self.hist.nonpawn_corr[stm][BLACK][idx(b.nonpawn_hash[BLACK])], diff, depth);
    }

    fn cont_idx(&self, ply: usize) -> ContIdx {
        let p1 = self.ss_back(ply, 1);
        let p2 = self.ss_back(ply, 2);
        [(p1.piece as usize, p1.mv.to()), (p2.piece as usize, p2.mv.to())]
    }

    fn update_quiet(&mut self, b: &Board, m: Move, cont: &ContIdx, bonus: i32) {
        let pc = b.moved_piece(m) as usize;
        gravity(&mut self.hist.main[b.stm][m.from()][m.to()], bonus);
        for &(p, t) in cont {
            if p != NO_PIECE as usize {
                gravity(&mut self.hist.cont[p][t][pc][m.to()], bonus);
            }
        }
    }

    fn update_capture(&mut self, b: &Board, m: Move, bonus: i32) {
        let pc = b.moved_piece(m) as usize;
        let cap = b.captured_type(m);
        gravity(&mut self.hist.capture[pc][m.to()][cap], bonus);
    }

    fn quiet_score(&self, b: &Board, m: Move, cont: &ContIdx) -> i32 {
        let pc = b.moved_piece(m) as usize;
        let to = m.to();
        self.hist.main[b.stm][m.from()][to] as i32
            + self.hist.cont[cont[0].0][cont[0].1][pc][to] as i32
            + self.hist.cont[cont[1].0][cont[1].1][pc][to] as i32
    }

    fn update_pv(&mut self, ply: usize, m: Move) {
        let child_len = self.pv_len[ply + 1];
        let (head, tail) = self.pv.split_at_mut(ply + 1);
        head[ply][0] = m;
        head[ply][1..=child_len].copy_from_slice(&tail[0][..child_len]);
        self.pv_len[ply] = child_len + 1;
    }

    /// Runs iterative deepening on `root`. Returns the best move and its score.
    pub fn search(&mut self, root: &Board, game_hashes: &[u64], limits: Limits, tm: TimeManager) -> (Move, i32) {
        self.limits = limits;
        self.tm = tm;
        self.nodes = 0;
        self.shared.nodes[self.id].0.store(0, Relaxed);
        self.hashes.clear();
        self.hashes.extend_from_slice(game_hashes);
        self.best_move = Move::NONE;
        self.best_score = 0;
        self.completed_depth = 0;
        for e in self.ss.iter_mut() {
            *e = StackEntry::default();
        }
        for row in self.root_nodes.iter_mut() {
            row.fill(0);
        }
        self.eval.reset(root);

        let mut legal = MoveList::new();
        generate_all(root, &mut legal);
        if legal.len == 0 {
            return (Move::NONE, if root.in_check() { -MATE } else { 0 });
        }
        self.best_move = legal.moves[0];

        let max_depth = self.limits.depth.unwrap_or(MAX_PLY as i32 - 1).clamp(1, MAX_PLY as i32 - 1);
        let mut score = 0;
        let mut prev_best = Move::NONE;
        let mut stability = 0;

        for depth in 1..=max_depth {
            self.root_depth = depth;
            self.seldepth = 0;
            let mut delta = 15;
            let (mut alpha, mut beta) = (-INF, INF);
            if depth >= 4 {
                alpha = (score - delta).max(-INF);
                beta = (score + delta).min(INF);
            }
            loop {
                let s = self.negamax::<true>(root, alpha, beta, depth, 0, false);
                if self.stopped() {
                    break;
                }
                if s <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (s - delta).max(-INF);
                } else if s >= beta {
                    beta = (s + delta).min(INF);
                } else {
                    score = s;
                    break;
                }
                delta += delta / 2;
                if delta > 1000 {
                    alpha = -INF;
                    beta = INF;
                }
            }
            if self.stopped() {
                // A partially searched iteration still improves on the previous best move.
                if self.completed_depth > 0 && self.root_score > self.best_score {
                    self.best_score = self.root_score;
                }
                break;
            }
            self.completed_depth = depth;
            self.best_score = score;
            if self.pv_len[0] > 0 {
                self.best_move = self.pv[0][0];
            }
            self.shared.nodes[self.id].0.store(self.nodes, Relaxed);

            if self.id != 0 {
                continue;
            }
            if !self.silent {
                self.report(depth, score);
            }
            stability = if self.best_move == prev_best { stability + 1 } else { 0 };
            prev_best = self.best_move;
            let bm = self.best_move;
            let frac = self.root_nodes[bm.from()][bm.to()] as f64 / self.nodes.max(1) as f64;
            if depth >= 4 && self.tm.soft_exceeded(frac, stability) {
                break;
            }
            if let Some(sn) = self.limits.soft_nodes {
                if self.nodes >= sn {
                    break;
                }
            }
        }
        if self.id == 0 {
            // An interrupted iteration may have produced a new best move; report its line.
            if !self.silent && self.completed_depth > 0 && self.pv_len[0] > 0 && self.best_move != prev_best {
                self.report(self.completed_depth, self.root_score);
            }
            while self.limits.infinite && !self.stopped() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.shared.stop.store(true, Relaxed);
        }
        (self.best_move, self.best_score)
    }

    fn report(&self, depth: i32, score: i32) {
        let ms = self.tm.elapsed_ms().max(1);
        let nodes = self.shared.total_nodes().max(self.nodes);
        let pv: Vec<String> = self.pv[0][..self.pv_len[0]].iter().map(|m| m.to_uci()).collect();
        println!(
            "info depth {depth} seldepth {} score {} nodes {nodes} nps {} hashfull {} time {ms} pv {}",
            self.seldepth,
            format_score(score),
            nodes * 1000 / ms,
            self.shared.tt.hashfull(),
            pv.join(" ")
        );
    }

    fn negamax<const PV: bool>(
        &mut self,
        b: &Board,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
        cutnode: bool,
    ) -> i32 {
        let root = ply == 0;
        if PV {
            self.pv_len[ply] = 0;
        }
        let in_check = b.in_check();
        if in_check && !root && ply < 2 * self.root_depth as usize {
            depth += 1;
        }
        if depth <= 0 {
            return self.qsearch::<PV>(b, alpha, beta, ply);
        }

        self.nodes += 1;
        if self.nodes & 2047 == 0 {
            self.check_limits();
        }
        if PV {
            self.seldepth = self.seldepth.max(ply);
        }

        if !root {
            if self.stopped() {
                return 0;
            }
            if self.is_draw(b) {
                return 0;
            }
            if ply >= MAX_PLY - 1 {
                return if in_check { 0 } else { self.eval.evaluate(b, ply) };
            }
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let excluded = self.ss(ply).excluded;
        let tt_entry = if excluded.is_none() { self.shared.tt.probe(b.hash) } else { None };
        let mut tt_move = tt_entry.map_or(Move::NONE, |e| e.mv);
        if root && self.completed_depth > 0 {
            tt_move = self.best_move;
        }
        let tt_score = tt_entry.map_or(EVAL_NONE, |e| score_from_tt(e.score, ply));
        let tt_pv = PV || tt_entry.is_some_and(|e| e.pv);

        if !PV && excluded.is_none() {
            if let Some(e) = tt_entry {
                let fits = match e.bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => tt_score >= beta,
                    BOUND_UPPER => tt_score <= alpha,
                    _ => false,
                };
                if fits && e.depth >= depth && b.halfmove < 90 {
                    return tt_score;
                }
            }
        }

        let raw_eval;
        let static_eval;
        let mut eval;
        if in_check {
            raw_eval = EVAL_NONE;
            static_eval = EVAL_NONE;
            eval = EVAL_NONE;
        } else if excluded.is_some() {
            raw_eval = self.ss(ply).raw_eval;
            static_eval = self.ss(ply).static_eval;
            eval = static_eval;
        } else {
            raw_eval = match tt_entry {
                Some(e) if e.eval != EVAL_NONE => e.eval,
                _ => self.eval.evaluate(b, ply),
            };
            static_eval = self.corrected_eval(b, raw_eval);
            eval = static_eval;
            if let Some(e) = tt_entry {
                if tt_score != EVAL_NONE
                    && (e.bound == BOUND_EXACT
                        || (e.bound == BOUND_LOWER && tt_score > eval)
                        || (e.bound == BOUND_UPPER && tt_score < eval))
                {
                    eval = tt_score;
                }
            } else {
                self.shared.tt.store(b.hash, Move::NONE, EVAL_NONE, raw_eval, 0, BOUND_NONE, tt_pv);
            }
        }
        {
            let e = self.ss_mut(ply);
            e.static_eval = static_eval;
            e.raw_eval = raw_eval;
        }
        self.ss_mut(ply + 2).killer = Move::NONE;

        let improving = if in_check {
            false
        } else if self.ss_back(ply, 2).static_eval != EVAL_NONE {
            static_eval > self.ss_back(ply, 2).static_eval
        } else if self.ss_back(ply, 4).static_eval != EVAL_NONE {
            static_eval > self.ss_back(ply, 4).static_eval
        } else {
            true
        };

        if !PV && !in_check && excluded.is_none() {
            // Reverse futility pruning.
            if depth <= 8 && eval < MATE_BOUND && eval - 75 * (depth - improving as i32) >= beta {
                return (eval + beta) / 2;
            }

            // Razoring.
            if depth <= 3 && eval + 200 + 250 * depth <= alpha {
                let s = self.qsearch::<false>(b, alpha, alpha + 1, ply);
                if s <= alpha {
                    return s;
                }
            }

            // Null move pruning.
            if depth >= 3
                && eval >= beta
                && static_eval >= beta
                && self.ss_back(ply, 1).mv != Move::NULL
                && beta > -MATE_BOUND
                && b.has_non_pawn_material(b.stm)
            {
                let r = 3 + depth / 3 + ((eval - beta) / 200).min(3);
                let mut child = *b;
                child.make_null();
                {
                    let de = self.ss(ply).double_exts;
                    let e = self.ss_mut(ply);
                    e.mv = Move::NULL;
                    e.piece = NO_PIECE;
                    self.ss_mut(ply + 1).double_exts = de;
                }
                self.hashes.push(b.hash);
                self.eval.push_null(ply);
                let s = -self.negamax::<false>(&child, -beta, -beta + 1, depth - r, ply + 1, !cutnode);
                self.hashes.pop();
                if self.stopped() {
                    return 0;
                }
                if s >= beta {
                    return if s >= MATE_BOUND { beta } else { s };
                }
            }
        }

        // Internal iterative reduction.
        if depth >= 4 && tt_move.is_none() && excluded.is_none() && (PV || cutnode) {
            depth -= 1;
        }

        let cont = self.cont_idx(ply);
        let prev = self.ss_back(ply, 1);
        let counter = if prev.mv.is_some() && prev.mv != Move::NULL {
            self.hist.counter[prev.piece as usize][prev.mv.to()]
        } else {
            Move::NONE
        };
        let killer = self.ss(ply).killer;
        let mut picker = MovePicker::new(tt_move, killer, counter, false, cont);

        let us = b.stm;
        let orig_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move = Move::NONE;
        let mut legal = 0usize;
        let mut quiets = [Move::NONE; 64];
        let mut n_quiets = 0;
        let mut noisies = [Move::NONE; 32];
        let mut n_noisies = 0;

        while let Some(m) = picker.next(b, &self.hist) {
            if m == excluded {
                continue;
            }
            legal += 1;
            let is_quiet = !m.is_noisy();
            let pc = b.moved_piece(m);
            let hist_score = if is_quiet {
                self.quiet_score(b, m, &cont)
            } else {
                self.hist.capture[pc as usize][m.to()][b.captured_type(m)] as i32
            };
            let base_r = self.lmr[is_quiet as usize][(depth as usize).min(63)][legal.min(63)];

            if !root && best_score > -MATE_BOUND && b.has_non_pawn_material(us) {
                let lmr_depth = (depth - base_r).max(0);
                if is_quiet {
                    if legal as i32 >= (3 + depth * depth) / (2 - improving as i32) {
                        picker.skip_quiets = true;
                        continue;
                    }
                    if !in_check && lmr_depth <= 8 && static_eval + 120 + 100 * lmr_depth <= alpha {
                        picker.skip_quiets = true;
                        continue;
                    }
                    if lmr_depth <= 4 && hist_score < -2048 * depth {
                        continue;
                    }
                    if !b.see_ge(m, -30 * lmr_depth * lmr_depth) {
                        continue;
                    }
                } else if depth <= 10 && !b.see_ge(m, -100 * depth) {
                    continue;
                }
            }

            // Singular extensions.
            let mut ext = 0;
            if !root && m == tt_move && depth >= 7 && excluded.is_none() && ply < 2 * self.root_depth as usize {
                if let Some(e) = tt_entry {
                    if e.depth >= depth - 3 && e.bound & BOUND_LOWER != 0 && tt_score.abs() < MATE_BOUND {
                        let s_beta = tt_score - depth;
                        let s_depth = (depth - 1) / 2;
                        self.ss_mut(ply).excluded = m;
                        let s = self.negamax::<false>(b, s_beta - 1, s_beta, s_depth, ply, cutnode);
                        self.ss_mut(ply).excluded = Move::NONE;
                        if self.stopped() {
                            return 0;
                        }
                        if s < s_beta {
                            ext = 1;
                            if !PV && s < s_beta - 20 && self.ss(ply).double_exts < 8 {
                                ext = 2;
                            }
                        } else if s_beta >= beta {
                            return s_beta;
                        } else if tt_score >= beta || cutnode {
                            ext = -1;
                        }
                    }
                }
            }

            let mut child = *b;
            child.make_move(m);
            {
                let de = self.ss(ply).double_exts + (ext >= 2) as i32;
                let e = self.ss_mut(ply);
                e.mv = m;
                e.piece = pc;
                self.ss_mut(ply + 1).double_exts = de;
            }
            self.hashes.push(b.hash);
            self.eval.push_move(b, m, ply);
            let nodes_before = self.nodes;
            let new_depth = depth - 1 + ext;
            let gives_check = child.in_check();

            let mut score;
            if legal == 1 {
                score = -self.negamax::<PV>(&child, -beta, -alpha, new_depth, ply + 1, !PV && !cutnode);
            } else {
                if depth >= 2 && legal > 1 + root as usize {
                    let mut r = base_r;
                    r += !tt_pv as i32;
                    r += cutnode as i32;
                    r += !improving as i32;
                    r -= gives_check as i32;
                    if is_quiet {
                        r -= hist_score / 8192;
                        if m == killer || m == counter {
                            r -= 1;
                        }
                    } else {
                        r -= hist_score / 6144;
                    }
                    let reduced = (new_depth - r).max(1).min(new_depth);
                    score = -self.negamax::<false>(&child, -alpha - 1, -alpha, reduced, ply + 1, true);
                    if score > alpha && reduced < new_depth {
                        score = -self.negamax::<false>(&child, -alpha - 1, -alpha, new_depth, ply + 1, !cutnode);
                    }
                } else {
                    score = -self.negamax::<false>(&child, -alpha - 1, -alpha, new_depth, ply + 1, !cutnode);
                }
                if PV && score > alpha {
                    score = -self.negamax::<true>(&child, -beta, -alpha, new_depth, ply + 1, false);
                }
            }
            self.hashes.pop();

            if self.stopped() {
                return 0;
            }
            if root && self.id == 0 {
                self.root_nodes[m.from()][m.to()] += self.nodes - nodes_before;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = m;
                    if PV {
                        self.update_pv(ply, m);
                    }
                    if root {
                        self.root_score = score;
                        self.best_move = m;
                    }
                    if score >= beta {
                        break;
                    }
                    alpha = score;
                }
            }
            if m != best_move {
                if is_quiet {
                    if n_quiets < quiets.len() {
                        quiets[n_quiets] = m;
                        n_quiets += 1;
                    }
                } else if n_noisies < noisies.len() {
                    noisies[n_noisies] = m;
                    n_noisies += 1;
                }
            }
        }

        if legal == 0 {
            return if excluded.is_some() {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                0
            };
        }

        if best_score >= beta {
            let bonus = stat_bonus(depth);
            if !best_move.is_noisy() {
                self.update_quiet(b, best_move, &cont, bonus);
                for &q in &quiets[..n_quiets] {
                    self.update_quiet(b, q, &cont, -bonus);
                }
                self.ss_mut(ply).killer = best_move;
                let prev = *self.ss_back(ply, 1);
                if prev.mv.is_some() && prev.mv != Move::NULL {
                    self.hist.counter[prev.piece as usize][prev.mv.to()] = best_move;
                }
            } else {
                self.update_capture(b, best_move, bonus);
            }
            for &n in &noisies[..n_noisies] {
                self.update_capture(b, n, -bonus);
            }
        }

        if excluded.is_none() {
            let bound = if best_score >= beta {
                BOUND_LOWER
            } else if best_score > orig_alpha {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            let store_move = if best_move.is_some() { best_move } else { tt_move };
            self.shared.tt.store(b.hash, store_move, score_to_tt(best_score, ply), raw_eval, depth, bound, tt_pv);

            if !in_check
                && (best_move.is_none() || !best_move.is_noisy())
                && !(bound == BOUND_LOWER && best_score <= static_eval)
                && !(bound == BOUND_UPPER && best_score >= static_eval)
            {
                self.update_correction(b, depth, best_score - raw_eval);
            }
        }
        best_score
    }

    fn qsearch<const PV: bool>(&mut self, b: &Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if PV {
            self.pv_len[ply] = 0;
            self.seldepth = self.seldepth.max(ply);
        }
        self.nodes += 1;
        if self.nodes & 2047 == 0 {
            self.check_limits();
        }
        if self.stopped() {
            return 0;
        }
        if b.halfmove >= 100 || b.is_insufficient_material() {
            return 0;
        }
        let in_check = b.in_check();
        if ply >= MAX_PLY - 1 {
            return if in_check { 0 } else { self.eval.evaluate(b, ply) };
        }

        let tt_entry = self.shared.tt.probe(b.hash);
        let tt_score = tt_entry.map_or(EVAL_NONE, |e| score_from_tt(e.score, ply));
        let tt_move = tt_entry.map_or(Move::NONE, |e| e.mv);
        let tt_pv = PV || tt_entry.is_some_and(|e| e.pv);
        if !PV {
            if let Some(e) = tt_entry {
                let fits = match e.bound {
                    BOUND_EXACT => true,
                    BOUND_LOWER => tt_score >= beta,
                    BOUND_UPPER => tt_score <= alpha,
                    _ => false,
                };
                if fits {
                    return tt_score;
                }
            }
        }

        let raw_eval;
        let static_eval;
        let mut best_score;
        if in_check {
            raw_eval = EVAL_NONE;
            static_eval = EVAL_NONE;
            best_score = -INF;
        } else {
            raw_eval = match tt_entry {
                Some(e) if e.eval != EVAL_NONE => e.eval,
                _ => self.eval.evaluate(b, ply),
            };
            static_eval = self.corrected_eval(b, raw_eval);
            best_score = static_eval;
            if let Some(e) = tt_entry {
                if tt_score != EVAL_NONE
                    && (e.bound == BOUND_EXACT
                        || (e.bound == BOUND_LOWER && tt_score > best_score)
                        || (e.bound == BOUND_UPPER && tt_score < best_score))
                {
                    best_score = tt_score;
                }
            }
            if best_score >= beta {
                if tt_entry.is_none() {
                    self.shared.tt.store(b.hash, Move::NONE, score_to_tt(best_score, ply), raw_eval, 0, BOUND_LOWER, false);
                }
                return best_score;
            }
            alpha = alpha.max(best_score);
        }

        let futility = static_eval + 120;
        let no_cont = [(NO_PIECE as usize, 0); 2];
        let mut picker = MovePicker::new(tt_move, Move::NONE, Move::NONE, !in_check, no_cont);
        let mut best_move = Move::NONE;
        let mut legal = 0;

        while let Some(m) = picker.next(b, &self.hist) {
            legal += 1;
            if best_score > -MATE_BOUND {
                if !in_check && !m.is_promo() {
                    let gain = futility + SEE_VALUES[b.captured_type(m)];
                    if gain <= alpha {
                        best_score = best_score.max(gain);
                        continue;
                    }
                    if futility <= alpha && !b.see_ge(m, 1) {
                        best_score = best_score.max(futility);
                        continue;
                    }
                }
                if !b.see_ge(m, -50) {
                    continue;
                }
            }

            let mut child = *b;
            child.make_move(m);
            self.eval.push_move(b, m, ply);
            let score = -self.qsearch::<PV>(&child, -beta, -alpha, ply + 1);
            if self.stopped() {
                return 0;
            }
            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = m;
                    if PV {
                        self.update_pv(ply, m);
                    }
                    if score >= beta {
                        break;
                    }
                    alpha = score;
                }
            }
        }

        if in_check && legal == 0 {
            return -MATE + ply as i32;
        }

        let bound = if best_score >= beta { BOUND_LOWER } else { BOUND_UPPER };
        self.shared.tt.store(b.hash, best_move, score_to_tt(best_score, ply), raw_eval, 0, bound, tt_pv);
        best_score
    }
}
