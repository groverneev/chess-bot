//! Evaluation front-end used by the search: NNUE with lazily updated accumulators when a
//! network is available, otherwise the hand-crafted evaluation.

use crate::board::Board;
use crate::hce;
use crate::moves::Move;
use crate::nnue::{self, Accumulator, Delta, Network};
use crate::search::{MATE_BOUND, MAX_PLY};

pub struct Evaluator {
    net: Option<&'static Network>,
    acc: Vec<Accumulator>,
    computed: Vec<bool>,
    deltas: Vec<Delta>,
}

impl Evaluator {
    pub fn new() -> Evaluator {
        Evaluator::with_network(nnue::network())
    }

    pub fn with_network(net: Option<&'static Network>) -> Evaluator {
        let n = if net.is_some() { MAX_PLY + 2 } else { 0 };
        Evaluator {
            net,
            acc: vec![Accumulator { v: [[0; nnue::HIDDEN]; 2] }; n],
            computed: vec![false; n],
            deltas: vec![Delta::default(); n],
        }
    }

    /// Called once at the root of every search.
    pub fn reset(&mut self, root: &Board) {
        if let Some(net) = self.net {
            self.acc[0] = Accumulator::refresh(net, root);
            self.computed[0] = true;
        }
    }

    /// Called after `m` is played from `parent` at `ply` (the child is at `ply + 1`).
    #[inline(always)]
    pub fn push_move(&mut self, parent: &Board, m: Move, ply: usize) {
        if self.net.is_some() {
            self.deltas[ply + 1] = Delta::from_move(parent, m);
            self.computed[ply + 1] = false;
        }
    }

    /// Called after a null move at `ply`.
    #[inline(always)]
    pub fn push_null(&mut self, ply: usize) {
        if self.net.is_some() {
            self.deltas[ply + 1] = Delta::default();
            self.computed[ply + 1] = false;
        }
    }

    /// Static evaluation of `b` (the position at `ply`) from the side to move's view.
    pub fn evaluate(&mut self, b: &Board, ply: usize) -> i32 {
        let Some(net) = self.net else {
            return hce::evaluate(b);
        };
        if !self.computed[ply] {
            let mut k = ply;
            while !self.computed[k] {
                k -= 1;
            }
            for j in k + 1..=ply {
                let (head, tail) = self.acc.split_at_mut(j);
                self.deltas[j].apply(net, &head[j - 1], &mut tail[0]);
                self.computed[j] = true;
            }
        }
        debug_assert!(
            Accumulator::refresh(net, b).v == self.acc[ply].v,
            "incremental accumulator diverged at ply {ply}: {}",
            b.to_fen()
        );
        let raw = nnue::evaluate(net, &self.acc[ply], b.stm, b.occupied().count_ones());
        // Fade toward a draw as the fifty-move counter grows.
        let scaled = raw * (200 - (b.halfmove as i32).min(199)) / 200;
        scaled.clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }
}
