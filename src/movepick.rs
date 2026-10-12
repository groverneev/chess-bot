//! Staged move ordering: TT move, good captures, quiets (killer/counter first), bad captures.

use crate::board::*;
use crate::history::History;
use crate::movegen::*;
use crate::moves::*;
use crate::types::*;

const STAGE_TT: u8 = 0;
const STAGE_GEN_NOISY: u8 = 1;
const STAGE_GOOD_NOISY: u8 = 2;
const STAGE_GEN_QUIET: u8 = 3;
const STAGE_QUIET: u8 = 4;
const STAGE_BAD_NOISY: u8 = 5;
const STAGE_DONE: u8 = 6;

/// (piece, to) of the moves 1 and 2 plies back; piece 12 selects an all-zero table.
pub type ContIdx = [(usize, usize); 2];

pub struct MovePicker {
    noisy: MoveList,
    quiet: MoveList,
    scores: [i32; MAX_MOVES],
    idx: usize,
    bad_len: usize,
    bad_idx: usize,
    stage: u8,
    tt_move: Move,
    killer: Move,
    counter: Move,
    noisy_gen: bool,
    quiet_gen: bool,
    noisy_only: bool,
    cont: ContIdx,
    pub skip_quiets: bool,
}

#[inline(always)]
fn pick_best(moves: &mut MoveList, scores: &mut [i32; MAX_MOVES], start: usize) {
    let mut best = start;
    for i in start + 1..moves.len {
        if scores[i] > scores[best] {
            best = i;
        }
    }
    moves.moves.swap(start, best);
    scores.swap(start, best);
}

impl MovePicker {
    pub fn new(tt_move: Move, killer: Move, counter: Move, noisy_only: bool, cont: ContIdx) -> MovePicker {
        MovePicker {
            noisy: MoveList::new(),
            quiet: MoveList::new(),
            scores: [0; MAX_MOVES],
            idx: 0,
            bad_len: 0,
            bad_idx: 0,
            stage: STAGE_TT,
            tt_move,
            killer,
            counter,
            noisy_gen: false,
            quiet_gen: false,
            noisy_only,
            cont,
            skip_quiets: false,
        }
    }

    fn score_noisy(&mut self, b: &Board, h: &History) {
        for i in 0..self.noisy.len {
            let m = self.noisy.moves[i];
            let cap = b.captured_type(m);
            let pc = b.moved_piece(m) as usize;
            let mut s = 16 * SEE_VALUES[cap] + h.capture[pc][m.to()][cap] as i32 / 16;
            if m.is_promo() && m.promo_type() == QUEEN {
                s += 16 * SEE_VALUES[QUEEN];
            }
            self.scores[i] = s;
        }
    }

    fn score_quiet(&mut self, b: &Board, h: &History) {
        let stm = b.stm;
        let [(p1, t1), (p2, t2)] = self.cont;
        let c1 = &h.cont[p1][t1];
        let c2 = &h.cont[p2][t2];
        for i in 0..self.quiet.len {
            let m = self.quiet.moves[i];
            let pc = b.moved_piece(m) as usize;
            let to = m.to();
            let mut s = h.main[stm][m.from()][to] as i32 + c1[pc][to] as i32 + c2[pc][to] as i32;
            if m == self.killer {
                s += 1 << 20;
            } else if m == self.counter {
                s += 1 << 19;
            }
            self.scores[i] = s;
        }
    }

    pub fn next(&mut self, b: &Board, h: &History) -> Option<Move> {
        loop {
            match self.stage {
                STAGE_TT => {
                    self.stage = STAGE_GEN_NOISY;
                    let m = self.tt_move;
                    if m.is_some() {
                        if m.is_noisy() {
                            generate_noisy(b, &mut self.noisy);
                            self.noisy_gen = true;
                            if self.noisy.contains(m) {
                                return Some(m);
                            }
                        } else if !self.noisy_only {
                            generate_quiet(b, &mut self.quiet);
                            self.quiet_gen = true;
                            if self.quiet.contains(m) {
                                return Some(m);
                            }
                        }
                    }
                }
                STAGE_GEN_NOISY => {
                    if !self.noisy_gen {
                        generate_noisy(b, &mut self.noisy);
                        self.noisy_gen = true;
                    }
                    self.score_noisy(b, h);
                    self.idx = 0;
                    self.stage = STAGE_GOOD_NOISY;
                }
                STAGE_GOOD_NOISY => {
                    while self.idx < self.noisy.len {
                        let i = self.idx;
                        pick_best(&mut self.noisy, &mut self.scores, i);
                        self.idx += 1;
                        let m = self.noisy.moves[i];
                        if m == self.tt_move {
                            continue;
                        }
                        if !self.noisy_only && !b.see_ge(m, -self.scores[i] / 64) {
                            // Reuse the already-consumed prefix of the list to hold bad captures.
                            self.noisy.moves[self.bad_len] = m;
                            self.bad_len += 1;
                            continue;
                        }
                        return Some(m);
                    }
                    self.stage = if self.noisy_only { STAGE_DONE } else { STAGE_GEN_QUIET };
                }
                STAGE_GEN_QUIET => {
                    if !self.skip_quiets {
                        if !self.quiet_gen {
                            generate_quiet(b, &mut self.quiet);
                            self.quiet_gen = true;
                        }
                        self.score_quiet(b, h);
                    }
                    self.idx = 0;
                    self.stage = STAGE_QUIET;
                }
                STAGE_QUIET => {
                    if !self.skip_quiets {
                        while self.idx < self.quiet.len {
                            let i = self.idx;
                            pick_best(&mut self.quiet, &mut self.scores, i);
                            self.idx += 1;
                            let m = self.quiet.moves[i];
                            if m == self.tt_move {
                                continue;
                            }
                            return Some(m);
                        }
                    }
                    self.stage = STAGE_BAD_NOISY;
                }
                STAGE_BAD_NOISY => {
                    while self.bad_idx < self.bad_len {
                        let m = self.noisy.moves[self.bad_idx];
                        self.bad_idx += 1;
                        if m != self.tt_move {
                            return Some(m);
                        }
                    }
                    self.stage = STAGE_DONE;
                }
                _ => return None,
            }
        }
    }
}
