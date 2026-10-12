use crate::types::*;

pub const QUIET: u16 = 0;
pub const DOUBLE_PUSH: u16 = 1;
pub const KING_CASTLE: u16 = 2;
pub const QUEEN_CASTLE: u16 = 3;
pub const CAPTURE: u16 = 4;
pub const EP_CAPTURE: u16 = 5;
/// Promotion flags are `PROMO | (piece_type - KNIGHT)`, optionally with the capture bit.
pub const PROMO: u16 = 8;
pub const PROMO_CAPTURE: u16 = 12;

/// A move packed into 16 bits: from (6) | to (6) | flags (4).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Hash)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);
    /// Marker stored in the search stack for null moves; never generated.
    pub const NULL: Move = Move(65);

    #[inline(always)]
    pub const fn new(from: usize, to: usize, flag: u16) -> Move {
        Move(from as u16 | (to as u16) << 6 | flag << 12)
    }

    #[inline(always)]
    pub const fn from(self) -> usize {
        (self.0 & 63) as usize
    }

    #[inline(always)]
    pub const fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }

    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn is_some(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.flag() & CAPTURE != 0
    }

    #[inline(always)]
    pub const fn is_promo(self) -> bool {
        self.flag() & PROMO != 0
    }

    #[inline(always)]
    pub const fn promo_type(self) -> usize {
        (self.flag() & 3) as usize + KNIGHT
    }

    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        self.flag() == KING_CASTLE || self.flag() == QUEEN_CASTLE
    }

    #[inline(always)]
    pub const fn is_ep(self) -> bool {
        self.flag() == EP_CAPTURE
    }

    /// Noisy moves are captures (including all capture-promotions) and quiet queen promotions.
    /// This must match the split used by the move generator.
    #[inline(always)]
    pub const fn is_noisy(self) -> bool {
        self.is_capture() || self.flag() == PROMO | 3
    }

    pub fn to_uci(self) -> String {
        if self.is_none() {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", sq_to_str(self.from()), sq_to_str(self.to()));
        if self.is_promo() {
            s.push(b"nbrq"[self.promo_type() - KNIGHT] as char);
        }
        s
    }
}

pub const MAX_MOVES: usize = 256;

pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> MoveList {
        MoveList { moves: [Move::NONE; MAX_MOVES], len: 0 }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        // SAFETY: a legal position has at most 218 moves, well under MAX_MOVES.
        unsafe { *self.moves.get_unchecked_mut(self.len) = m };
        self.len += 1;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    pub fn contains(&self, m: Move) -> bool {
        self.as_slice().contains(&m)
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}
