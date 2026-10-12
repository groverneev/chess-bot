use crate::moves::Move;

pub const MAX_HISTORY: i32 = 16384;
pub const CORR_SIZE: usize = 16384;
pub const CORR_GRAIN: i32 = 256;
const CORR_MAX: i32 = CORR_GRAIN * 32;
const CORR_WEIGHT_SCALE: i32 = 256;

/// Continuation history for one (previous piece, previous destination): [piece][to].
pub type ContHist = [[i16; 64]; 12];

/// Per-thread move-ordering and evaluation-correction statistics.
pub struct History {
    /// [side][from][to]
    pub main: [[[i16; 64]; 64]; 2],
    /// [moved piece][to][captured type, KING meaning "none"]
    pub capture: [[[i16; 6]; 64]; 12],
    /// [prev piece (12 = none)][prev to] -> ContHist. The "none" row is never written.
    pub cont: [[ContHist; 64]; 13],
    /// [prev piece][prev to] -> refutation
    pub counter: [[Move; 64]; 13],
    /// [side][pawn hash]
    pub pawn_corr: [[i16; CORR_SIZE]; 2],
    /// [side][color][non-pawn hash of color]
    pub nonpawn_corr: [[[i16; CORR_SIZE]; 2]; 2],
}

impl History {
    pub fn new() -> Box<History> {
        // SAFETY: History is plain integers, for which all-zero is a valid value.
        unsafe { Box::<History>::new_zeroed().assume_init() }
    }

    pub fn clear(&mut self) {
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(self as *mut History, 0, 1) };
    }
}

#[inline(always)]
pub fn gravity(entry: &mut i16, bonus: i32) {
    let b = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
    let e = *entry as i32;
    *entry = (e + b - e * b.abs() / MAX_HISTORY) as i16;
}

#[inline(always)]
pub fn corr_update(entry: &mut i16, diff: i32, depth: i32) {
    let scaled = diff * CORR_GRAIN;
    let weight = (depth + 1).min(16);
    let e = *entry as i32;
    let v = (e * (CORR_WEIGHT_SCALE - weight) + scaled * weight) / CORR_WEIGHT_SCALE;
    *entry = v.clamp(-CORR_MAX, CORR_MAX) as i16;
}
