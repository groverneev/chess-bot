//! NNUE inference: (768 -> HIDDEN) x 2 perspectives -> SCReLU -> 1, with material output buckets.

use crate::board::Board;
use crate::moves::*;
use crate::types::*;
use std::sync::OnceLock;

pub const INPUTS: usize = 768;
pub const HIDDEN: usize = 256;
pub const BUCKETS: usize = 8;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const SCALE: i32 = 400;

#[repr(C, align(64))]
pub struct Network {
    pub w1: [[i16; HIDDEN]; INPUTS],
    pub b1: [i16; HIDDEN],
    pub w2: [[i16; 2 * HIDDEN]; BUCKETS],
    pub b2: [i32; 32],
}

pub const NET_BYTES: usize = std::mem::size_of::<Network>();

static EMBEDDED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net.bin"));
static NET: OnceLock<Option<Box<Network>>> = OnceLock::new();

pub fn network_from_bytes(bytes: &[u8]) -> Option<Box<Network>> {
    if bytes.len() != NET_BYTES {
        return None;
    }
    // SAFETY: Network is plain integers; we fill every byte from little-endian data.
    let mut net = unsafe { Box::<Network>::new_zeroed().assume_init() };
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), &mut *net as *mut Network as *mut u8, NET_BYTES);
    }
    Some(net)
}

/// The embedded network, if the binary was built with one.
pub fn network() -> Option<&'static Network> {
    NET.get_or_init(|| network_from_bytes(EMBEDDED)).as_deref()
}

#[inline(always)]
pub fn feature(perspective: usize, pc: u8, sq: usize) -> usize {
    let (c, pt) = (piece_color(pc), piece_type(pc));
    let sq = if perspective == WHITE { sq } else { sq ^ 56 };
    (c ^ perspective) * 384 + pt * 64 + sq
}

#[inline(always)]
pub fn output_bucket(piece_count: u32) -> usize {
    ((piece_count.saturating_sub(2)) / 4).min(BUCKETS as u32 - 1) as usize
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Accumulator {
    pub v: [[i16; HIDDEN]; 2],
}

impl Accumulator {
    pub fn refresh(net: &Network, b: &Board) -> Accumulator {
        let mut acc = Accumulator { v: [net.b1; 2] };
        for sq in BitIter(b.occupied()) {
            let pc = b.mailbox[sq];
            for p in [WHITE, BLACK] {
                let w = &net.w1[feature(p, pc, sq)];
                for (a, &x) in acc.v[p].iter_mut().zip(w.iter()) {
                    *a += x;
                }
            }
        }
        acc
    }
}

/// Feature changes from one ply to the next: (piece, square) pairs.
#[derive(Clone, Copy, Default)]
pub struct Delta {
    add: [(u8, u8); 2],
    sub: [(u8, u8); 2],
    n_add: u8,
    n_sub: u8,
}

impl Delta {
    pub fn from_move(b: &Board, m: Move) -> Delta {
        let mut d = Delta::default();
        let (from, to) = (m.from(), m.to());
        let pc = b.mailbox[from];
        let us = b.stm;
        d.sub[0] = (pc, from as u8);
        d.n_sub = 1;
        let placed = if m.is_promo() { make_piece(us, m.promo_type()) } else { pc };
        d.add[0] = (placed, to as u8);
        d.n_add = 1;
        if m.is_capture() {
            let cap_sq = if m.is_ep() { to ^ 8 } else { to };
            d.sub[1] = (b.mailbox[cap_sq], cap_sq as u8);
            d.n_sub = 2;
        } else if m.is_castle() {
            let (rf, rt) = if m.flag() == KING_CASTLE { (to + 1, to - 1) } else { (to - 2, to + 1) };
            let rook = make_piece(us, ROOK);
            d.sub[1] = (rook, rf as u8);
            d.add[1] = (rook, rt as u8);
            d.n_sub = 2;
            d.n_add = 2;
        }
        d
    }

    #[inline(always)]
    pub fn apply(&self, net: &Network, src: &Accumulator, dst: &mut Accumulator) {
        for p in [WHITE, BLACK] {
            let s = &src.v[p];
            let d = &mut dst.v[p];
            let row = |(pc, sq): (u8, u8)| &net.w1[feature(p, pc, sq as usize)];
            match (self.n_add, self.n_sub) {
                (1, 1) => {
                    let (a0, s0) = (row(self.add[0]), row(self.sub[0]));
                    for i in 0..HIDDEN {
                        d[i] = s[i] + a0[i] - s0[i];
                    }
                }
                (1, 2) => {
                    let (a0, s0, s1) = (row(self.add[0]), row(self.sub[0]), row(self.sub[1]));
                    for i in 0..HIDDEN {
                        d[i] = s[i] + a0[i] - s0[i] - s1[i];
                    }
                }
                (2, 2) => {
                    let (a0, a1) = (row(self.add[0]), row(self.add[1]));
                    let (s0, s1) = (row(self.sub[0]), row(self.sub[1]));
                    for i in 0..HIDDEN {
                        d[i] = s[i] + a0[i] + a1[i] - s0[i] - s1[i];
                    }
                }
                _ => *d = *s,
            }
        }
    }
}

/// sum(clamp(acc, 0, QA)^2 * w). Relies on |w| <= 127 so that clamp(acc) * w fits in i16.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn screlu_dot(acc: &[i16; HIDDEN], w: &[i16]) -> i32 {
    use std::arch::aarch64::*;
    debug_assert!(w.len() >= HIDDEN);
    // SAFETY: NEON is always available on aarch64; all loads stay within HIDDEN elements.
    unsafe {
        let zero = vdupq_n_s16(0);
        let qa = vdupq_n_s16(QA as i16);
        let mut s0 = vdupq_n_s32(0);
        let mut s1 = vdupq_n_s32(0);
        let mut i = 0;
        while i < HIDDEN {
            let x = vminq_s16(vmaxq_s16(vld1q_s16(acc.as_ptr().add(i)), zero), qa);
            let xw = vmulq_s16(x, vld1q_s16(w.as_ptr().add(i)));
            s0 = vmlal_s16(s0, vget_low_s16(xw), vget_low_s16(x));
            s1 = vmlal_high_s16(s1, xw, x);
            i += 8;
        }
        vaddvq_s32(vaddq_s32(s0, s1))
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn screlu_dot(acc: &[i16; HIDDEN], w: &[i16]) -> i32 {
    let mut sum = 0i32;
    for i in 0..HIDDEN {
        let x = acc[i].clamp(0, QA as i16) as i32;
        sum += (x * w[i] as i32) * x;
    }
    sum
}

/// Network output in centipawns from the side to move's point of view.
#[inline(always)]
pub fn evaluate(net: &Network, acc: &Accumulator, stm: usize, piece_count: u32) -> i32 {
    let bucket = output_bucket(piece_count);
    let w = &net.w2[bucket];
    let sum = screlu_dot(&acc.v[stm], &w[..HIDDEN]) + screlu_dot(&acc.v[stm ^ 1], &w[HIDDEN..]);
    (sum / QA + net.b2[bucket]) * SCALE / (QA * QB)
}
