use crate::types::*;

#[derive(Clone, Copy)]
pub struct Magic {
    pub mask: u64,
    pub magic: u64,
    pub shift: u32,
    pub offset: u32,
}

include!(concat!(env!("OUT_DIR"), "/magics.rs"));

static SLIDER_ATTACKS: [u64; SLIDER_TABLE_SIZE] =
    unsafe { std::mem::transmute(*include_bytes!(concat!(env!("OUT_DIR"), "/slider_attacks.bin"))) };

#[inline(always)]
fn magic_index(m: &Magic, occ: Bitboard) -> usize {
    m.offset as usize + ((occ & m.mask).wrapping_mul(m.magic) >> m.shift) as usize
}

#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: Bitboard) -> Bitboard {
    let idx = magic_index(&BISHOP_MAGICS[sq], occ);
    // SAFETY: build.rs guarantees every masked occupancy maps inside this square's slice.
    unsafe { *SLIDER_ATTACKS.get_unchecked(idx) }
}

#[inline(always)]
pub fn rook_attacks(sq: usize, occ: Bitboard) -> Bitboard {
    let idx = magic_index(&ROOK_MAGICS[sq], occ);
    // SAFETY: see bishop_attacks.
    unsafe { *SLIDER_ATTACKS.get_unchecked(idx) }
}

#[inline(always)]
pub fn queen_attacks(sq: usize, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

const fn gen_leaper(deltas: &[(i32, i32)]) -> [Bitboard; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let r = (sq / 8) as i32;
        let f = (sq % 8) as i32;
        let mut i = 0;
        while i < deltas.len() {
            let nr = r + deltas[i].0;
            let nf = f + deltas[i].1;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                table[sq] |= 1u64 << (nr * 8 + nf);
            }
            i += 1;
        }
        sq += 1;
    }
    table
}

pub static KNIGHT_ATTACKS: [Bitboard; 64] =
    gen_leaper(&[(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]);
pub static KING_ATTACKS: [Bitboard; 64] =
    gen_leaper(&[(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]);
/// PAWN_ATTACKS[color][sq]: squares attacked by a pawn of `color` standing on `sq`.
pub static PAWN_ATTACKS: [[Bitboard; 64]; 2] = [gen_leaper(&[(1, -1), (1, 1)]), gen_leaper(&[(-1, -1), (-1, 1)])];

const fn gen_between_line() -> ([[Bitboard; 64]; 64], [[Bitboard; 64]; 64]) {
    let mut between = [[0u64; 64]; 64];
    let mut line = [[0u64; 64]; 64];
    let dirs: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
    let mut a = 0;
    while a < 64 {
        let r0 = (a / 8) as i32;
        let f0 = (a % 8) as i32;
        let mut d = 0;
        while d < 8 {
            let (dr, df) = dirs[d];
            let mut full = 1u64 << a;
            let (mut r, mut f) = (r0 + dr, f0 + df);
            while r >= 0 && r < 8 && f >= 0 && f < 8 {
                full |= 1u64 << (r * 8 + f);
                r += dr;
                f += df;
            }
            let (mut r, mut f) = (r0 - dr, f0 - df);
            while r >= 0 && r < 8 && f >= 0 && f < 8 {
                full |= 1u64 << (r * 8 + f);
                r -= dr;
                f -= df;
            }
            let mut ray = 0u64;
            let (mut r, mut f) = (r0 + dr, f0 + df);
            while r >= 0 && r < 8 && f >= 0 && f < 8 {
                let s = (r * 8 + f) as usize;
                between[a][s] = ray;
                line[a][s] = full;
                ray |= 1u64 << s;
                r += dr;
                f += df;
            }
            d += 1;
        }
        a += 1;
    }
    (between, line)
}

static BETWEEN_LINE: ([[Bitboard; 64]; 64], [[Bitboard; 64]; 64]) = gen_between_line();

/// Squares strictly between `a` and `b` if they share a rank, file or diagonal, else empty.
#[inline(always)]
pub fn between(a: usize, b: usize) -> Bitboard {
    BETWEEN_LINE.0[a][b]
}

/// The full board-spanning line through `a` and `b` if aligned, else empty.
#[inline(always)]
pub fn line(a: usize, b: usize) -> Bitboard {
    BETWEEN_LINE.1[a][b]
}
