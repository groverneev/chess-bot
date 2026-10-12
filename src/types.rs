pub type Bitboard = u64;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

/// Pieces are encoded as `color * 6 + piece_type`; 12 means an empty square.
pub const NO_PIECE: u8 = 12;
pub const NO_SQ: u8 = 64;

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_H: Bitboard = FILE_A << 7;
pub const RANK_1: Bitboard = 0xFF;
pub const RANK_2: Bitboard = RANK_1 << 8;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_4: Bitboard = RANK_1 << 24;
pub const RANK_5: Bitboard = RANK_1 << 32;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_7: Bitboard = RANK_1 << 48;
pub const RANK_8: Bitboard = RANK_1 << 56;

pub const PIECE_CHARS: &[u8; 12] = b"PNBRQKpnbrqk";

const PIECE_TYPE: [u8; 13] = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6];
const PIECE_COLOR: [u8; 13] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2];

#[inline(always)]
pub const fn make_piece(color: usize, pt: usize) -> u8 {
    (color * 6 + pt) as u8
}

#[inline(always)]
pub const fn piece_type(pc: u8) -> usize {
    PIECE_TYPE[pc as usize] as usize
}

#[inline(always)]
pub const fn piece_color(pc: u8) -> usize {
    PIECE_COLOR[pc as usize] as usize
}

#[inline(always)]
pub const fn bit(sq: usize) -> Bitboard {
    1u64 << sq
}

#[inline(always)]
pub const fn lsb(bb: Bitboard) -> usize {
    bb.trailing_zeros() as usize
}

#[inline(always)]
pub const fn rank_of(sq: usize) -> usize {
    sq >> 3
}

#[inline(always)]
pub const fn file_of(sq: usize) -> usize {
    sq & 7
}

pub fn sq_to_str(sq: usize) -> String {
    let f = (b'a' + file_of(sq) as u8) as char;
    let r = (b'1' + rank_of(sq) as u8) as char;
    format!("{f}{r}")
}

pub fn str_to_sq(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some(((b[1] - b'1') * 8 + (b[0] - b'a')) as usize)
}

/// Iterates over the set squares of a bitboard, lowest first.
pub struct BitIter(pub Bitboard);

impl Iterator for BitIter {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let sq = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            Some(sq)
        }
    }
}
