//! Compact 32-byte training-position format shared by datagen and the trainer.

use crate::board::Board;
use crate::types::*;

pub const RECORD_SIZE: usize = 32;

/// result: 0 = black win, 1 = draw, 2 = white win. score is centipawns from White's view.
#[derive(Clone, Copy, Default)]
pub struct PackedPos {
    pub occ: u64,
    /// One nibble per occupied square in ascending square order: color << 3 | piece type.
    pub pieces: [u8; 16],
    pub score: i16,
    pub result: u8,
    pub stm: u8,
}

impl PackedPos {
    pub fn pack(b: &Board, white_score: i16) -> PackedPos {
        let occ = b.occupied();
        let mut pieces = [0u8; 16];
        for (i, sq) in BitIter(occ).enumerate() {
            let pc = b.mailbox[sq];
            let nib = ((piece_color(pc) << 3) | piece_type(pc)) as u8;
            pieces[i / 2] |= nib << (4 * (i % 2));
        }
        PackedPos { occ, pieces, score: white_score, result: 1, stm: b.stm as u8 }
    }

    pub fn to_bytes(&self) -> [u8; RECORD_SIZE] {
        let mut out = [0u8; RECORD_SIZE];
        out[0..8].copy_from_slice(&self.occ.to_le_bytes());
        out[8..24].copy_from_slice(&self.pieces);
        out[24..26].copy_from_slice(&self.score.to_le_bytes());
        out[26] = self.result;
        out[27] = self.stm;
        out
    }

    pub fn from_bytes(b: &[u8]) -> PackedPos {
        let mut pieces = [0u8; 16];
        pieces.copy_from_slice(&b[8..24]);
        PackedPos {
            occ: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            pieces,
            score: i16::from_le_bytes([b[24], b[25]]),
            result: b[26],
            stm: b[27],
        }
    }

    /// Calls `f(square, color, piece_type)` for every piece.
    #[inline(always)]
    pub fn for_each_piece(&self, mut f: impl FnMut(usize, usize, usize)) {
        for (i, sq) in BitIter(self.occ).enumerate() {
            let nib = (self.pieces[i / 2] >> (4 * (i % 2))) & 0xF;
            f(sq, (nib >> 3) as usize, (nib & 7) as usize);
        }
    }

    pub fn to_fen(&self) -> String {
        let mut board = [b'.'; 64];
        self.for_each_piece(|sq, c, pt| board[sq] = PIECE_CHARS[c * 6 + pt]);
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let ch = board[rank * 8 + file];
                if ch == b'.' {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    fen.push(ch as char);
                }
            }
            if empty > 0 {
                fen.push((b'0' + empty) as char);
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push_str(if self.stm == 0 { " w - - 0 1" } else { " b - - 0 1" });
        fen
    }
}
