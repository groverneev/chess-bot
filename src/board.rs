use crate::attacks::*;
use crate::moves::*;
use crate::types::*;
use crate::zobrist::ZOBRIST;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub const WK: u8 = 1;
pub const WQ: u8 = 2;
pub const BK: u8 = 4;
pub const BQ: u8 = 8;

/// Castling rights that survive a move touching each square.
static CASTLE_MASK: [u8; 64] = {
    let mut m = [15u8; 64];
    m[0] = 15 & !WQ;
    m[4] = 15 & !(WK | WQ);
    m[7] = 15 & !WK;
    m[56] = 15 & !BQ;
    m[60] = 15 & !(BK | BQ);
    m[63] = 15 & !BK;
    m
};

pub const SEE_VALUES: [i32; 7] = [100, 300, 300, 500, 900, 0, 0];

#[derive(Clone, Copy)]
pub struct Board {
    pub pieces: [Bitboard; 6],
    pub colors: [Bitboard; 2],
    pub mailbox: [u8; 64],
    pub stm: usize,
    pub castling: u8,
    pub ep: u8,
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
    pub pawn_hash: u64,
    pub nonpawn_hash: [u64; 2],
    pub checkers: Bitboard,
}

impl Board {
    fn empty() -> Board {
        Board {
            pieces: [0; 6],
            colors: [0; 2],
            mailbox: [NO_PIECE; 64],
            stm: WHITE,
            castling: 0,
            ep: NO_SQ,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            pawn_hash: 0,
            nonpawn_hash: [0; 2],
            checkers: 0,
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen(STARTPOS).unwrap()
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let mut b = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("invalid fen: {fen}"));
        }

        let (mut rank, mut file) = (7i32, 0i32);
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += ch as i32 - '0' as i32,
                _ => {
                    let pc = PIECE_CHARS
                        .iter()
                        .position(|&c| c as char == ch)
                        .ok_or_else(|| format!("invalid piece '{ch}' in fen"))?;
                    if !(0..8).contains(&rank) || !(0..8).contains(&file) {
                        return Err("fen board overflows".into());
                    }
                    b.add_piece((rank * 8 + file) as usize, pc as u8);
                    file += 1;
                }
            }
        }

        b.stm = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            s => return Err(format!("invalid side to move '{s}'")),
        };

        if let Some(c) = parts.get(2) {
            for ch in c.chars() {
                match ch {
                    'K' => b.castling |= WK,
                    'Q' => b.castling |= WQ,
                    'k' => b.castling |= BK,
                    'q' => b.castling |= BQ,
                    '-' => {}
                    _ => return Err(format!("invalid castling rights '{c}'")),
                }
            }
        }
        // Drop rights that the piece placement cannot support.
        let wk = make_piece(WHITE, KING);
        let wr = make_piece(WHITE, ROOK);
        let bk = make_piece(BLACK, KING);
        let br = make_piece(BLACK, ROOK);
        if b.mailbox[4] != wk || b.mailbox[7] != wr {
            b.castling &= !WK;
        }
        if b.mailbox[4] != wk || b.mailbox[0] != wr {
            b.castling &= !WQ;
        }
        if b.mailbox[60] != bk || b.mailbox[63] != br {
            b.castling &= !BK;
        }
        if b.mailbox[60] != bk || b.mailbox[56] != br {
            b.castling &= !BQ;
        }

        if let Some(&ep) = parts.get(3) {
            if ep != "-" {
                let sq = str_to_sq(ep).ok_or_else(|| format!("invalid ep square '{ep}'"))?;
                if PAWN_ATTACKS[b.stm ^ 1][sq] & b.pieces_of(b.stm, PAWN) != 0 {
                    b.ep = sq as u8;
                }
            }
        }
        b.halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        b.fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);

        for c in [WHITE, BLACK] {
            if b.pieces_of(c, KING).count_ones() != 1 {
                return Err("each side needs exactly one king".into());
            }
        }

        b.hash ^= ZOBRIST.castling[b.castling as usize];
        if b.ep != NO_SQ {
            b.hash ^= ZOBRIST.ep[file_of(b.ep as usize)];
        }
        if b.stm == BLACK {
            b.hash ^= ZOBRIST.side;
        }
        let them = b.stm ^ 1;
        if b.is_attacked(b.king_sq(them), b.stm, b.occupied()) {
            return Err("side not to move is in check".into());
        }
        b.checkers = b.attackers_to(b.king_sq(b.stm), b.occupied()) & b.colors[them];
        Ok(b)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let pc = self.mailbox[rank * 8 + file];
                if pc == NO_PIECE {
                    empty += 1;
                } else {
                    if empty > 0 {
                        s.push_str(&empty.to_string());
                        empty = 0;
                    }
                    s.push(PIECE_CHARS[pc as usize] as char);
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == WHITE { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (bit, ch) in [(WK, 'K'), (WQ, 'Q'), (BK, 'k'), (BQ, 'q')] {
                if self.castling & bit != 0 {
                    s.push(ch);
                }
            }
        }
        s.push(' ');
        if self.ep == NO_SQ {
            s.push('-');
        } else {
            s.push_str(&sq_to_str(self.ep as usize));
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[WHITE] | self.colors[BLACK]
    }

    #[inline(always)]
    pub fn pieces_of(&self, c: usize, pt: usize) -> Bitboard {
        self.pieces[pt] & self.colors[c]
    }

    #[inline(always)]
    pub fn king_sq(&self, c: usize) -> usize {
        lsb(self.pieces[KING] & self.colors[c])
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }

    #[inline(always)]
    fn add_piece(&mut self, sq: usize, pc: u8) {
        let bb = bit(sq);
        let (pt, c) = (piece_type(pc), piece_color(pc));
        self.pieces[pt] |= bb;
        self.colors[c] |= bb;
        self.mailbox[sq] = pc;
        let key = ZOBRIST.pieces[pc as usize][sq];
        self.hash ^= key;
        if pt == PAWN {
            self.pawn_hash ^= key;
        } else {
            self.nonpawn_hash[c] ^= key;
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: usize) {
        let pc = self.mailbox[sq];
        let bb = bit(sq);
        let (pt, c) = (piece_type(pc), piece_color(pc));
        self.pieces[pt] ^= bb;
        self.colors[c] ^= bb;
        self.mailbox[sq] = NO_PIECE;
        let key = ZOBRIST.pieces[pc as usize][sq];
        self.hash ^= key;
        if pt == PAWN {
            self.pawn_hash ^= key;
        } else {
            self.nonpawn_hash[c] ^= key;
        }
    }

    #[inline(always)]
    fn move_piece(&mut self, from: usize, to: usize) {
        let pc = self.mailbox[from];
        let bb = bit(from) | bit(to);
        let (pt, c) = (piece_type(pc), piece_color(pc));
        self.pieces[pt] ^= bb;
        self.colors[c] ^= bb;
        self.mailbox[from] = NO_PIECE;
        self.mailbox[to] = pc;
        let key = ZOBRIST.pieces[pc as usize][from] ^ ZOBRIST.pieces[pc as usize][to];
        self.hash ^= key;
        if pt == PAWN {
            self.pawn_hash ^= key;
        } else {
            self.nonpawn_hash[c] ^= key;
        }
    }

    #[inline(always)]
    pub fn attackers_to(&self, sq: usize, occ: Bitboard) -> Bitboard {
        let bq = self.pieces[BISHOP] | self.pieces[QUEEN];
        let rq = self.pieces[ROOK] | self.pieces[QUEEN];
        (PAWN_ATTACKS[WHITE][sq] & self.pieces_of(BLACK, PAWN))
            | (PAWN_ATTACKS[BLACK][sq] & self.pieces_of(WHITE, PAWN))
            | (KNIGHT_ATTACKS[sq] & self.pieces[KNIGHT])
            | (KING_ATTACKS[sq] & self.pieces[KING])
            | (bishop_attacks(sq, occ) & bq)
            | (rook_attacks(sq, occ) & rq)
    }

    /// Is `sq` attacked by any piece of color `by`, given occupancy `occ`?
    #[inline(always)]
    pub fn is_attacked(&self, sq: usize, by: usize, occ: Bitboard) -> bool {
        let them = self.colors[by];
        (PAWN_ATTACKS[by ^ 1][sq] & self.pieces[PAWN] & them) != 0
            || (KNIGHT_ATTACKS[sq] & self.pieces[KNIGHT] & them) != 0
            || (KING_ATTACKS[sq] & self.pieces[KING] & them) != 0
            || (bishop_attacks(sq, occ) & (self.pieces[BISHOP] | self.pieces[QUEEN]) & them) != 0
            || (rook_attacks(sq, occ) & (self.pieces[ROOK] | self.pieces[QUEEN]) & them) != 0
    }

    /// Plays a legal move in place.
    pub fn make_move(&mut self, m: Move) {
        let us = self.stm;
        let them = us ^ 1;
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let pt = piece_type(self.mailbox[from]);

        self.hash ^= ZOBRIST.side;
        if self.ep != NO_SQ {
            self.hash ^= ZOBRIST.ep[file_of(self.ep as usize)];
            self.ep = NO_SQ;
        }
        self.halfmove += 1;

        if m.is_capture() {
            let cap_sq = if flag == EP_CAPTURE { to ^ 8 } else { to };
            self.remove_piece(cap_sq);
            self.halfmove = 0;
        }
        self.move_piece(from, to);

        if pt == PAWN {
            self.halfmove = 0;
            if m.is_promo() {
                self.remove_piece(to);
                self.add_piece(to, make_piece(us, m.promo_type()));
            } else if flag == DOUBLE_PUSH {
                let ep = to ^ 8;
                if PAWN_ATTACKS[us][ep] & self.pieces_of(them, PAWN) != 0 {
                    self.ep = ep as u8;
                    self.hash ^= ZOBRIST.ep[file_of(ep)];
                }
            }
        } else if flag == KING_CASTLE {
            self.move_piece(to + 1, to - 1);
        } else if flag == QUEEN_CASTLE {
            self.move_piece(to - 2, to + 1);
        }

        let rights = self.castling & CASTLE_MASK[from] & CASTLE_MASK[to];
        if rights != self.castling {
            self.hash ^= ZOBRIST.castling[self.castling as usize] ^ ZOBRIST.castling[rights as usize];
            self.castling = rights;
        }

        if us == BLACK {
            self.fullmove += 1;
        }
        self.stm = them;
        self.checkers = self.attackers_to(self.king_sq(them), self.occupied()) & self.colors[us];
    }

    /// Passes the turn. Must not be called while in check.
    pub fn make_null(&mut self) {
        self.hash ^= ZOBRIST.side;
        if self.ep != NO_SQ {
            self.hash ^= ZOBRIST.ep[file_of(self.ep as usize)];
            self.ep = NO_SQ;
        }
        self.halfmove += 1;
        if self.stm == BLACK {
            self.fullmove += 1;
        }
        self.stm ^= 1;
        self.checkers = 0;
    }

    /// Type of the piece captured by `m`, or KING as a "nothing" sentinel for quiet promotions.
    #[inline(always)]
    pub fn captured_type(&self, m: Move) -> usize {
        if m.is_ep() {
            PAWN
        } else if m.is_capture() {
            piece_type(self.mailbox[m.to()])
        } else {
            KING
        }
    }

    #[inline(always)]
    pub fn moved_piece(&self, m: Move) -> u8 {
        self.mailbox[m.from()]
    }

    pub fn has_non_pawn_material(&self, c: usize) -> bool {
        (self.colors[c] & !(self.pieces[PAWN] | self.pieces[KING])) != 0
    }

    pub fn is_insufficient_material(&self) -> bool {
        if self.pieces[PAWN] | self.pieces[ROOK] | self.pieces[QUEEN] != 0 {
            return false;
        }
        (self.pieces[KNIGHT] | self.pieces[BISHOP]).count_ones() <= 1
    }

    /// Static exchange evaluation: does `m` win at least `threshold` material?
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        if m.is_castle() || m.is_ep() || m.is_promo() {
            return 0 >= threshold;
        }
        let (from, to) = (m.from(), m.to());

        let mut swap = SEE_VALUES[piece_type(self.mailbox[to])] - threshold;
        if swap < 0 {
            return false;
        }
        swap = SEE_VALUES[piece_type(self.mailbox[from])] - swap;
        if swap <= 0 {
            return true;
        }

        let mut occ = self.occupied() ^ bit(from) ^ bit(to);
        let mut stm = self.stm;
        let mut attackers = self.attackers_to(to, occ);
        let bq = self.pieces[BISHOP] | self.pieces[QUEEN];
        let rq = self.pieces[ROOK] | self.pieces[QUEEN];
        let mut res = true;

        loop {
            stm ^= 1;
            attackers &= occ;
            let stm_attackers = attackers & self.colors[stm];
            if stm_attackers == 0 {
                break;
            }
            res = !res;

            let mut pt = PAWN;
            while stm_attackers & self.pieces[pt] == 0 {
                pt += 1;
            }
            if pt == KING {
                // The king may only capture if the opponent has no recapture.
                return if attackers & self.colors[stm ^ 1] != 0 { !res } else { res };
            }

            swap = SEE_VALUES[pt] - swap;
            if swap < res as i32 {
                break;
            }
            occ ^= bit(lsb(stm_attackers & self.pieces[pt]));
            if pt == PAWN || pt == BISHOP || pt == QUEEN {
                attackers |= bishop_attacks(to, occ) & bq;
            }
            if pt == ROOK || pt == QUEEN {
                attackers |= rook_attacks(to, occ) & rq;
            }
        }
        res
    }

    pub fn print(&self) {
        for rank in (0..8).rev() {
            let mut line = format!("{} ", rank + 1);
            for file in 0..8 {
                let pc = self.mailbox[rank * 8 + file];
                line.push(if pc == NO_PIECE { '.' } else { PIECE_CHARS[pc as usize] as char });
                line.push(' ');
            }
            println!("{line}");
        }
        println!("  a b c d e f g h");
        println!("fen: {}", self.to_fen());
        println!("hash: {:016x}", self.hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::parse_uci_move;

    fn see(fen: &str, mv: &str, threshold: i32) -> bool {
        let b = Board::from_fen(fen).unwrap();
        let m = parse_uci_move(&b, mv).expect("legal move");
        b.see_ge(m, threshold)
    }

    #[test]
    fn static_exchange() {
        // Pawn takes an undefended knight.
        assert!(see("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1", "e4d5", 300));
        // Queen takes a pawn defended by a pawn: loses the queen.
        assert!(!see("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1", "d2d5", 0));
        // Rook takes rook defended by a rook, with our rook battery behind: even trade.
        assert!(see("3rk3/3r4/8/8/8/8/3R4/3RK3 w - - 0 1", "d2d7", 0));
        assert!(!see("3rk3/3r4/8/8/8/8/3R4/3RK3 w - - 0 1", "d2d7", 1));
    }
}
