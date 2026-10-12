//! Fully legal move generation using check and pin masks.
//!
//! Moves are split into "noisy" (captures, every capture-promotion, quiet queen promotions)
//! and "quiet" (everything else) so the search can generate them in stages.

use crate::attacks::*;
use crate::board::*;
use crate::moves::*;
use crate::types::*;

pub fn generate_all(b: &Board, list: &mut MoveList) {
    generate::<true, true>(b, list);
}

pub fn generate_noisy(b: &Board, list: &mut MoveList) {
    generate::<true, false>(b, list);
}

pub fn generate_quiet(b: &Board, list: &mut MoveList) {
    generate::<false, true>(b, list);
}

#[inline(always)]
fn push_targets(list: &mut MoveList, from: usize, targets: Bitboard, flag: u16) {
    for to in BitIter(targets) {
        list.push(Move::new(from, to, flag));
    }
}

fn pinned_pieces(b: &Board, us: usize, ksq: usize, occ: Bitboard) -> Bitboard {
    let them = us ^ 1;
    let theirs = b.colors[them];
    let snipers = (rook_attacks(ksq, theirs) & (b.pieces[ROOK] | b.pieces[QUEEN]) & theirs)
        | (bishop_attacks(ksq, theirs) & (b.pieces[BISHOP] | b.pieces[QUEEN]) & theirs);
    let mut pinned = 0;
    for s in BitIter(snipers) {
        let blockers = between(ksq, s) & occ;
        if blockers.count_ones() == 1 {
            pinned |= blockers & b.colors[us];
        }
    }
    pinned
}

fn generate<const GEN_NOISY: bool, const GEN_QUIET: bool>(b: &Board, list: &mut MoveList) {
    let us = b.stm;
    let them = us ^ 1;
    let occ = b.occupied();
    let ours = b.colors[us];
    let theirs = b.colors[them];
    let ksq = b.king_sq(us);
    let checkers = b.checkers;

    let king_targets = KING_ATTACKS[ksq] & !ours;
    let occ_no_king = occ ^ bit(ksq);
    if GEN_NOISY {
        for to in BitIter(king_targets & theirs) {
            if !b.is_attacked(to, them, occ_no_king) {
                list.push(Move::new(ksq, to, CAPTURE));
            }
        }
    }
    if GEN_QUIET {
        for to in BitIter(king_targets & !occ) {
            if !b.is_attacked(to, them, occ_no_king) {
                list.push(Move::new(ksq, to, QUIET));
            }
        }
    }
    if checkers & checkers.wrapping_sub(1) != 0 {
        return; // double check: only king moves
    }

    let check_mask = if checkers != 0 { between(ksq, lsb(checkers)) | checkers } else { !0 };
    let pinned = pinned_pieces(b, us, ksq, occ);
    let cap_mask = theirs & check_mask;
    let quiet_mask = !occ & check_mask;

    for from in BitIter(b.pieces[KNIGHT] & ours & !pinned) {
        let att = KNIGHT_ATTACKS[from];
        if GEN_NOISY {
            push_targets(list, from, att & cap_mask, CAPTURE);
        }
        if GEN_QUIET {
            push_targets(list, from, att & quiet_mask, QUIET);
        }
    }

    for from in BitIter((b.pieces[BISHOP] | b.pieces[QUEEN]) & ours) {
        let mut att = bishop_attacks(from, occ);
        if pinned & bit(from) != 0 {
            att &= line(ksq, from);
        }
        if GEN_NOISY {
            push_targets(list, from, att & cap_mask, CAPTURE);
        }
        if GEN_QUIET {
            push_targets(list, from, att & quiet_mask, QUIET);
        }
    }

    for from in BitIter((b.pieces[ROOK] | b.pieces[QUEEN]) & ours) {
        let mut att = rook_attacks(from, occ);
        if pinned & bit(from) != 0 {
            att &= line(ksq, from);
        }
        if GEN_NOISY {
            push_targets(list, from, att & cap_mask, CAPTURE);
        }
        if GEN_QUIET {
            push_targets(list, from, att & quiet_mask, QUIET);
        }
    }

    gen_pawns::<GEN_NOISY, GEN_QUIET>(b, list, ksq, pinned, check_mask);

    if GEN_QUIET && checkers == 0 {
        let base = if us == WHITE { 0 } else { 56 };
        let (ks, qs) = if us == WHITE { (WK, WQ) } else { (BK, BQ) };
        if b.castling & ks != 0
            && occ & (0x60u64 << base) == 0
            && !b.is_attacked(base + 5, them, occ)
            && !b.is_attacked(base + 6, them, occ)
        {
            list.push(Move::new(base + 4, base + 6, KING_CASTLE));
        }
        if b.castling & qs != 0
            && occ & (0x0Eu64 << base) == 0
            && !b.is_attacked(base + 3, them, occ)
            && !b.is_attacked(base + 2, them, occ)
        {
            list.push(Move::new(base + 4, base + 2, QUEEN_CASTLE));
        }
    }
}

#[inline(always)]
fn shift(bb: Bitboard, n: i32) -> Bitboard {
    if n > 0 {
        bb << n
    } else {
        bb >> -n
    }
}

#[inline(always)]
fn gen_pawns<const GEN_NOISY: bool, const GEN_QUIET: bool>(
    b: &Board,
    list: &mut MoveList,
    ksq: usize,
    pinned: Bitboard,
    check_mask: Bitboard,
) {
    let us = b.stm;
    let them = us ^ 1;
    let occ = b.occupied();
    let theirs = b.colors[them];
    let pawns = b.pieces_of(us, PAWN);
    let empty = !occ;

    let (up, rank7, rank3) = if us == WHITE { (8i32, RANK_7, RANK_3) } else { (-8i32, RANK_2, RANK_6) };
    let pin_ok = |from: usize, to: usize| pinned & bit(from) == 0 || line(ksq, from) & bit(to) != 0;
    let promo_pawns = pawns & rank7;
    let other_pawns = pawns & !rank7;

    if GEN_QUIET {
        let single = shift(other_pawns, up) & empty;
        let double = shift(single & rank3, up) & empty & check_mask;
        for to in BitIter(single & check_mask) {
            let from = (to as i32 - up) as usize;
            if pin_ok(from, to) {
                list.push(Move::new(from, to, QUIET));
            }
        }
        for to in BitIter(double) {
            let from = (to as i32 - 2 * up) as usize;
            if pin_ok(from, to) {
                list.push(Move::new(from, to, DOUBLE_PUSH));
            }
        }
    }

    if promo_pawns != 0 {
        for to in BitIter(shift(promo_pawns, up) & empty & check_mask) {
            let from = (to as i32 - up) as usize;
            if pin_ok(from, to) {
                if GEN_NOISY {
                    list.push(Move::new(from, to, PROMO | 3));
                }
                if GEN_QUIET {
                    for p in 0..3 {
                        list.push(Move::new(from, to, PROMO | p));
                    }
                }
            }
        }
        if GEN_NOISY {
            for (mask, delta) in [(!FILE_A, up - 1), (!FILE_H, up + 1)] {
                for to in BitIter(shift(promo_pawns & mask, delta) & theirs & check_mask) {
                    let from = (to as i32 - delta) as usize;
                    if pin_ok(from, to) {
                        for p in (0..4).rev() {
                            list.push(Move::new(from, to, PROMO_CAPTURE | p));
                        }
                    }
                }
            }
        }
    }

    if GEN_NOISY {
        for (mask, delta) in [(!FILE_A, up - 1), (!FILE_H, up + 1)] {
            for to in BitIter(shift(other_pawns & mask, delta) & theirs & check_mask) {
                let from = (to as i32 - delta) as usize;
                if pin_ok(from, to) {
                    list.push(Move::new(from, to, CAPTURE));
                }
            }
        }

        if b.ep != NO_SQ {
            let ep = b.ep as usize;
            let cap_sq = ep ^ 8;
            if check_mask & (bit(ep) | bit(cap_sq)) != 0 {
                let rq = (b.pieces[ROOK] | b.pieces[QUEEN]) & theirs;
                let bq = (b.pieces[BISHOP] | b.pieces[QUEEN]) & theirs;
                for from in BitIter(PAWN_ATTACKS[them][ep] & pawns) {
                    let occ2 = (occ ^ bit(from) ^ bit(cap_sq)) | bit(ep);
                    if rook_attacks(ksq, occ2) & rq == 0 && bishop_attacks(ksq, occ2) & bq == 0 {
                        list.push(Move::new(from, ep, EP_CAPTURE));
                    }
                }
            }
        }
    }
}

/// Parses a UCI move string (e.g. "e2e4", "e7e8q") against the legal moves of `b`.
pub fn parse_uci_move(b: &Board, s: &str) -> Option<Move> {
    let mut list = MoveList::new();
    generate_all(b, &mut list);
    list.as_slice().iter().copied().find(|m| m.to_uci() == s)
}

pub fn perft(b: &Board, depth: u32) -> u64 {
    let mut list = MoveList::new();
    generate_all(b, &mut list);
    if depth <= 1 {
        return if depth == 0 { 1 } else { list.len as u64 };
    }
    let mut nodes = 0;
    for &m in list.as_slice() {
        let mut child = *b;
        child.make_move(m);
        nodes += perft(&child, depth - 1);
    }
    nodes
}

/// Perft through the staged (noisy then quiet) generators, to check the split is a partition.
pub fn perft_split(b: &Board, depth: u32) -> u64 {
    let mut noisy = MoveList::new();
    let mut quiet = MoveList::new();
    generate_noisy(b, &mut noisy);
    generate_quiet(b, &mut quiet);
    if depth == 1 {
        return (noisy.len + quiet.len) as u64;
    }
    let mut nodes = 0;
    for &m in noisy.as_slice().iter().chain(quiet.as_slice()) {
        debug_assert_eq!(m.is_noisy(), noisy.contains(m));
        let mut child = *b;
        child.make_move(m);
        nodes += perft_split(&child, depth - 1);
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUITE: &[(&str, &[u64])] = &[
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &[20, 400, 8902, 197281, 4865609]),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", &[48, 2039, 97862, 4085603]),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2812, 43238, 674624]),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &[6, 264, 9467, 422333]),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[44, 1486, 62379, 2103487]),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", &[46, 2079, 89890, 3894594]),
    ];

    #[test]
    fn perft_suite() {
        for (fen, counts) in SUITE {
            let b = Board::from_fen(fen).unwrap();
            assert_eq!(&b.to_fen(), fen);
            for (i, &expected) in counts.iter().enumerate() {
                assert_eq!(perft(&b, i as u32 + 1), expected, "{fen} depth {}", i + 1);
            }
        }
    }

    #[test]
    fn staged_generation_partitions_moves() {
        for (fen, counts) in SUITE {
            let b = Board::from_fen(fen).unwrap();
            assert_eq!(perft_split(&b, 3), counts[2], "{fen}");
        }
    }

    #[test]
    fn incremental_hash_matches_fen_hash() {
        let mut b = Board::from_fen(SUITE[1].0).unwrap();
        for _ in 0..40 {
            let mut list = MoveList::new();
            generate_all(&b, &mut list);
            if list.len == 0 {
                break;
            }
            b.make_move(list.moves[list.len / 2]);
            let fresh = Board::from_fen(&b.to_fen()).unwrap();
            assert_eq!(fresh.hash, b.hash, "{}", b.to_fen());
            assert_eq!(fresh.pawn_hash, b.pawn_hash);
            assert_eq!(fresh.nonpawn_hash, b.nonpawn_hash);
        }
    }
}
