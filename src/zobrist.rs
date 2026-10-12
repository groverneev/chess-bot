pub struct Zobrist {
    pub pieces: [[u64; 64]; 12],
    pub castling: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
}

const fn splitmix(state: u64) -> (u64, u64) {
    let s = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (s, z ^ (z >> 31))
}

const fn generate() -> Zobrist {
    let mut z = Zobrist { pieces: [[0; 64]; 12], castling: [0; 16], ep: [0; 8], side: 0 };
    let mut state = 0x0F3C_5A69_D2E1_B487u64;
    let mut p = 0;
    while p < 12 {
        let mut s = 0;
        while s < 64 {
            let (ns, v) = splitmix(state);
            state = ns;
            z.pieces[p][s] = v;
            s += 1;
        }
        p += 1;
    }
    // Castling keys are the XOR of one key per right so that the encoding is consistent.
    let mut rights = [0u64; 4];
    let mut i = 0;
    while i < 4 {
        let (ns, v) = splitmix(state);
        state = ns;
        rights[i] = v;
        i += 1;
    }
    let mut mask = 0;
    while mask < 16 {
        let mut key = 0;
        let mut i = 0;
        while i < 4 {
            if mask & (1 << i) != 0 {
                key ^= rights[i];
            }
            i += 1;
        }
        z.castling[mask] = key;
        mask += 1;
    }
    let mut f = 0;
    while f < 8 {
        let (ns, v) = splitmix(state);
        state = ns;
        z.ep[f] = v;
        f += 1;
    }
    z.side = splitmix(state).1;
    z
}

pub static ZOBRIST: Zobrist = generate();
