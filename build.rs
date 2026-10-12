// Generates fancy magic bitboard tables for sliding pieces at build time.
//
// Outputs:
//   $OUT_DIR/magics.rs          - magic entries (mask, multiplier, shift, offset) per square
//   $OUT_DIR/slider_attacks.bin - the shared attack table, little-endian u64s

use std::env;
use std::fs;
use std::path::Path;

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn on_board(r: i32, f: i32) -> bool {
    (0..8).contains(&r) && (0..8).contains(&f)
}

fn slider_attacks(sq: usize, occ: u64, dirs: &[(i32, i32)]) -> u64 {
    let mut attacks = 0;
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    for &(dr, df) in dirs {
        let (mut r, mut f) = (r0 + dr, f0 + df);
        while on_board(r, f) {
            let bb = 1u64 << (r * 8 + f);
            attacks |= bb;
            if occ & bb != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

/// Squares whose occupancy matters for a slider on `sq` (rays without the board edge).
fn relevant_mask(sq: usize, dirs: &[(i32, i32)]) -> u64 {
    let mut mask = 0;
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    for &(dr, df) in dirs {
        let (mut r, mut f) = (r0 + dr, f0 + df);
        while on_board(r + dr, f + df) {
            mask |= 1u64 << (r * 8 + f);
            r += dr;
            f += df;
        }
    }
    mask
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2685821657736338717)
    }

    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

struct Found {
    mask: u64,
    magic: u64,
    shift: u32,
    table: Vec<u64>,
}

fn find_magic(sq: usize, dirs: &[(i32, i32)], rng: &mut Rng) -> Found {
    let mask = relevant_mask(sq, dirs);
    let bits = mask.count_ones();
    let n = 1usize << bits;

    let mut occs = Vec::with_capacity(n);
    let mut atts = Vec::with_capacity(n);
    let mut sub = 0u64;
    loop {
        occs.push(sub);
        atts.push(slider_attacks(sq, sub, dirs));
        sub = sub.wrapping_sub(mask) & mask;
        if sub == 0 {
            break;
        }
    }

    let shift = 64 - bits;
    let mut table = vec![0u64; n];
    let mut epoch = vec![0u32; n];
    let mut attempt = 0u32;
    loop {
        let magic = rng.sparse();
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }
        attempt += 1;
        let mut ok = true;
        for i in 0..n {
            let idx = (occs[i].wrapping_mul(magic) >> shift) as usize;
            if epoch[idx] < attempt {
                epoch[idx] = attempt;
                table[idx] = atts[i];
            } else if table[idx] != atts[i] {
                ok = false;
                break;
            }
        }
        if ok {
            return Found { mask, magic, shift, table };
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out_dir = env::var("OUT_DIR").unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    let mut table: Vec<u64> = Vec::new();
    let mut src = String::new();

    for (name, dirs) in [("ROOK_MAGICS", &ROOK_DIRS), ("BISHOP_MAGICS", &BISHOP_DIRS)] {
        src.push_str(&format!("pub static {name}: [Magic; 64] = [\n"));
        for sq in 0..64 {
            let found = find_magic(sq, dirs, &mut rng);
            src.push_str(&format!(
                "    Magic {{ mask: {:#018x}, magic: {:#018x}, shift: {}, offset: {} }},\n",
                found.mask,
                found.magic,
                found.shift,
                table.len()
            ));
            table.extend_from_slice(&found.table);
        }
        src.push_str("];\n");
    }
    src.push_str(&format!("pub const SLIDER_TABLE_SIZE: usize = {};\n", table.len()));

    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(Path::new(&out_dir).join("magics.rs"), src).unwrap();
    fs::write(Path::new(&out_dir).join("slider_attacks.bin"), bytes).unwrap();

    // Embed the default network if present; an empty file makes the engine fall back to HCE.
    let net_path = env::var("FERRUM_NET").unwrap_or_else(|_| "nets/default.nnue".to_string());
    println!("cargo:rerun-if-changed={net_path}");
    println!("cargo:rerun-if-env-changed=FERRUM_NET");
    let net = fs::read(&net_path).unwrap_or_default();
    fs::write(Path::new(&out_dir).join("net.bin"), net).unwrap();
}
