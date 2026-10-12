//! Lock-free shared transposition table (XOR-verified entries, 4-way buckets).

use crate::moves::Move;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_LOWER: u8 = 1;
pub const BOUND_UPPER: u8 = 2;
pub const BOUND_EXACT: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: u8,
    pub pv: bool,
}

#[derive(Default)]
struct Slot {
    key: AtomicU64,
    data: AtomicU64,
}

#[repr(align(64))]
#[derive(Default)]
struct Bucket {
    slots: [Slot; 4],
}

pub struct TT {
    buckets: *mut Bucket,
    len: usize,
    age: AtomicU8,
}

// SAFETY: all access to the buckets goes through atomics.
unsafe impl Send for TT {}
unsafe impl Sync for TT {}

#[inline(always)]
fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: u8, pv: bool, age: u8) -> u64 {
    (mv.0 as u64)
        | ((score as i16 as u16 as u64) << 16)
        | ((eval as i16 as u16 as u64) << 32)
        | ((depth.clamp(0, 255) as u64) << 48)
        | ((bound as u64) << 56)
        | ((pv as u64) << 58)
        | ((age as u64) << 59)
}

#[inline(always)]
fn unpack(d: u64) -> TTEntry {
    TTEntry {
        mv: Move(d as u16),
        score: (d >> 16) as u16 as i16 as i32,
        eval: (d >> 32) as u16 as i16 as i32,
        depth: ((d >> 48) & 0xFF) as i32,
        bound: ((d >> 56) & 3) as u8,
        pv: (d >> 58) & 1 != 0,
    }
}

#[inline(always)]
fn entry_age(d: u64) -> u8 {
    (d >> 59) as u8
}

impl TT {
    pub fn new(mb: usize) -> TT {
        let len = (mb.max(1) * 1024 * 1024 / std::mem::size_of::<Bucket>()).max(1);
        let layout = Layout::array::<Bucket>(len).unwrap();
        // SAFETY: zeroed atomics are valid; the layout is non-zero sized.
        let buckets = unsafe { alloc_zeroed(layout) } as *mut Bucket;
        assert!(!buckets.is_null(), "failed to allocate transposition table");
        TT { buckets, len, age: AtomicU8::new(0) }
    }

    pub fn clear(&self) {
        for i in 0..self.len {
            let b = self.bucket_at(i);
            for s in &b.slots {
                s.key.store(0, Relaxed);
                s.data.store(0, Relaxed);
            }
        }
        self.age.store(0, Relaxed);
    }

    pub fn new_search(&self) {
        self.age.store((self.age.load(Relaxed) + 1) & 31, Relaxed);
    }

    #[inline(always)]
    fn bucket_at(&self, i: usize) -> &Bucket {
        // SAFETY: i < len, and the allocation lives as long as self.
        unsafe { &*self.buckets.add(i) }
    }

    #[inline(always)]
    fn bucket(&self, hash: u64) -> &Bucket {
        self.bucket_at(((hash as u128 * self.len as u128) >> 64) as usize)
    }

    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        for s in &self.bucket(hash).slots {
            let data = s.data.load(Relaxed);
            if data != 0 && s.key.load(Relaxed) ^ data == hash {
                return Some(unpack(data));
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(&self, hash: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: u8, pv: bool) {
        let bucket = self.bucket(hash);
        let age = self.age.load(Relaxed);

        let mut idx = 0;
        let mut worst = i32::MAX;
        let mut same = false;
        for (i, s) in bucket.slots.iter().enumerate() {
            let data = s.data.load(Relaxed);
            if data == 0 || s.key.load(Relaxed) ^ data == hash {
                idx = i;
                same = data != 0;
                break;
            }
            let rel_age = ((32 + age as i32 - entry_age(data) as i32) & 31) as i32;
            let value = unpack(data).depth - 8 * rel_age;
            if value < worst {
                worst = value;
                idx = i;
            }
        }

        let slot = &bucket.slots[idx];
        let mut mv = mv;
        if same {
            let old_data = slot.data.load(Relaxed);
            let old = unpack(old_data);
            if mv.is_none() {
                mv = old.mv;
            }
            if bound != BOUND_EXACT && depth + 4 + 2 * pv as i32 <= old.depth && entry_age(old_data) == age {
                return;
            }
        }
        let data = pack(mv, score, eval, depth, bound, pv, age);
        slot.key.store(hash ^ data, Relaxed);
        slot.data.store(data, Relaxed);
    }

    /// Approximate permille of the table filled with entries from the current search.
    pub fn hashfull(&self) -> usize {
        let age = self.age.load(Relaxed);
        let n = self.len.min(250);
        let mut used = 0;
        for i in 0..n {
            for s in &self.bucket_at(i).slots {
                let d = s.data.load(Relaxed);
                if d != 0 && entry_age(d) == age {
                    used += 1;
                }
            }
        }
        used * 1000 / (n * 4)
    }
}

impl Drop for TT {
    fn drop(&mut self) {
        let layout = Layout::array::<Bucket>(self.len).unwrap();
        // SAFETY: allocated in `new` with the same layout.
        unsafe { dealloc(self.buckets as *mut u8, layout) };
    }
}
