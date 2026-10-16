//! Multi-threaded CPU trainer for the NNUE architecture in `nnue.rs`.
//!
//! Usage: ferrum train <data.bin> <out-dir> [epochs] [lr] [wdl] [threads]
//!
//! Loss: MSE between sigmoid(output) and a blend of sigmoid(score / 400) and the game result.

use crate::data::*;
use crate::datagen::Rng;
use crate::nnue::{output_bucket, BUCKETS, HIDDEN as H, INPUTS, QA, QB};
use std::io::Write;
use std::time::Instant;

const W1: usize = 0;
const B1: usize = W1 + INPUTS * H;
const W2: usize = B1 + H;
const B2: usize = W2 + BUCKETS * 2 * H;
const NPARAMS: usize = B2 + BUCKETS;

const EVAL_SCALE: f32 = 400.0;
const CLIP: f32 = 1.98;

pub struct Config {
    pub data: String,
    pub out_dir: String,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub final_lr: f32,
    pub wdl: f32,
    pub threads: usize,
    /// Randomly mirror training positions across the d/e file boundary (data augmentation).
    pub mirror: bool,
}

impl Config {
    fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 || self.batch == 0 || self.threads == 0 {
            return Err("epochs, batch size, and threads must be positive".into());
        }
        if !self.lr.is_finite() || self.lr <= 0.0 || !self.final_lr.is_finite() || self.final_lr <= 0.0 {
            return Err("learning rates must be finite and positive".into());
        }
        if !self.wdl.is_finite() || !(0.0..=1.0).contains(&self.wdl) {
            return Err("wdl must be between 0 and 1".into());
        }
        Ok(())
    }
}

fn load_positions(bytes: &[u8]) -> Result<Vec<PackedPos>, String> {
    if bytes.len() % RECORD_SIZE != 0 {
        return Err("training data contains a truncated record".into());
    }
    if bytes.len() / RECORD_SIZE < 2 {
        return Err("training data needs at least two positions".into());
    }
    bytes
        .chunks_exact(RECORD_SIZE)
        .enumerate()
        .map(|(index, record)| {
            let position = PackedPos::from_bytes(record);
            if !(2..=32).contains(&position.occ.count_ones()) || position.stm > 1 || position.result > 2 {
                return Err(format!("invalid training record {index}"));
            }
            let mut valid = true;
            let mut kings = [0; 2];
            position.for_each_piece(|_, color, piece_type| {
                valid &= piece_type < 6;
                if piece_type == 5 {
                    kings[color] += 1;
                }
            });
            if !valid || kings != [1, 1] {
                return Err(format!("invalid pieces in training record {index}"));
            }
            Ok(position)
        })
        .collect()
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn screlu(x: f32) -> f32 {
    let c = x.clamp(0.0, 1.0);
    c * c
}

struct Sample {
    fw: [u16; 32],
    fb: [u16; 32],
    n: usize,
    stm: usize,
    bucket: usize,
    target: f32,
}

fn decode(p: &PackedPos, wdl: f32, mirror: bool) -> Sample {
    let mut s = Sample { fw: [0; 32], fb: [0; 32], n: 0, stm: p.stm as usize, bucket: 0, target: 0.0 };
    let flip = if mirror { 7 } else { 0 };
    p.for_each_piece(|sq, c, pt| {
        let sq = sq ^ flip;
        s.fw[s.n] = (c * 384 + pt * 64 + sq) as u16;
        s.fb[s.n] = ((c ^ 1) * 384 + pt * 64 + (sq ^ 56)) as u16;
        s.n += 1;
    });
    s.bucket = output_bucket(s.n as u32);
    let white_result = p.result as f32 / 2.0;
    let (score, result) = if s.stm == 0 { (p.score as f32, white_result) } else { (-(p.score as f32), 1.0 - white_result) };
    s.target = (1.0 - wdl) * sigmoid(score / EVAL_SCALE) + wdl * result;
    s
}

/// Forward pass; returns (prediction, us accumulator, them accumulator).
#[inline(always)]
fn forward(s: &Sample, params: &[f32], aw: &mut [f32; H], ab: &mut [f32; H]) -> f32 {
    let w1 = &params[W1..B1];
    aw.copy_from_slice(&params[B1..W2]);
    ab.copy_from_slice(&params[B1..W2]);
    for k in 0..s.n {
        let rw = &w1[s.fw[k] as usize * H..][..H];
        let rb = &w1[s.fb[k] as usize * H..][..H];
        for i in 0..H {
            aw[i] += rw[i];
            ab[i] += rb[i];
        }
    }
    let (us, them) = if s.stm == 0 { (&*aw, &*ab) } else { (&*ab, &*aw) };
    let w2 = &params[W2 + s.bucket * 2 * H..][..2 * H];
    let mut out = params[B2 + s.bucket];
    for i in 0..H {
        out += w2[i] * screlu(us[i]) + w2[H + i] * screlu(them[i]);
    }
    sigmoid(out)
}

fn train_sample(s: &Sample, params: &[f32], grad: &mut [f32]) -> f32 {
    let mut aw = [0f32; H];
    let mut ab = [0f32; H];
    let pred = forward(s, params, &mut aw, &mut ab);
    let err = pred - s.target;
    let g = 2.0 * err * pred * (1.0 - pred);

    let (us, them) = if s.stm == 0 { (&aw, &ab) } else { (&ab, &aw) };
    let w2 = &params[W2 + s.bucket * 2 * H..][..2 * H];
    grad[B2 + s.bucket] += g;
    let mut d_us = [0f32; H];
    let mut d_them = [0f32; H];
    {
        let gw2 = &mut grad[W2 + s.bucket * 2 * H..][..2 * H];
        for i in 0..H {
            gw2[i] += g * screlu(us[i]);
            gw2[H + i] += g * screlu(them[i]);
            let du = if us[i] > 0.0 && us[i] < 1.0 { 2.0 * us[i] } else { 0.0 };
            let dt = if them[i] > 0.0 && them[i] < 1.0 { 2.0 * them[i] } else { 0.0 };
            d_us[i] = g * w2[i] * du;
            d_them[i] = g * w2[H + i] * dt;
        }
    }
    let (dw, db) = if s.stm == 0 { (&d_us, &d_them) } else { (&d_them, &d_us) };
    {
        let gb1 = &mut grad[B1..W2];
        for i in 0..H {
            gb1[i] += dw[i] + db[i];
        }
    }
    let gw1 = &mut grad[W1..B1];
    for k in 0..s.n {
        let rw = &mut gw1[s.fw[k] as usize * H..][..H];
        for i in 0..H {
            rw[i] += dw[i];
        }
        let rb = &mut gw1[s.fb[k] as usize * H..][..H];
        for i in 0..H {
            rb[i] += db[i];
        }
    }
    err * err
}

/// Deterministic per-epoch coin flip for mirroring a position.
#[inline(always)]
fn mirrored(p: &PackedPos, seed: u64) -> bool {
    (p.occ ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63 == 1
}

fn batch_gradient(batch: &[PackedPos], params: &[f32], grads: &mut [Vec<f32>], wdl: f32, mirror_seed: Option<u64>) -> f64 {
    let chunk = batch.len().div_ceil(grads.len());
    let active = batch.len().div_ceil(chunk);
    let grads = &mut grads[..active];
    let loss = std::thread::scope(|scope| {
        let handles: Vec<_> = grads
            .iter_mut()
            .zip(batch.chunks(chunk))
            .map(|(gradient, part)| {
                scope.spawn(move || {
                    gradient.fill(0.0);
                    part.iter()
                        .map(|position| {
                            let mirror = mirror_seed.is_some_and(|seed| mirrored(position, seed));
                            train_sample(&decode(position, wdl, mirror), params, gradient) as f64
                        })
                        .sum::<f64>()
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).sum()
    });
    let (first, rest) = grads.split_at_mut(1);
    for gradient in rest {
        for (total, value) in first[0].iter_mut().zip(gradient.iter()) {
            *total += *value;
        }
    }
    loss
}

fn validation_loss(val: &[PackedPos], params: &[f32], wdl: f32, threads: usize) -> f64 {
    let chunk = val.len().div_ceil(threads).max(1);
    let total: f64 = std::thread::scope(|s| {
        let handles: Vec<_> = val
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut aw = [0f32; H];
                    let mut ab = [0f32; H];
                    part.iter()
                        .map(|p| {
                            let smp = decode(p, wdl, false);
                            let e = forward(&smp, params, &mut aw, &mut ab) - smp.target;
                            (e * e) as f64
                        })
                        .sum::<f64>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    total / val.len().max(1) as f64
}

pub fn save_quantized(params: &[f32], path: &str) {
    let mut out: Vec<u8> = Vec::with_capacity(crate::nnue::NET_BYTES);
    let q16 = |x: f32, scale: i32| ((x * scale as f32).round() as i32).clamp(-32767, 32767) as i16;
    for &w in &params[W1..B1] {
        out.extend_from_slice(&q16(w, QA).to_le_bytes());
    }
    for &b in &params[B1..W2] {
        out.extend_from_slice(&q16(b, QA).to_le_bytes());
    }
    for &w in &params[W2..B2] {
        out.extend_from_slice(&q16(w, QB).to_le_bytes());
    }
    for i in 0..32 {
        let v = if i < BUCKETS { (params[B2 + i] * (QA * QB) as f32).round() as i32 } else { 0 };
        out.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(out.len(), crate::nnue::NET_BYTES);
    std::fs::write(path, out).expect("failed to write network");
}

pub fn run(cfg: Config) -> Result<(), String> {
    cfg.validate()?;
    let bytes = std::fs::read(&cfg.data).map_err(|error| format!("cannot read training data: {error}"))?;
    let mut positions = load_positions(&bytes)?;
    let n = positions.len();
    drop(bytes);
    // Datagen writes whole games contiguously, so holding out the tail of the file (before
    // shuffling) keeps validation games out of the training set, apart from one boundary game.
    let n_val = (n / 100).clamp(1, 200_000);
    let val = positions.split_off(n - n_val);
    let mut train = positions;
    let mut rng = Rng::new(0xC0FFEE);
    println!("loaded {} training / {} validation positions", train.len(), val.len());
    std::fs::create_dir_all(&cfg.out_dir).unwrap();

    let mut params = vec![0f32; NPARAMS];
    let l1 = 1.0 / (INPUTS as f32).sqrt();
    let l2 = 1.0 / ((2 * H) as f32).sqrt();
    for (i, p) in params.iter_mut().enumerate() {
        let u = (rng.next_u64() >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0;
        if i < B1 {
            *p = u * l1;
        } else if (W2..B2).contains(&i) {
            *p = u * l2;
        }
    }
    let mut m = vec![0f32; NPARAMS];
    let mut v = vec![0f32; NPARAMS];
    let mut grads: Vec<Vec<f32>> = (0..cfg.threads).map(|_| vec![0f32; NPARAMS]).collect();
    let (beta1, beta2, eps, decay) = (0.9f32, 0.999f32, 1e-8f32, 0.01f32);

    let start = Instant::now();
    for epoch in 0..cfg.epochs {
        let t = epoch as f32 / (cfg.epochs - 1).max(1) as f32;
        let lr = cfg.final_lr + 0.5 * (cfg.lr - cfg.final_lr) * (1.0 + (std::f32::consts::PI * t).cos());
        for i in (1..train.len()).rev() {
            train.swap(i, rng.below(i + 1));
        }
        let mut epoch_loss = 0f64;
        for batch in train.chunks(cfg.batch) {
            let mirror_seed = cfg.mirror.then(|| rng.next_u64());
            let loss = batch_gradient(batch, &params, &mut grads, cfg.wdl, mirror_seed);
            epoch_loss += loss;
            let g0 = &grads[0];
            let inv = 1.0 / batch.len() as f32;
            for i in 0..NPARAMS {
                let gi = g0[i] * inv;
                m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
                v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
                let mut p = params[i] * (1.0 - lr * decay);
                p -= lr * m[i] / (v[i].sqrt() + eps);
                params[i] = if i < B2 { p.clamp(-CLIP, CLIP) } else { p };
            }
        }
        let train_loss = epoch_loss / train.len() as f64;
        let val_loss = validation_loss(&val, &params, cfg.wdl, cfg.threads);
        println!(
            "epoch {:>3}  lr {:.6}  train {:.6}  val {:.6}  ({:.0}s)",
            epoch + 1,
            lr,
            train_loss,
            val_loss,
            start.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().ok();
        save_quantized(&params, &format!("{}/epoch{}.nnue", cfg.out_dir, epoch + 1));
    }
    let final_path = format!("{}/final.nnue", cfg.out_dir);
    save_quantized(&params, &final_path);
    check_quantization(&params, &final_path, &val[..val.len().min(8)]);
    Ok(())
}

/// Compares float and quantized integer inference on a few positions.
fn check_quantization(params: &[f32], path: &str, sample: &[PackedPos]) {
    let bytes = std::fs::read(path).unwrap();
    let net = crate::nnue::network_from_bytes(&bytes).expect("bad network size");
    let mut aw = [0f32; H];
    let mut ab = [0f32; H];
    for p in sample {
        let smp = decode(p, 0.0, false);
        let pred = forward(&smp, params, &mut aw, &mut ab);
        let float_cp = (pred / (1.0 - pred)).ln() * EVAL_SCALE;
        let board = crate::board::Board::from_fen(&p.to_fen()).unwrap();
        let acc = crate::nnue::Accumulator::refresh(&net, &board);
        let q = crate::nnue::evaluate(&net, &acc, board.stm, board.occupied().count_ones());
        println!("check: float {float_cp:>8.1}  quantized {q:>6}  data {:>6}  {}", p.score, p.to_fen());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn position() -> PackedPos {
        PackedPos::pack(&Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap(), 400)
    }

    fn config() -> Config {
        Config {
            data: String::new(), out_dir: String::new(), epochs: 1, batch: 16384,
            lr: 0.001, final_lr: 0.00001, wdl: 0.25, threads: 8, mirror: false,
        }
    }

    #[test]
    fn mirroring_matches_mirrored_position() {
        let pack = |fen: &str| PackedPos::pack(&Board::from_fen(fen).unwrap(), -35);
        let features = |s: &Sample| {
            let (mut w, mut b) = (s.fw[..s.n].to_vec(), s.fb[..s.n].to_vec());
            w.sort_unstable();
            b.sort_unstable();
            (w, b, s.stm, s.bucket, s.target)
        };
        let original = pack("r3k2r/pb1n1ppp/1p6/2pP4/8/2N2N2/PP3PPP/R2QK2R b KQkq - 1 12");
        let mirrored = pack("r2k3r/ppp1n1bp/6p1/4Pp2/8/2N2N2/PPP3PP/R2KQ2R b - - 1 12");
        assert_eq!(features(&decode(&original, 0.25, true)), features(&decode(&mirrored, 0.25, false)));
        assert_ne!(features(&decode(&original, 0.25, false)), features(&decode(&mirrored, 0.25, false)));
    }

    #[test]
    fn rejects_invalid_config() {
        let mut cfg = config();
        assert!(cfg.validate().is_ok());
        cfg.threads = 0;
        assert!(cfg.validate().is_err());
        cfg = config();
        cfg.batch = 0;
        assert!(cfg.validate().is_err());
        cfg = config();
        cfg.epochs = 0;
        assert!(cfg.validate().is_err());
        for value in [f32::NAN, f32::INFINITY, 0.0, -0.1] {
            cfg = config();
            cfg.lr = value;
            assert!(cfg.validate().is_err());
            cfg = config();
            cfg.final_lr = value;
            assert!(cfg.validate().is_err());
        }
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            cfg = config();
            cfg.wdl = value;
            assert!(cfg.validate().is_err());
        }
    }

    #[test]
    fn rejects_malformed_data() {
        let valid = position().to_bytes().repeat(2);
        assert_eq!(load_positions(&valid).unwrap().len(), 2);
        assert!(load_positions(&[]).is_err());
        assert!(load_positions(&valid[..RECORD_SIZE]).is_err());
        assert!(load_positions(&valid[..valid.len() - 1]).is_err());
        for (offset, value) in [(26, 3), (27, 2), (8, 7), (8, 0)] {
            let mut invalid = valid.clone();
            invalid[offset] = value;
            assert!(load_positions(&invalid).is_err());
        }
        let mut invalid = valid;
        invalid[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(load_positions(&invalid).is_err());
    }

    #[test]
    fn partial_batch_ignores_inactive_gradients() {
        let params = vec![0.1; NPARAMS];
        let mut grads = vec![vec![0.0; NPARAMS]; 8];
        batch_gradient(&[position(); 16], &params, &mut grads, 0.25, None);
        let mut expected = vec![0.0; NPARAMS];
        let expected_loss = train_sample(&decode(&position(), 0.25, false), &params, &mut expected);
        let actual_loss = batch_gradient(&[position()], &params, &mut grads, 0.25, None);
        assert_eq!(actual_loss, expected_loss as f64);
        assert_eq!(grads[0], expected);
    }

    #[test]
    fn trains_dataset_smaller_than_batch() {
        let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let directory = std::env::temp_dir().join(format!("ferrum-trainer-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let data = directory.join("data.bin");
        std::fs::write(&data, position().to_bytes().repeat(4)).unwrap();
        for batch in [16384, 2] {
            let mut cfg = config();
            cfg.batch = batch;
            cfg.data = data.to_str().unwrap().into();
            cfg.out_dir = directory.to_str().unwrap().into();
            run(cfg).unwrap();
            let bytes = std::fs::read(directory.join("final.nnue")).unwrap();
            let net = crate::nnue::network_from_bytes(&bytes).unwrap();
            assert_ne!(net.b2[output_bucket(3)], 0);
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
