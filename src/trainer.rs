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

fn decode(p: &PackedPos, wdl: f32) -> Sample {
    let mut s = Sample { fw: [0; 32], fb: [0; 32], n: 0, stm: p.stm as usize, bucket: 0, target: 0.0 };
    p.for_each_piece(|sq, c, pt| {
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
                            let smp = decode(p, wdl);
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

pub fn run(cfg: Config) {
    let bytes = std::fs::read(&cfg.data).expect("cannot read training data");
    let n = bytes.len() / RECORD_SIZE;
    let mut positions: Vec<PackedPos> = (0..n).map(|i| PackedPos::from_bytes(&bytes[i * RECORD_SIZE..])).collect();
    drop(bytes);
    let mut rng = Rng::new(0xC0FFEE);
    for i in (1..positions.len()).rev() {
        positions.swap(i, rng.below(i + 1));
    }
    let n_val = (n / 100).min(200_000);
    let val = positions.split_off(n - n_val);
    let mut train = positions;
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

    let batches = train.len() / cfg.batch;
    let start = Instant::now();
    for epoch in 0..cfg.epochs {
        let t = epoch as f32 / (cfg.epochs - 1).max(1) as f32;
        let lr = cfg.final_lr + 0.5 * (cfg.lr - cfg.final_lr) * (1.0 + (std::f32::consts::PI * t).cos());
        for i in (1..train.len()).rev() {
            train.swap(i, rng.below(i + 1));
        }
        let mut epoch_loss = 0f64;
        for bi in 0..batches {
            let batch = &train[bi * cfg.batch..(bi + 1) * cfg.batch];
            let chunk = batch.len().div_ceil(cfg.threads);
            let params_ref = &params;
            let wdl = cfg.wdl;
            let loss: f64 = std::thread::scope(|s| {
                let handles: Vec<_> = grads
                    .iter_mut()
                    .zip(batch.chunks(chunk))
                    .map(|(g, part)| {
                        s.spawn(move || {
                            g.fill(0.0);
                            part.iter().map(|p| train_sample(&decode(p, wdl), params_ref, g) as f64).sum::<f64>()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).sum()
            });
            epoch_loss += loss;

            let (first, rest) = grads.split_at_mut(1);
            let g0 = &mut first[0];
            for g in rest.iter() {
                for (a, b) in g0.iter_mut().zip(g.iter()) {
                    *a += *b;
                }
            }
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
        let train_loss = epoch_loss / (batches * cfg.batch) as f64;
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
}

/// Compares float and quantized integer inference on a few positions.
fn check_quantization(params: &[f32], path: &str, sample: &[PackedPos]) {
    let bytes = std::fs::read(path).unwrap();
    let net = crate::nnue::network_from_bytes(&bytes).expect("bad network size");
    let mut aw = [0f32; H];
    let mut ab = [0f32; H];
    for p in sample {
        let smp = decode(p, 0.0);
        let pred = forward(&smp, params, &mut aw, &mut ab);
        let float_cp = (pred / (1.0 - pred)).ln() * EVAL_SCALE;
        let board = crate::board::Board::from_fen(&p.to_fen()).unwrap();
        let acc = crate::nnue::Accumulator::refresh(&net, &board);
        let q = crate::nnue::evaluate(&net, &acc, board.stm, board.occupied().count_ones());
        println!("check: float {float_cp:>8.1}  quantized {q:>6}  data {:>6}  {}", p.score, p.to_fen());
    }
}
