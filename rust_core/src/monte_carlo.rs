/// monte_carlo.rs — Parallel Monte Carlo simulation engine in Rust.
///
/// Strategy
/// --------
/// 1. Receive the pre-resolved circuit topology from Python (node indices,
///    nominal values, tolerances) as a `RustCircuit`.
/// 2. Use `rayon::par_iter` to scatter N independent samples across all
///    available CPU cores simultaneously (no GIL, true parallelism).
/// 3. Each sample thread draws perturbed resistances from a seeded PRNG
///    (ChaCha8 for speed + reproducibility), assembles and solves MNA,
///    and records the node-voltage vector.
/// 4. Aggregate results (mean, std, min, max, p5, p95) and return to Python
///    as raw Vec<f64> data that numpy wraps without copying.
///
/// Performance expectations vs pure Python + NumPy:
///   n_samples = 10_000 :  Python ~2 s  →  Rust parallel ~0.06 s  (≈33×)
///   n_samples = 100_000:  Python ~20 s →  Rust parallel ~0.5 s   (≈40×)

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rayon::prelude::*;

use crate::types::RustCircuit;
use crate::mna_core::solve_mna;

/// Raw output from a Monte Carlo run — flat arrays for each statistical metric.
/// Each array has length `n_nodes` (non-ground nodes in circuit order).
#[allow(dead_code)]
pub struct MonteCarloOutput {
    pub node_count: usize,
    pub n_samples: usize,
    pub n_failed: usize,
    /// All valid samples, row-major: shape (n_valid, n_nodes).
    pub samples_flat: Vec<f64>,
    pub mean: Vec<f64>,
    pub std_dev: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub p5: Vec<f64>,
    pub p95: Vec<f64>,
}

/// Run Monte Carlo analysis.
///
/// Parameters
/// ----------
/// circuit       : pre-resolved circuit topology
/// n_samples     : total number of random trials
/// seed          : base RNG seed (each worker thread gets seed + thread_index)
/// override_tol  : if Some(t), override all resistor tolerances with t
pub fn run_monte_carlo(
    circuit: &RustCircuit,
    n_samples: usize,
    seed: u64,
    override_tol: Option<f64>,
) -> MonteCarloOutput {
    let n_nodes = circuit.n_nodes;
    let _n_resistors = circuit.resistors.len();

    // Tolerances vector — built once, shared read-only across threads.
    let tolerances: Vec<f64> = circuit.resistors.iter().map(|r| {
        override_tol.unwrap_or(r.tolerance)
    }).collect();

    let nominal: Vec<f64> = circuit.resistors.iter().map(|r| r.resistance).collect();

    // ---- Parallel sampling ------------------------------------------------
    // Each sample is independent: use (seed, sample_index) to seed a per-sample
    // ChaCha8 PRNG → fully reproducible AND parallelisable.
    let results: Vec<Option<Vec<f64>>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            // Unique, deterministic seed per sample
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));

            // Perturb resistances
            let perturbed: Vec<f64> = nominal.iter().zip(tolerances.iter()).map(|(&r, &tol)| {
                let delta: f64 = rng.gen_range(-tol..=tol);
                let r_actual = r * (1.0 + delta);
                r_actual.max(1e-9) // physical minimum
            }).collect();

            // Solve MNA with perturbed values
            match solve_mna(circuit, Some(&perturbed)) {
                Some(x) => {
                    // Extract only the first n_nodes entries (node voltages)
                    let v: Vec<f64> = (0..n_nodes).map(|j| x[j]).collect();
                    Some(v)
                }
                None => None, // singular → count as failure
            }
        })
        .collect();

    // ---- Aggregate results ------------------------------------------------
    let mut samples_flat: Vec<f64> = Vec::new();
    let mut n_failed = 0usize;

    for result in &results {
        match result {
            Some(v) => samples_flat.extend_from_slice(v),
            None => n_failed += 1,
        }
    }

    let n_valid = n_samples - n_failed;

    if n_valid == 0 {
        // Return empty stats
        let z = vec![0.0f64; n_nodes];
        return MonteCarloOutput {
            node_count: n_nodes,
            n_samples,
            n_failed,
            samples_flat: vec![],
            mean: z.clone(), std_dev: z.clone(), min: z.clone(),
            max: z.clone(), p5: z.clone(), p95: z.clone(),
        };
    }

    // Reshape samples_flat into (n_valid, n_nodes) conceptually
    // Compute per-node statistics
    let mut mean = vec![0.0f64; n_nodes];
    let mut min_v = vec![f64::INFINITY; n_nodes];
    let mut max_v = vec![f64::NEG_INFINITY; n_nodes];

    for s in 0..n_valid {
        for j in 0..n_nodes {
            let v = samples_flat[s * n_nodes + j];
            mean[j] += v;
            if v < min_v[j] { min_v[j] = v; }
            if v > max_v[j] { max_v[j] = v; }
        }
    }
    for j in 0..n_nodes {
        mean[j] /= n_valid as f64;
    }

    let mut variance = vec![0.0f64; n_nodes];
    for s in 0..n_valid {
        for j in 0..n_nodes {
            let diff = samples_flat[s * n_nodes + j] - mean[j];
            variance[j] += diff * diff;
        }
    }
    let std_dev: Vec<f64> = variance.iter().map(|&v| (v / n_valid as f64).sqrt()).collect();

    // Percentiles — collect per-node sorted values
    let p5 = percentile_columns(&samples_flat, n_valid, n_nodes, 5);
    let p95 = percentile_columns(&samples_flat, n_valid, n_nodes, 95);

    MonteCarloOutput {
        node_count: n_nodes,
        n_samples,
        n_failed,
        samples_flat,
        mean,
        std_dev,
        min: min_v,
        max: max_v,
        p5,
        p95,
    }
}

/// Compute a percentile for each column of a row-major matrix.
fn percentile_columns(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    pct: usize,
) -> Vec<f64> {
    (0..n_cols).map(|j| {
        let mut col: Vec<f64> = (0..n_rows).map(|i| data[i * n_cols + j]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((pct as f64 / 100.0) * (n_rows as f64 - 1.0)).round() as usize;
        col[idx.min(n_rows - 1)]
    }).collect()
}
