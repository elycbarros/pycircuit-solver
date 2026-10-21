/// mna_core.rs — Pure-Rust MNA matrix assembly + dense LU solver.
///
/// This module replicates the stamp rules from `circuit_solver/engine/mna.py`
/// in Rust for maximum throughput. It is called thousands of times per
/// Monte Carlo run (once per sample), so every allocation is minimised.
///
/// Stamp rules (same as Python reference):
///   Resistor  R between p, n:  G[p,p]+=g; G[n,n]+=g; G[p,n]-=g; G[n,p]-=g
///   VSource   V between p, n:  B[p,k]+=1; B[n,k]-=1; C[k,p]+=1; C[k,n]-=1; e[k]=V
///   ISource   I from n to p:   i[p]+=I; i[n]-=I

use nalgebra::{DMatrix, DVector};
use faer::sparse::*;
use faer::prelude::Solve;

use crate::types::RustCircuit;

/// Build the MNA matrix A and RHS z for the given circuit.
/// Resistor values are taken from `resistors_override` if provided
/// (used in Monte Carlo to inject perturbed values without cloning the circuit).
pub fn build_mna(
    circuit: &RustCircuit,
    resistors_override: Option<&[f64]>, // perturbed resistances in same order
) -> (DMatrix<f64>, DVector<f64>) {
    let n = circuit.size;
    let mut a = DMatrix::<f64>::zeros(n, n);
    let mut z = DVector::<f64>::zeros(n);

    // ---- Resistors --------------------------------------------------------
    for (i, r) in circuit.resistors.iter().enumerate() {
        let resistance = match resistors_override {
            Some(ov) => ov[i],
            None => r.resistance,
        };
        let g = 1.0 / resistance;
        stamp_conductance(&mut a, r.node_p_idx, r.node_n_idx, g);
    }

    // ---- Independent voltage sources --------------------------------------
    for vs in &circuit.vsources {
        let k = vs.aux_idx;
        // B column
        if let Some(p) = vs.node_p_idx {
            a[(p, k)] += 1.0;
        }
        if let Some(n) = vs.node_n_idx {
            a[(n, k)] -= 1.0;
        }
        // C row
        if let Some(p) = vs.node_p_idx {
            a[(k, p)] += 1.0;
        }
        if let Some(n) = vs.node_n_idx {
            a[(k, n)] -= 1.0;
        }
        // RHS
        z[k] = vs.voltage;
    }

    // ---- Independent current sources --------------------------------------
    for is in &circuit.isources {
        if let Some(p) = is.node_p_idx {
            z[p] += is.current;
        }
        if let Some(n) = is.node_n_idx {
            z[n] -= is.current;
        }
    }

    // ---- VCVS (E element) -------------------------------------------------
    for e in &circuit.vcvs {
        let k = e.aux_idx;
        if let Some(p) = e.node_p_idx {
            a[(p, k)] += 1.0;
            a[(k, p)] += 1.0;
        }
        if let Some(n) = e.node_n_idx {
            a[(n, k)] -= 1.0;
            a[(k, n)] -= 1.0;
        }
        if let Some(cp) = e.ctrl_p_idx {
            a[(k, cp)] -= e.gain;
        }
        if let Some(cn) = e.ctrl_n_idx {
            a[(k, cn)] += e.gain;
        }
        z[k] = 0.0;
    }

    // ---- VCCS (G element) -------------------------------------------------
    for g in &circuit.vccs {
        let gm = g.gm;
        if let (Some(p), Some(cp)) = (g.node_p_idx, g.ctrl_p_idx) {
            a[(p, cp)] -= gm;
        }
        if let (Some(p), Some(cn)) = (g.node_p_idx, g.ctrl_n_idx) {
            a[(p, cn)] += gm;
        }
        if let (Some(n), Some(cp)) = (g.node_n_idx, g.ctrl_p_idx) {
            a[(n, cp)] += gm;
        }
        if let (Some(n), Some(cn)) = (g.node_n_idx, g.ctrl_n_idx) {
            a[(n, cn)] -= gm;
        }
    }

    // ---- CCCS (F element) -------------------------------------------------
    for f in &circuit.cccs {
        let k_vctrl = f.vctrl_aux_idx;
        if let Some(p) = f.node_p_idx {
            a[(p, k_vctrl)] += f.gain;
        }
        if let Some(n) = f.node_n_idx {
            a[(n, k_vctrl)] -= f.gain;
        }
    }

    // ---- CCVS (H element) -------------------------------------------------
    for h in &circuit.ccvs {
        let k = h.aux_idx;
        let k_vctrl = h.vctrl_aux_idx;
        if let Some(p) = h.node_p_idx {
            a[(p, k)] += 1.0;
            a[(k, p)] += 1.0;
        }
        if let Some(n) = h.node_n_idx {
            a[(n, k)] -= 1.0;
            a[(k, n)] -= 1.0;
        }
        a[(k, k_vctrl)] += h.transresistance;
        z[k] = 0.0;
    }

    // ---- Op-Amp (Ideal) ---------------------------------------------------
    for op in &circuit.opamps {
        let k = op.aux_idx;
        if let Some(out) = op.out_idx {
            a[(out, k)] += 1.0;
        }
        if let Some(gnd) = op.gnd_idx {
            a[(gnd, k)] -= 1.0;
        }
        if let Some(inp) = op.in_p_idx {
            a[(k, inp)] += 1.0;
        }
        if let Some(inn) = op.in_n_idx {
            a[(k, inn)] -= 1.0;
        }
        z[k] = 0.0;
    }

    // ---- Inductors in DC (1e6 short) --------------------------------------
    for ind in &circuit.inductors_dc {
        stamp_conductance(&mut a, ind.node_p_idx, ind.node_n_idx, 1e6);
    }

    (a, z)
}

/// Stamp a two-terminal conductance into the G sub-matrix.
#[inline(always)]
fn stamp_conductance(
    a: &mut DMatrix<f64>,
    p: Option<usize>,
    n: Option<usize>,
    g: f64,
) {
    if let Some(pi) = p {
        a[(pi, pi)] += g;
    }
    if let Some(ni) = n {
        a[(ni, ni)] += g;
    }
    if let (Some(pi), Some(ni)) = (p, n) {
        a[(pi, ni)] -= g;
        a[(ni, pi)] -= g;
    }
}

/// Solve MNA and return node-voltage vector.
/// Returns None if the system is singular.
pub fn solve_mna(
    circuit: &RustCircuit,
    resistors_override: Option<&[f64]>,
) -> Option<DVector<f64>> {
    let (a, z) = build_mna(circuit, resistors_override);
    // nalgebra full-pivoting LU — numerically stable for ill-conditioned systems
    let lu = a.full_piv_lu();
    lu.solve(&z)
}

/// Stamp a two-terminal conductance into the sparse triplet list.
#[inline(always)]
fn stamp_conductance_sparse(
    triplets: &mut Vec<Triplet<usize, usize, f64>>,
    p: Option<usize>,
    n: Option<usize>,
    g: f64,
) {
    if let Some(pi) = p {
        triplets.push(Triplet::new(pi, pi, g));
    }
    if let Some(ni) = n {
        triplets.push(Triplet::new(ni, ni, g));
    }
    if let (Some(pi), Some(ni)) = (p, n) {
        triplets.push(Triplet::new(pi, ni, -g));
        triplets.push(Triplet::new(ni, pi, -g));
    }
}

/// Build the sparse MNA system in COO format and RHS vector.
pub fn build_mna_sparse(
    circuit: &RustCircuit,
    resistors_override: Option<&[f64]>,
) -> (Vec<Triplet<usize, usize, f64>>, Vec<f64>) {
    let n = circuit.size;
    let mut triplets = Vec::with_capacity(n * 4);
    let mut z = vec![0.0f64; n];

    // ---- Resistors --------------------------------------------------------
    for (i, r) in circuit.resistors.iter().enumerate() {
        let resistance = match resistors_override {
            Some(ov) => ov[i],
            None => r.resistance,
        };
        let g = 1.0 / resistance;
        stamp_conductance_sparse(&mut triplets, r.node_p_idx, r.node_n_idx, g);
    }

    // ---- Independent voltage sources --------------------------------------
    for vs in &circuit.vsources {
        let k = vs.aux_idx;
        if let Some(p) = vs.node_p_idx {
            triplets.push(Triplet::new(p, k, 1.0));
            triplets.push(Triplet::new(k, p, 1.0));
        }
        if let Some(n) = vs.node_n_idx {
            triplets.push(Triplet::new(n, k, -1.0));
            triplets.push(Triplet::new(k, n, -1.0));
        }
        z[k] = vs.voltage;
    }

    // ---- Independent current sources --------------------------------------
    for is in &circuit.isources {
        if let Some(p) = is.node_p_idx {
            z[p] += is.current;
        }
        if let Some(n) = is.node_n_idx {
            z[n] -= is.current;
        }
    }

    // ---- VCVS (E element) -------------------------------------------------
    for e in &circuit.vcvs {
        let k = e.aux_idx;
        if let Some(p) = e.node_p_idx {
            triplets.push(Triplet::new(p, k, 1.0));
            triplets.push(Triplet::new(k, p, 1.0));
        }
        if let Some(n) = e.node_n_idx {
            triplets.push(Triplet::new(n, k, -1.0));
            triplets.push(Triplet::new(k, n, -1.0));
        }
        if let Some(cp) = e.ctrl_p_idx {
            triplets.push(Triplet::new(k, cp, -e.gain));
        }
        if let Some(cn) = e.ctrl_n_idx {
            triplets.push(Triplet::new(k, cn, e.gain));
        }
        z[k] = 0.0;
    }

    // ---- VCCS (G element) -------------------------------------------------
    for g in &circuit.vccs {
        let gm = g.gm;
        if let (Some(p), Some(cp)) = (g.node_p_idx, g.ctrl_p_idx) {
            triplets.push(Triplet::new(p, cp, -gm));
        }
        if let (Some(p), Some(cn)) = (g.node_p_idx, g.ctrl_n_idx) {
            triplets.push(Triplet::new(p, cn, gm));
        }
        if let (Some(n), Some(cp)) = (g.node_n_idx, g.ctrl_p_idx) {
            triplets.push(Triplet::new(n, cp, gm));
        }
        if let (Some(n), Some(cn)) = (g.node_n_idx, g.ctrl_n_idx) {
            triplets.push(Triplet::new(n, cn, -gm));
        }
    }

    // ---- CCCS (F element) -------------------------------------------------
    for f in &circuit.cccs {
        let k_vctrl = f.vctrl_aux_idx;
        if let Some(p) = f.node_p_idx {
            triplets.push(Triplet::new(p, k_vctrl, f.gain));
        }
        if let Some(n) = f.node_n_idx {
            triplets.push(Triplet::new(n, k_vctrl, -f.gain));
        }
    }

    // ---- CCVS (H element) -------------------------------------------------
    for h in &circuit.ccvs {
        let k = h.aux_idx;
        let k_vctrl = h.vctrl_aux_idx;
        if let Some(p) = h.node_p_idx {
            triplets.push(Triplet::new(p, k, 1.0));
            triplets.push(Triplet::new(k, p, 1.0));
        }
        if let Some(n) = h.node_n_idx {
            triplets.push(Triplet::new(n, k, -1.0));
            triplets.push(Triplet::new(k, n, -1.0));
        }
        triplets.push(Triplet::new(k, k_vctrl, h.transresistance));
        z[k] = 0.0;
    }

    // ---- Op-Amp (Ideal) ---------------------------------------------------
    for op in &circuit.opamps {
        let k = op.aux_idx;
        if let Some(out) = op.out_idx {
            triplets.push(Triplet::new(out, k, 1.0));
        }
        if let Some(gnd) = op.gnd_idx {
            triplets.push(Triplet::new(gnd, k, -1.0));
        }
        if let Some(inp) = op.in_p_idx {
            triplets.push(Triplet::new(k, inp, 1.0));
        }
        if let Some(inn) = op.in_n_idx {
            triplets.push(Triplet::new(k, inn, -1.0));
        }
        z[k] = 0.0;
    }

    // ---- Inductors in DC (1e6 short) --------------------------------------
    for ind in &circuit.inductors_dc {
        stamp_conductance_sparse(&mut triplets, ind.node_p_idx, ind.node_n_idx, 1e6);
    }

    (triplets, z)
}

/// Solve sparse MNA system using Faer sparse LU decomposition.
/// Handles circuits with thousands of nodes in $O(N)$ memory and fast sparse solve.
pub fn solve_mna_sparse(
    circuit: &RustCircuit,
    resistors_override: Option<&[f64]>,
) -> Option<Vec<f64>> {
    let n = circuit.size;
    let (triplets, z) = build_mna_sparse(circuit, resistors_override);

    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).ok()?;
    let rhs = faer::col::Col::from_fn(n, |i| z[i]);

    let lu = mat.sp_lu().ok()?;
    let x = lu.solve(&rhs);

    let mut sol = vec![0.0f64; n];
    for i in 0..n {
        sol[i] = x[i];
    }
    Some(sol)
}
