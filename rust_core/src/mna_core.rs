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
