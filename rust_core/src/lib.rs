/// lib.rs — PyO3 entry point: exposes Rust functions to Python.
///
/// Module: `pycircuit_core`
/// Python import: `from pycircuit_core import monte_carlo_run`
///
/// Architecture
/// ------------
///   Python side:  resolves node names → integer indices,
///                 builds RustCircuit descriptor,
///                 calls `monte_carlo_run(...)`,
///                 wraps returned Vec<f64> arrays into numpy arrays.
///
///   Rust side:    pure numeric work — MNA assembly, LU solve, rayon parallelism.

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2};

mod types;
mod mna_core;
mod monte_carlo;
mod transient;

use crate::types::{
    RustCircuit, RustResistorStamped, RustVSource, RustISource,
};
use crate::monte_carlo::run_monte_carlo;

// ---------------------------------------------------------------------------
// Python-exposed data types
// ---------------------------------------------------------------------------

/// Resistor descriptor passed from Python → Rust.
#[pyclass]
#[derive(Clone)]
pub struct PyResistor {
    #[pyo3(get, set)] pub resistance: f64,
    #[pyo3(get, set)] pub tolerance: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,  // -1 = ground
    #[pyo3(get, set)] pub node_n_idx: i64,
}

#[pymethods]
impl PyResistor {
    #[new]
    fn new(resistance: f64, tolerance: f64, node_p_idx: i64, node_n_idx: i64) -> Self {
        PyResistor { resistance, tolerance, node_p_idx, node_n_idx }
    }
}

/// Voltage source descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyVSource {
    #[pyo3(get, set)] pub voltage: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
    #[pyo3(get, set)] pub aux_idx: usize,
}

#[pymethods]
impl PyVSource {
    #[new]
    fn new(voltage: f64, node_p_idx: i64, node_n_idx: i64, aux_idx: usize) -> Self {
        PyVSource { voltage, node_p_idx, node_n_idx, aux_idx }
    }
}

/// Current source descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyISource {
    #[pyo3(get, set)] pub current: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
}

#[pymethods]
impl PyISource {
    #[new]
    fn new(current: f64, node_p_idx: i64, node_n_idx: i64) -> Self {
        PyISource { current, node_p_idx, node_n_idx }
    }
}

/// Voltage-controlled voltage source descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyVCVS {
    #[pyo3(get, set)] pub gain: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
    #[pyo3(get, set)] pub ctrl_p_idx: i64,
    #[pyo3(get, set)] pub ctrl_n_idx: i64,
    #[pyo3(get, set)] pub aux_idx: usize,
}

#[pymethods]
impl PyVCVS {
    #[new]
    fn new(gain: f64, node_p_idx: i64, node_n_idx: i64, ctrl_p_idx: i64, ctrl_n_idx: i64, aux_idx: usize) -> Self {
        PyVCVS { gain, node_p_idx, node_n_idx, ctrl_p_idx, ctrl_n_idx, aux_idx }
    }
}

/// Voltage-controlled current source descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyVCCS {
    #[pyo3(get, set)] pub gm: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
    #[pyo3(get, set)] pub ctrl_p_idx: i64,
    #[pyo3(get, set)] pub ctrl_n_idx: i64,
}

#[pymethods]
impl PyVCCS {
    #[new]
    fn new(gm: f64, node_p_idx: i64, node_n_idx: i64, ctrl_p_idx: i64, ctrl_n_idx: i64) -> Self {
        PyVCCS { gm, node_p_idx, node_n_idx, ctrl_p_idx, ctrl_n_idx }
    }
}

/// Current-controlled current source descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyCCCS {
    #[pyo3(get, set)] pub gain: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
    #[pyo3(get, set)] pub vctrl_aux_idx: usize,
}

#[pymethods]
impl PyCCCS {
    #[new]
    fn new(gain: f64, node_p_idx: i64, node_n_idx: i64, vctrl_aux_idx: usize) -> Self {
        PyCCCS { gain, node_p_idx, node_n_idx, vctrl_aux_idx }
    }
}

/// Current-controlled voltage source descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyCCVS {
    #[pyo3(get, set)] pub transresistance: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
    #[pyo3(get, set)] pub aux_idx: usize,
    #[pyo3(get, set)] pub vctrl_aux_idx: usize,
}

#[pymethods]
impl PyCCVS {
    #[new]
    fn new(transresistance: f64, node_p_idx: i64, node_n_idx: i64, aux_idx: usize, vctrl_aux_idx: usize) -> Self {
        PyCCVS { transresistance, node_p_idx, node_n_idx, aux_idx, vctrl_aux_idx }
    }
}

/// Ideal Op-Amp descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyOpAmp {
    #[pyo3(get, set)] pub in_p_idx: i64,
    #[pyo3(get, set)] pub in_n_idx: i64,
    #[pyo3(get, set)] pub out_idx: i64,
    #[pyo3(get, set)] pub gnd_idx: i64,
    #[pyo3(get, set)] pub aux_idx: usize,
}

#[pymethods]
impl PyOpAmp {
    #[new]
    fn new(in_p_idx: i64, in_n_idx: i64, out_idx: i64, gnd_idx: i64, aux_idx: usize) -> Self {
        PyOpAmp { in_p_idx, in_n_idx, out_idx, gnd_idx, aux_idx }
    }
}

/// Inductor (DC short) descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyInductorDC {
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
}

#[pymethods]
impl PyInductorDC {
    #[new]
    fn new(node_p_idx: i64, node_n_idx: i64) -> Self {
        PyInductorDC { node_p_idx, node_n_idx }
    }
}

/// Capacitor descriptor for transient analysis.
#[pyclass]
#[derive(Clone)]
pub struct PyCapacitor {
    #[pyo3(get, set)] pub capacitance: f64,
    #[pyo3(get, set)] pub ic: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
}

#[pymethods]
impl PyCapacitor {
    #[new]
    fn new(capacitance: f64, ic: f64, node_p_idx: i64, node_n_idx: i64) -> Self {
        PyCapacitor { capacitance, ic, node_p_idx, node_n_idx }
    }
}

/// Inductor descriptor for transient analysis.
#[pyclass]
#[derive(Clone)]
pub struct PyInductor {
    #[pyo3(get, set)] pub inductance: f64,
    #[pyo3(get, set)] pub ic: f64,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
}

#[pymethods]
impl PyInductor {
    #[new]
    fn new(inductance: f64, ic: f64, node_p_idx: i64, node_n_idx: i64) -> Self {
        PyInductor { inductance, ic, node_p_idx, node_n_idx }
    }
}

/// Coupled inductors pair descriptor.
#[pyclass]
#[derive(Clone)]
pub struct PyCoupledPair {
    #[pyo3(get, set)] pub ic1: f64,
    #[pyo3(get, set)] pub ic2: f64,
    #[pyo3(get, set)] pub l1: f64,
    #[pyo3(get, set)] pub l2: f64,
    #[pyo3(get, set)] pub k: f64,
    #[pyo3(get, set)] pub p1_idx: i64,
    #[pyo3(get, set)] pub n1_idx: i64,
    #[pyo3(get, set)] pub p2_idx: i64,
    #[pyo3(get, set)] pub n2_idx: i64,
}

#[pymethods]
impl PyCoupledPair {
    #[new]
    #[pyo3(signature = (l1, l2, k, p1_idx, n1_idx, p2_idx, n2_idx, ic1=0.0, ic2=0.0))]
    fn new(l1: f64, l2: f64, k: f64, p1_idx: i64, n1_idx: i64, p2_idx: i64, n2_idx: i64, ic1: f64, ic2: f64) -> Self {
        PyCoupledPair { l1, l2, k, p1_idx, n1_idx, p2_idx, n2_idx, ic1, ic2 }
    }
}

/// Source with waveform descriptor for transient analysis.
#[pyclass]
#[derive(Clone)]
pub struct PySourceTran {
    #[pyo3(get, set)] pub kind: u8,
    #[pyo3(get, set)] pub params: Vec<f64>,
    #[pyo3(get, set)] pub node_p_idx: i64,
    #[pyo3(get, set)] pub node_n_idx: i64,
    #[pyo3(get, set)] pub aux_idx: usize,
}

#[pymethods]
impl PySourceTran {
    #[new]
    fn new(kind: u8, params: Vec<f64>, node_p_idx: i64, node_n_idx: i64, aux_idx: usize) -> Self {
        PySourceTran { kind, params, node_p_idx, node_n_idx, aux_idx }
    }
}

fn to_rust_waveform(kind: u8, p: &[f64]) -> crate::types::RustWaveform {
    match kind {
        0 => crate::types::RustWaveform::DC(p.get(0).copied().unwrap_or(0.0)),
        1 => crate::types::RustWaveform::Step {
            v1: p.get(0).copied().unwrap_or(0.0),
            v2: p.get(1).copied().unwrap_or(1.0),
            t_step: p.get(2).copied().unwrap_or(0.0),
        },
        2 => crate::types::RustWaveform::Pulse {
            v1: p.get(0).copied().unwrap_or(0.0),
            v2: p.get(1).copied().unwrap_or(1.0),
            td: p.get(2).copied().unwrap_or(0.0),
            tr: p.get(3).copied().unwrap_or(1e-9),
            tf: p.get(4).copied().unwrap_or(1e-9),
            pw: p.get(5).copied().unwrap_or(1.0),
            per: p.get(6).copied().unwrap_or(2.0),
        },
        3 => crate::types::RustWaveform::Sine {
            vo: p.get(0).copied().unwrap_or(0.0),
            va: p.get(1).copied().unwrap_or(1.0),
            freq: p.get(2).copied().unwrap_or(60.0),
            td: p.get(3).copied().unwrap_or(0.0),
            theta: p.get(4).copied().unwrap_or(0.0),
        },
        _ => crate::types::RustWaveform::DC(0.0),
    }
}

// ---------------------------------------------------------------------------
// Helper conversions
// ---------------------------------------------------------------------------

fn opt_idx(i: i64) -> Option<usize> {
    if i < 0 { None } else { Some(i as usize) }
}

fn to_rust_circuit(
    n_nodes: usize,
    size: usize,
    resistors: &[PyResistor],
    vsources: &[PyVSource],
    isources: &[PyISource],
    vcvs: Option<&[PyVCVS]>,
    vccs: Option<&[PyVCCS]>,
    cccs: Option<&[PyCCCS]>,
    ccvs: Option<&[PyCCVS]>,
    opamps: Option<&[PyOpAmp]>,
    inductors_dc: Option<&[PyInductorDC]>,
) -> RustCircuit {
    RustCircuit {
        n_nodes,
        size,
        resistors: resistors.iter().map(|r| RustResistorStamped {
            resistance: r.resistance,
            tolerance: r.tolerance,
            node_p_idx: opt_idx(r.node_p_idx),
            node_n_idx: opt_idx(r.node_n_idx),
        }).collect(),
        vsources: vsources.iter().map(|v| RustVSource {
            voltage: v.voltage,
            node_p_idx: opt_idx(v.node_p_idx),
            node_n_idx: opt_idx(v.node_n_idx),
            aux_idx: v.aux_idx,
        }).collect(),
        isources: isources.iter().map(|i| RustISource {
            current: i.current,
            node_p_idx: opt_idx(i.node_p_idx),
            node_n_idx: opt_idx(i.node_n_idx),
        }).collect(),
        vcvs: vcvs.unwrap_or(&[]).iter().map(|e| crate::types::RustVCVS {
            gain: e.gain,
            node_p_idx: opt_idx(e.node_p_idx),
            node_n_idx: opt_idx(e.node_n_idx),
            ctrl_p_idx: opt_idx(e.ctrl_p_idx),
            ctrl_n_idx: opt_idx(e.ctrl_n_idx),
            aux_idx: e.aux_idx,
        }).collect(),
        vccs: vccs.unwrap_or(&[]).iter().map(|g| crate::types::RustVCCS {
            gm: g.gm,
            node_p_idx: opt_idx(g.node_p_idx),
            node_n_idx: opt_idx(g.node_n_idx),
            ctrl_p_idx: opt_idx(g.ctrl_p_idx),
            ctrl_n_idx: opt_idx(g.ctrl_n_idx),
        }).collect(),
        cccs: cccs.unwrap_or(&[]).iter().map(|f| crate::types::RustCCCS {
            gain: f.gain,
            node_p_idx: opt_idx(f.node_p_idx),
            node_n_idx: opt_idx(f.node_n_idx),
            vctrl_aux_idx: f.vctrl_aux_idx,
        }).collect(),
        ccvs: ccvs.unwrap_or(&[]).iter().map(|h| crate::types::RustCCVS {
            transresistance: h.transresistance,
            node_p_idx: opt_idx(h.node_p_idx),
            node_n_idx: opt_idx(h.node_n_idx),
            aux_idx: h.aux_idx,
            vctrl_aux_idx: h.vctrl_aux_idx,
        }).collect(),
        opamps: opamps.unwrap_or(&[]).iter().map(|op| crate::types::RustOpAmp {
            in_p_idx: opt_idx(op.in_p_idx),
            in_n_idx: opt_idx(op.in_n_idx),
            out_idx: opt_idx(op.out_idx),
            gnd_idx: opt_idx(op.gnd_idx),
            aux_idx: op.aux_idx,
        }).collect(),
        inductors_dc: inductors_dc.unwrap_or(&[]).iter().map(|ind| crate::types::RustInductorDC {
            node_p_idx: opt_idx(ind.node_p_idx),
            node_n_idx: opt_idx(ind.node_n_idx),
        }).collect(),
    }
}

// ---------------------------------------------------------------------------
// Exported Python function: monte_carlo_run
// ---------------------------------------------------------------------------

/// Run parallel Monte Carlo analysis in Rust.
///
/// Parameters (from Python)
/// -------------------------
/// n_nodes     : number of non-ground nodes
/// size        : total MNA system size (n_nodes + n_vsources)
/// resistors   : list of PyResistor
/// vsources    : list of PyVSource
/// isources    : list of PyISource
/// n_samples   : number of random trials
/// seed        : base RNG seed (u64)
/// override_tol: float or None — overrides all resistor tolerances
///
/// Returns
/// -------
/// Tuple of 7 numpy arrays (each length n_nodes):
///   (mean, std, min, max, p5, p95, samples_2d)
///   where samples_2d has shape (n_valid, n_nodes).
#[pyfunction]
#[pyo3(signature = (n_nodes, size, resistors, vsources, isources, n_samples, seed, override_tol=None, vcvs=None, vccs=None, cccs=None, ccvs=None, opamps=None, inductors_dc=None))]
fn monte_carlo_run<'py>(
    py: Python<'py>,
    n_nodes: usize,
    size: usize,
    resistors: Vec<PyResistor>,
    vsources: Vec<PyVSource>,
    isources: Vec<PyISource>,
    n_samples: usize,
    seed: u64,
    override_tol: Option<f64>,
    vcvs: Option<Vec<PyVCVS>>,
    vccs: Option<Vec<PyVCCS>>,
    cccs: Option<Vec<PyCCCS>>,
    ccvs: Option<Vec<PyCCVS>>,
    opamps: Option<Vec<PyOpAmp>>,
    inductors_dc: Option<Vec<PyInductorDC>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,  // mean
    Bound<'py, PyArray1<f64>>,  // std
    Bound<'py, PyArray1<f64>>,  // min
    Bound<'py, PyArray1<f64>>,  // max
    Bound<'py, PyArray1<f64>>,  // p5
    Bound<'py, PyArray1<f64>>,  // p95
    Bound<'py, PyArray2<f64>>,  // samples (n_valid, n_nodes)
    usize,                       // n_failed
)> {
    let circuit = to_rust_circuit(
        n_nodes,
        size,
        &resistors,
        &vsources,
        &isources,
        vcvs.as_deref(),
        vccs.as_deref(),
        cccs.as_deref(),
        ccvs.as_deref(),
        opamps.as_deref(),
        inductors_dc.as_deref(),
    );

    let out = run_monte_carlo(&circuit, n_samples, seed, override_tol);

    let n_valid = n_samples - out.n_failed;

    // Reshape flat samples into 2D array
    let samples_2d = if n_valid > 0 && n_nodes > 0 {
        let arr = ndarray::Array2::from_shape_vec(
            (n_valid, n_nodes),
            out.samples_flat,
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        arr.into_pyarray(py)
    } else {
        ndarray::Array2::<f64>::zeros((0, n_nodes)).into_pyarray(py)
    };

    Ok((
        out.mean.into_pyarray(py),
        out.std_dev.into_pyarray(py),
        out.min.into_pyarray(py),
        out.max.into_pyarray(py),
        out.p5.into_pyarray(py),
        out.p95.into_pyarray(py),
        samples_2d,
        out.n_failed,
    ))
}

// ---------------------------------------------------------------------------
// Exported Python function: solve_mna_single
// ---------------------------------------------------------------------------

/// Solve a single MNA system and return node voltages as a numpy array.
/// Useful for benchmarking the Rust solver vs Python/NumPy.
#[pyfunction]
#[pyo3(signature = (n_nodes, size, resistors, vsources, isources, vcvs=None, vccs=None, cccs=None, ccvs=None, opamps=None, inductors_dc=None))]
fn solve_mna_single<'py>(
    py: Python<'py>,
    n_nodes: usize,
    size: usize,
    resistors: Vec<PyResistor>,
    vsources: Vec<PyVSource>,
    isources: Vec<PyISource>,
    vcvs: Option<Vec<PyVCVS>>,
    vccs: Option<Vec<PyVCCS>>,
    cccs: Option<Vec<PyCCCS>>,
    ccvs: Option<Vec<PyCCVS>>,
    opamps: Option<Vec<PyOpAmp>>,
    inductors_dc: Option<Vec<PyInductorDC>>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let circuit = to_rust_circuit(
        n_nodes,
        size,
        &resistors,
        &vsources,
        &isources,
        vcvs.as_deref(),
        vccs.as_deref(),
        cccs.as_deref(),
        ccvs.as_deref(),
        opamps.as_deref(),
        inductors_dc.as_deref(),
    );
    match mna_core::solve_mna(&circuit, None) {
        Some(x) => {
            let v: Vec<f64> = (0..n_nodes).map(|i| x[i]).collect();
            Ok(Some(v.into_pyarray(py)))
        }
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Exported Python function: solve_mna_sparse
// ---------------------------------------------------------------------------

/// Solve a single sparse MNA system using Faer sparse LU and return node voltages as a numpy array.
/// Highly optimized for large circuits (> 1,000 nodes).
#[pyfunction]
#[pyo3(signature = (n_nodes, size, resistors, vsources, isources, vcvs=None, vccs=None, cccs=None, ccvs=None, opamps=None, inductors_dc=None))]
fn solve_mna_sparse<'py>(
    py: Python<'py>,
    n_nodes: usize,
    size: usize,
    resistors: Vec<PyResistor>,
    vsources: Vec<PyVSource>,
    isources: Vec<PyISource>,
    vcvs: Option<Vec<PyVCVS>>,
    vccs: Option<Vec<PyVCCS>>,
    cccs: Option<Vec<PyCCCS>>,
    ccvs: Option<Vec<PyCCVS>>,
    opamps: Option<Vec<PyOpAmp>>,
    inductors_dc: Option<Vec<PyInductorDC>>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let circuit = to_rust_circuit(
        n_nodes,
        size,
        &resistors,
        &vsources,
        &isources,
        vcvs.as_deref(),
        vccs.as_deref(),
        cccs.as_deref(),
        ccvs.as_deref(),
        opamps.as_deref(),
        inductors_dc.as_deref(),
    );
    match mna_core::solve_mna_sparse(&circuit, None) {
        Some(x) => {
            let v: Vec<f64> = (0..size).map(|i| x[i]).collect();
            Ok(Some(v.into_pyarray(py)))
        }
        None => Ok(None),
    }
}

fn to_rust_transient_circuit(
    n_nodes: usize,
    size: usize,
    resistors: &[PyResistor],
    vsources: &[PySourceTran],
    isources: &[PySourceTran],
    capacitors: &[PyCapacitor],
    inductors: &[PyInductor],
    coupled: Option<&[PyCoupledPair]>,
    vcvs: Option<&[PyVCVS]>,
    vccs: Option<&[PyVCCS]>,
    cccs: Option<&[PyCCCS]>,
    ccvs: Option<&[PyCCVS]>,
    opamps: Option<&[PyOpAmp]>,
) -> crate::types::RustTransientCircuit {
    crate::types::RustTransientCircuit {
        n_nodes,
        size,
        resistors: resistors.iter().map(|r| RustResistorStamped {
            resistance: r.resistance,
            tolerance: r.tolerance,
            node_p_idx: opt_idx(r.node_p_idx),
            node_n_idx: opt_idx(r.node_n_idx),
        }).collect(),
        vsources: vsources.iter().map(|v| crate::types::RustVSourceTran {
            waveform: to_rust_waveform(v.kind, &v.params),
            node_p_idx: opt_idx(v.node_p_idx),
            node_n_idx: opt_idx(v.node_n_idx),
            aux_idx: v.aux_idx,
        }).collect(),
        isources: isources.iter().map(|i| crate::types::RustISourceTran {
            waveform: to_rust_waveform(i.kind, &i.params),
            node_p_idx: opt_idx(i.node_p_idx),
            node_n_idx: opt_idx(i.node_n_idx),
        }).collect(),
        vcvs: vcvs.unwrap_or(&[]).iter().map(|e| crate::types::RustVCVS {
            gain: e.gain,
            node_p_idx: opt_idx(e.node_p_idx),
            node_n_idx: opt_idx(e.node_n_idx),
            ctrl_p_idx: opt_idx(e.ctrl_p_idx),
            ctrl_n_idx: opt_idx(e.ctrl_n_idx),
            aux_idx: e.aux_idx,
        }).collect(),
        vccs: vccs.unwrap_or(&[]).iter().map(|g| crate::types::RustVCCS {
            gm: g.gm,
            node_p_idx: opt_idx(g.node_p_idx),
            node_n_idx: opt_idx(g.node_n_idx),
            ctrl_p_idx: opt_idx(g.ctrl_p_idx),
            ctrl_n_idx: opt_idx(g.ctrl_n_idx),
        }).collect(),
        cccs: cccs.unwrap_or(&[]).iter().map(|f| crate::types::RustCCCS {
            gain: f.gain,
            node_p_idx: opt_idx(f.node_p_idx),
            node_n_idx: opt_idx(f.node_n_idx),
            vctrl_aux_idx: f.vctrl_aux_idx,
        }).collect(),
        ccvs: ccvs.unwrap_or(&[]).iter().map(|h| crate::types::RustCCVS {
            transresistance: h.transresistance,
            node_p_idx: opt_idx(h.node_p_idx),
            node_n_idx: opt_idx(h.node_n_idx),
            aux_idx: h.aux_idx,
            vctrl_aux_idx: h.vctrl_aux_idx,
        }).collect(),
        opamps: opamps.unwrap_or(&[]).iter().map(|op| crate::types::RustOpAmp {
            in_p_idx: opt_idx(op.in_p_idx),
            in_n_idx: opt_idx(op.in_n_idx),
            out_idx: opt_idx(op.out_idx),
            gnd_idx: opt_idx(op.gnd_idx),
            aux_idx: op.aux_idx,
        }).collect(),
        capacitors: capacitors.iter().map(|c| crate::types::RustCapacitor {
            capacitance: c.capacitance,
            ic: c.ic,
            node_p_idx: opt_idx(c.node_p_idx),
            node_n_idx: opt_idx(c.node_n_idx),
        }).collect(),
        inductors: inductors.iter().map(|l| crate::types::RustInductor {
            inductance: l.inductance,
            ic: l.ic,
            node_p_idx: opt_idx(l.node_p_idx),
            node_n_idx: opt_idx(l.node_n_idx),
        }).collect(),
        coupled: coupled.unwrap_or(&[]).iter().map(|cp| crate::types::RustCoupledInductorPair {
            ic1: cp.ic1,
            ic2: cp.ic2,
            l1: cp.l1,
            l2: cp.l2,
            k: cp.k,
            p1_idx: opt_idx(cp.p1_idx),
            n1_idx: opt_idx(cp.n1_idx),
            p2_idx: opt_idx(cp.p2_idx),
            n2_idx: opt_idx(cp.n2_idx),
        }).collect(),
    }
}

// ---------------------------------------------------------------------------
// Exported Python function: solve_transient_rust
// ---------------------------------------------------------------------------

/// Solve high-performance time-domain transient simulation in Rust.
#[pyfunction]
#[pyo3(signature = (n_nodes, size, resistors, vsources, isources, capacitors, inductors, t_stop, dt, method=0, uic=false, coupled=None, vcvs=None, vccs=None, cccs=None, ccvs=None, opamps=None))]
fn solve_transient_rust<'py>(
    py: Python<'py>,
    n_nodes: usize,
    size: usize,
    resistors: Vec<PyResistor>,
    vsources: Vec<PySourceTran>,
    isources: Vec<PySourceTran>,
    capacitors: Vec<PyCapacitor>,
    inductors: Vec<PyInductor>,
    t_stop: f64,
    dt: f64,
    method: u8,
    uic: bool,
    coupled: Option<Vec<PyCoupledPair>>,
    vcvs: Option<Vec<PyVCVS>>,
    vccs: Option<Vec<PyVCCS>>,
    cccs: Option<Vec<PyCCCS>>,
    ccvs: Option<Vec<PyCCVS>>,
    opamps: Option<Vec<PyOpAmp>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,  // time_vec (n_steps)
    Bound<'py, PyArray2<f64>>,  // node_voltages (n_nodes, n_steps)
    Bound<'py, PyArray2<f64>>,  // cap_currents (n_caps, n_steps)
    Bound<'py, PyArray2<f64>>,  // ind_currents (n_inds, n_steps)
    Bound<'py, PyArray2<f64>>,  // coupled_currents (2 * n_coupled, n_steps)
    Bound<'py, PyArray2<f64>>,  // vsource_currents (n_vsources, n_steps)
    Bound<'py, PyArray1<f64>>,  // energy_stored (n_steps)
    Bound<'py, PyArray1<f64>>,  // power_dissipated (n_steps)
    u64,                        // lu_factor_time_ns
    u64,                        // step_loop_time_ns
)> {
    let circuit = to_rust_transient_circuit(
        n_nodes,
        size,
        &resistors,
        &vsources,
        &isources,
        &capacitors,
        &inductors,
        coupled.as_deref(),
        vcvs.as_deref(),
        vccs.as_deref(),
        cccs.as_deref(),
        ccvs.as_deref(),
        opamps.as_deref(),
    );

    let out = transient::run_transient(&circuit, t_stop, dt, method, uic)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let n_steps = out.time_vec.len();
    let n_caps = circuit.capacitors.len();
    let n_inds = circuit.inductors.len();

    let node_voltages_arr = if n_nodes > 0 && n_steps > 0 {
        ndarray::Array2::from_shape_vec((n_nodes, n_steps), out.node_voltages)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py)
    } else {
        ndarray::Array2::<f64>::zeros((0, n_steps)).into_pyarray(py)
    };

    let cap_currents_arr = if n_caps > 0 && n_steps > 0 {
        ndarray::Array2::from_shape_vec((n_caps, n_steps), out.cap_currents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py)
    } else {
        ndarray::Array2::<f64>::zeros((0, n_steps)).into_pyarray(py)
    };

    let ind_currents_arr = if n_inds > 0 && n_steps > 0 {
        ndarray::Array2::from_shape_vec((n_inds, n_steps), out.ind_currents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py)
    } else {
        ndarray::Array2::<f64>::zeros((0, n_steps)).into_pyarray(py)
    };

    let n_coupled = circuit.coupled.len();
    let n_vsources = circuit.vsources.len();

    let coupled_currents_arr = if n_coupled > 0 && n_steps > 0 {
        ndarray::Array2::from_shape_vec((2 * n_coupled, n_steps), out.coupled_currents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py)
    } else {
        ndarray::Array2::<f64>::zeros((0, n_steps)).into_pyarray(py)
    };

    let vsource_currents_arr = if n_vsources > 0 && n_steps > 0 {
        ndarray::Array2::from_shape_vec((n_vsources, n_steps), out.vsource_currents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py)
    } else {
        ndarray::Array2::<f64>::zeros((0, n_steps)).into_pyarray(py)
    };

    Ok((
        out.time_vec.into_pyarray(py),
        node_voltages_arr,
        cap_currents_arr,
        ind_currents_arr,
        coupled_currents_arr,
        vsource_currents_arr,
        out.energy_stored.into_pyarray(py),
        out.power_dissipated.into_pyarray(py),
        out.lu_factor_time_ns,
        out.step_loop_time_ns,
    ))
}


// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn pycircuit_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResistor>()?;
    m.add_class::<PyVSource>()?;
    m.add_class::<PyISource>()?;
    m.add_class::<PyVCVS>()?;
    m.add_class::<PyVCCS>()?;
    m.add_class::<PyCCCS>()?;
    m.add_class::<PyCCVS>()?;
    m.add_class::<PyOpAmp>()?;
    m.add_class::<PyInductorDC>()?;
    m.add_class::<PyCapacitor>()?;
    m.add_class::<PyInductor>()?;
    m.add_class::<PyCoupledPair>()?;
    m.add_class::<PySourceTran>()?;
    m.add_function(wrap_pyfunction!(monte_carlo_run, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mna_single, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mna_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_transient_rust, m)?)?;
    Ok(())
}

