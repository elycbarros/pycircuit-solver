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
#[pyo3(signature = (n_nodes, size, resistors, vsources, isources, n_samples, seed, override_tol=None))]
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
    let circuit = to_rust_circuit(n_nodes, size, &resistors, &vsources, &isources);

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
fn solve_mna_single<'py>(
    py: Python<'py>,
    n_nodes: usize,
    size: usize,
    resistors: Vec<PyResistor>,
    vsources: Vec<PyVSource>,
    isources: Vec<PyISource>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let circuit = to_rust_circuit(n_nodes, size, &resistors, &vsources, &isources);
    match mna_core::solve_mna(&circuit, None) {
        Some(x) => {
            let v: Vec<f64> = (0..n_nodes).map(|i| x[i]).collect();
            Ok(Some(v.into_pyarray(py)))
        }
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn pycircuit_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResistor>()?;
    m.add_class::<PyVSource>()?;
    m.add_class::<PyISource>()?;
    m.add_function(wrap_pyfunction!(monte_carlo_run, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mna_single, m)?)?;
    Ok(())
}
