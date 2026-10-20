/// types.rs — Plain-data types shared between Rust modules.
///
/// All structs are Clone + Send + Sync so rayon can scatter them
/// across worker threads without any locking.

/// A single resistor with its nominal value and tolerance.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RustResistor {
    /// Nominal resistance in Ohms.
    pub resistance: f64,
    /// Fractional tolerance (e.g. 0.05 for ±5 %).
    pub tolerance: f64,
    /// Index into the circuit's resistor list (used for stamping).
    pub index: usize,
}

/// A voltage source fixed value (DC).
#[derive(Debug, Clone)]
pub struct RustVSource {
    pub voltage: f64,
    pub node_p_idx: Option<usize>, // None = ground
    pub node_n_idx: Option<usize>, // None = ground
    /// Auxiliary variable index in expanded MNA (= n_nodes + k).
    pub aux_idx: usize,
}

/// A current source fixed value (DC).
#[derive(Debug, Clone)]
pub struct RustISource {
    pub current: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
}

/// A resistor with its MNA node indices pre-resolved.
#[derive(Debug, Clone)]
pub struct RustResistorStamped {
    pub resistance: f64,
    pub tolerance: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
}

/// Complete circuit description suitable for repeated MNA assembly in Rust.
///
/// All string-based node lookups are pre-resolved to integer indices
/// on the Python side before passing to Rust, eliminating HashMap
/// overhead in the hot Monte Carlo loop.
#[derive(Debug, Clone)]
pub struct RustCircuit {
    /// Number of non-ground nodes (size of G sub-matrix).
    pub n_nodes: usize,
    /// Total MNA system size = n_nodes + n_aux.
    pub size: usize,
    pub resistors: Vec<RustResistorStamped>,
    pub vsources: Vec<RustVSource>,
    pub isources: Vec<RustISource>,
}
