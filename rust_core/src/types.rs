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

/// Voltage-Controlled Voltage Source (VCVS / E element).
#[derive(Debug, Clone)]
pub struct RustVCVS {
    pub gain: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
    pub ctrl_p_idx: Option<usize>,
    pub ctrl_n_idx: Option<usize>,
    pub aux_idx: usize,
}

/// Voltage-Controlled Current Source (VCCS / G element).
#[derive(Debug, Clone)]
pub struct RustVCCS {
    pub gm: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
    pub ctrl_p_idx: Option<usize>,
    pub ctrl_n_idx: Option<usize>,
}

/// Current-Controlled Current Source (CCCS / F element).
#[derive(Debug, Clone)]
pub struct RustCCCS {
    pub gain: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
    pub vctrl_aux_idx: usize,
}

/// Current-Controlled Voltage Source (CCVS / H element).
#[derive(Debug, Clone)]
pub struct RustCCVS {
    pub transresistance: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
    pub aux_idx: usize,
    pub vctrl_aux_idx: usize,
}

/// Ideal Operational Amplifier (Op-Amp / U or A element).
#[derive(Debug, Clone)]
pub struct RustOpAmp {
    pub in_p_idx: Option<usize>,
    pub in_n_idx: Option<usize>,
    pub out_idx: Option<usize>,
    pub gnd_idx: Option<usize>,
    pub aux_idx: usize,
}

/// Inductor in DC steady-state (modeled as 1e6 conductance short).
#[derive(Debug, Clone)]
pub struct RustInductorDC {
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
    pub vcvs: Vec<RustVCVS>,
    pub vccs: Vec<RustVCCS>,
    pub cccs: Vec<RustCCCS>,
    pub ccvs: Vec<RustCCVS>,
    pub opamps: Vec<RustOpAmp>,
    pub inductors_dc: Vec<RustInductorDC>,
}

// ---------------------------------------------------------------------------
// Transient Time-Domain Dynamic Elements & Waveforms
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum RustWaveform {
    DC(f64),
    Step { v1: f64, v2: f64, t_step: f64 },
    Pulse { v1: f64, v2: f64, td: f64, tr: f64, tf: f64, pw: f64, per: f64 },
    Sine { vo: f64, va: f64, freq: f64, td: f64, theta: f64 },
}

impl RustWaveform {
    #[inline]
    pub fn eval(&self, t: f64) -> f64 {
        match *self {
            RustWaveform::DC(val) => val,
            RustWaveform::Step { v1, v2, t_step } => {
                if t < t_step { v1 } else { v2 }
            }
            RustWaveform::Pulse { v1, v2, td, tr, tf, pw, per } => {
                if t < td {
                    return v1;
                }
                let t_rel = if per > 0.0 { (t - td) % per } else { t - td };
                if t_rel < tr {
                    v1 + (v2 - v1) * (t_rel / tr.max(1e-15))
                } else if t_rel < tr + pw {
                    v2
                } else if t_rel < tr + pw + tf {
                    v2 - (v2 - v1) * ((t_rel - tr - pw) / tf.max(1e-15))
                } else {
                    v1
                }
            }
            RustWaveform::Sine { vo, va, freq, td, theta } => {
                if t < td {
                    vo + va * (theta.to_radians()).sin()
                } else {
                    let angle = 2.0 * std::f64::consts::PI * freq * (t - td) + theta.to_radians();
                    vo + va * angle.sin()
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RustCapacitor {
    pub capacitance: f64,
    pub ic: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RustInductor {
    pub inductance: f64,
    pub ic: f64,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RustCoupledInductorPair {
    pub ic1: f64,
    pub ic2: f64,
    pub l1: f64,
    pub l2: f64,
    pub k: f64,
    pub p1_idx: Option<usize>,
    pub n1_idx: Option<usize>,
    pub p2_idx: Option<usize>,
    pub n2_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RustVSourceTran {
    pub waveform: RustWaveform,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
    pub aux_idx: usize,
}

#[derive(Debug, Clone)]
pub struct RustISourceTran {
    pub waveform: RustWaveform,
    pub node_p_idx: Option<usize>,
    pub node_n_idx: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RustTransientCircuit {
    pub n_nodes: usize,
    pub size: usize,
    pub resistors: Vec<RustResistorStamped>,
    pub vsources: Vec<RustVSourceTran>,
    pub isources: Vec<RustISourceTran>,
    pub vcvs: Vec<RustVCVS>,
    pub vccs: Vec<RustVCCS>,
    pub cccs: Vec<RustCCCS>,
    pub ccvs: Vec<RustCCVS>,
    pub opamps: Vec<RustOpAmp>,
    pub capacitors: Vec<RustCapacitor>,
    pub inductors: Vec<RustInductor>,
    pub coupled: Vec<RustCoupledInductorPair>,
}
