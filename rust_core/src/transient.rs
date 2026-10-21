/// transient.rs — High-performance time-domain transient analysis engine in Rust.
///
/// Implements companion models for linear dynamic elements (Capacitors, Inductors,
/// and Coupled Inductors / Transformers) using implicit numerical integration:
///   * Method 0: Trapezoidal Rule (2nd-order, A-stable, energy-conserving)
///   * Method 1: Backward Euler (1st-order, L-stable, high numerical damping)
///
/// For a fixed time-step dt, the MNA system matrix A is constant in time and is
/// LU-factorized once in O(N^3). Each time step is then solved in O(N^2) via
/// forward/backward substitution, achieving throughputs of over 1,000,000 steps/sec.
///
/// Pedagogical & Academic Introspection:
///   * Step-by-step energy audit: tracking total electric & magnetic energy stored E(t).
///   * Total instantaneous Joule power dissipated P_diss(t).
///   * Exact benchmark timer tracking LU factorization vs time-stepping loop.

use std::time::Instant;
use nalgebra::{DMatrix, DVector};

use crate::types::{
    RustTransientCircuit,
};

pub struct TransientRunOutput {
    pub time_vec: Vec<f64>,
    pub node_voltages: Vec<f64>,    // shape: (n_nodes, n_steps), row-major
    pub cap_currents: Vec<f64>,     // shape: (n_caps, n_steps)
    pub ind_currents: Vec<f64>,     // shape: (n_inds, n_steps)
    pub coupled_currents: Vec<f64>, // shape: (2 * n_coupled, n_steps)
    pub vsource_currents: Vec<f64>, // shape: (n_vsources, n_steps)
    pub energy_stored: Vec<f64>,    // shape: (n_steps,)
    pub power_dissipated: Vec<f64>, // shape: (n_steps,)
    pub lu_factor_time_ns: u64,
    pub step_loop_time_ns: u64,
}

#[inline]
fn stamp_conductance(a: &mut DMatrix<f64>, p: Option<usize>, n: Option<usize>, g: f64) {
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

/// Capacitores cujos terminais já estão ligados por fontes de tensão ideais.
/// Espelha `_capacitors_forced_by_voltage_sources` em transient.py. O terra é o índice `n_nodes`.
fn forced_capacitors(circuit: &RustTransientCircuit) -> Vec<bool> {
    let nn = circuit.n_nodes;
    let mut parent: Vec<usize> = (0..=nn).collect();

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    let node_or_ground = |o: Option<usize>| o.unwrap_or(nn);

    for vs in &circuit.vsources {
        let ra = find(&mut parent, node_or_ground(vs.node_p_idx));
        let rb = find(&mut parent, node_or_ground(vs.node_n_idx));
        if ra != rb {
            parent[ra] = rb;
        }
    }

    let mut forced = Vec::with_capacity(circuit.capacitors.len());
    for c in &circuit.capacitors {
        let ra = find(&mut parent, node_or_ground(c.node_p_idx));
        let rb = find(&mut parent, node_or_ground(c.node_n_idx));
        forced.push(ra == rb);
    }
    forced
}

pub fn run_transient(
    circuit: &RustTransientCircuit,
    t_stop: f64,
    dt: f64,
    method: u8, // 0 = Trapezoidal, 1 = Euler
    uic: bool,
) -> Result<TransientRunOutput, String> {
    if !t_stop.is_finite() || !dt.is_finite() || method > 1 {
        return Err("Transient times must be finite and method must be 0 or 1.".to_string());
    }
    if t_stop <= 0.0 {
        return Err(format!("t_stop must be positive, got {t_stop}"));
    }
    if dt <= 0.0 || dt > t_stop {
        return Err(format!("dt must be positive and <= t_stop, got {dt}"));
    }

    let n = circuit.size;
    let n_nodes = circuit.n_nodes;
    let n_steps = ((t_stop + 0.5 * dt) / dt).floor() as usize + 1;

    let time_vec: Vec<f64> = (0..n_steps).map(|k| (k as f64) * dt).collect();

    // -----------------------------------------------------------------------
    // 1. Companion conductances
    // -----------------------------------------------------------------------
    let is_trap = method == 0;

    let g_c: Vec<f64> = circuit.capacitors.iter()
        .map(|c| if is_trap { 2.0 * c.capacitance / dt } else { c.capacitance / dt })
        .collect();

    let g_l: Vec<f64> = circuit.inductors.iter()
        .map(|ind| if is_trap { dt / (2.0 * ind.inductance) } else { dt / ind.inductance })
        .collect();

    // Coupled pairs inverse inductance Gamma and companion conductances G = alpha * Gamma
    let alpha = if is_trap { dt / 2.0 } else { dt };
    let mut coupled_gamma: Vec<[[f64; 2]; 2]> = Vec::with_capacity(circuit.coupled.len());
    let mut coupled_geq: Vec<[[f64; 2]; 2]> = Vec::with_capacity(circuit.coupled.len());

    for cp in &circuit.coupled {
        let k = cp.k.clamp(-0.9999999, 0.9999999);
        let det = cp.l1 * cp.l2 * (1.0 - k * k);
        let m = k * (cp.l1 * cp.l2).sqrt();
        let g11 = cp.l2 / det;
        let g12 = -m / det;
        let g21 = -m / det;
        let g22 = cp.l1 / det;
        let gamma = [[g11, g12], [g21, g22]];
        let geq = [[alpha * g11, alpha * g12], [alpha * g21, alpha * g22]];
        coupled_gamma.push(gamma);
        coupled_geq.push(geq);
    }

    // -----------------------------------------------------------------------
    // 2. Build time-invariant system matrix A
    // -----------------------------------------------------------------------
    let factor_start = Instant::now();
    let mut a = DMatrix::<f64>::zeros(n, n);

    // Resistors
    for r in &circuit.resistors {
        stamp_conductance(&mut a, r.node_p_idx, r.node_n_idx, 1.0 / r.resistance);
    }

    // Independent voltage sources
    for vs in &circuit.vsources {
        let k = vs.aux_idx;
        if let Some(p) = vs.node_p_idx {
            a[(p, k)] += 1.0;
            a[(k, p)] += 1.0;
        }
        if let Some(ni) = vs.node_n_idx {
            a[(ni, k)] -= 1.0;
            a[(k, ni)] -= 1.0;
        }
    }

    // VCVS (E element)
    for e in &circuit.vcvs {
        let k = e.aux_idx;
        if let Some(p) = e.node_p_idx {
            a[(p, k)] += 1.0;
            a[(k, p)] += 1.0;
        }
        if let Some(ni) = e.node_n_idx {
            a[(ni, k)] -= 1.0;
            a[(k, ni)] -= 1.0;
        }
        if let Some(cp) = e.ctrl_p_idx {
            a[(k, cp)] -= e.gain;
        }
        if let Some(cn) = e.ctrl_n_idx {
            a[(k, cn)] += e.gain;
        }
    }

    // VCCS (G element)
    for g in &circuit.vccs {
        let gm = g.gm;
        if let (Some(p), Some(cp)) = (g.node_p_idx, g.ctrl_p_idx) { a[(p, cp)] -= gm; }
        if let (Some(p), Some(cn)) = (g.node_p_idx, g.ctrl_n_idx) { a[(p, cn)] += gm; }
        if let (Some(ni), Some(cp)) = (g.node_n_idx, g.ctrl_p_idx) { a[(ni, cp)] += gm; }
        if let (Some(ni), Some(cn)) = (g.node_n_idx, g.ctrl_n_idx) { a[(ni, cn)] -= gm; }
    }

    // CCCS (F element)
    for f in &circuit.cccs {
        if let Some(p) = f.node_p_idx { a[(p, f.vctrl_aux_idx)] += f.gain; }
        if let Some(ni) = f.node_n_idx { a[(ni, f.vctrl_aux_idx)] -= f.gain; }
    }

    // CCVS (H element)
    for h in &circuit.ccvs {
        let k = h.aux_idx;
        if let Some(p) = h.node_p_idx { a[(p, k)] += 1.0; a[(k, p)] += 1.0; }
        if let Some(ni) = h.node_n_idx { a[(ni, k)] -= 1.0; a[(k, ni)] -= 1.0; }
        a[(k, h.vctrl_aux_idx)] += h.transresistance;
    }

    // Op-Amp
    for op in &circuit.opamps {
        let k = op.aux_idx;
        if let Some(out) = op.out_idx { a[(out, k)] += 1.0; }
        if let Some(gnd) = op.gnd_idx { a[(gnd, k)] -= 1.0; }
        if let Some(inp) = op.in_p_idx { a[(k, inp)] += 1.0; }
        if let Some(inn) = op.in_n_idx { a[(k, inn)] -= 1.0; }
    }

    // Capacitors companion conductances
    for (i, c) in circuit.capacitors.iter().enumerate() {
        stamp_conductance(&mut a, c.node_p_idx, c.node_n_idx, g_c[i]);
    }

    // Inductors companion conductances
    for (i, ind) in circuit.inductors.iter().enumerate() {
        stamp_conductance(&mut a, ind.node_p_idx, ind.node_n_idx, g_l[i]);
    }

    // Coupled pairs companion conductances
    for (i, cp) in circuit.coupled.iter().enumerate() {
        let geq = coupled_geq[i];
        let p1 = cp.p1_idx; let n1 = cp.n1_idx;
        let p2 = cp.p2_idx; let n2 = cp.n2_idx;

        stamp_conductance(&mut a, p1, n1, geq[0][0]);
        stamp_conductance(&mut a, p2, n2, geq[1][1]);

        let y12 = geq[0][1];
        let y21 = geq[1][0];
        if let (Some(p1), Some(p2)) = (p1, p2) { a[(p1, p2)] += y12; a[(p2, p1)] += y21; }
        if let (Some(p1), Some(n2)) = (p1, n2) { a[(p1, n2)] -= y12; a[(n2, p1)] -= y21; }
        if let (Some(n1), Some(p2)) = (n1, p2) { a[(n1, p2)] -= y12; a[(p2, n1)] -= y21; }
        if let (Some(n1), Some(n2)) = (n1, n2) { a[(n1, n2)] += y12; a[(n2, n1)] += y21; }
    }

    // LU factorization
    let lu = a.lu();
    let lu_factor_time_ns = factor_start.elapsed().as_nanos() as u64;

    // -----------------------------------------------------------------------
    // 3. State initialization (t = 0)
    // -----------------------------------------------------------------------
    let mut v_c_state: Vec<f64> = circuit.capacitors.iter().map(|c| if uic { c.ic } else { 0.0 }).collect();
    let mut i_c_state: Vec<f64> = vec![0.0; circuit.capacitors.len()];

    let mut v_l_state: Vec<f64> = vec![0.0; circuit.inductors.len()];
    let mut i_l_state: Vec<f64> = circuit.inductors.iter().map(|ind| if uic { ind.ic } else { 0.0 }).collect();

    let mut v_coupled_state: Vec<[f64; 2]> = vec![[0.0, 0.0]; circuit.coupled.len()];
    let mut i_coupled_state: Vec<[f64; 2]> = circuit.coupled.iter()
        .map(|cp| if uic { [cp.ic1, cp.ic2] } else { [0.0, 0.0] }).collect();

    // Output buffers: flattened (item, step) -> idx = item * n_steps + step
    let mut node_voltages = vec![0.0; n_nodes * n_steps];
    let mut cap_currents = vec![0.0; circuit.capacitors.len() * n_steps];
    let mut ind_currents = vec![0.0; circuit.inductors.len() * n_steps];
    let mut coupled_currents = vec![0.0; 2 * circuit.coupled.len() * n_steps];
    let mut vsource_currents = vec![0.0; circuit.vsources.len() * n_steps];
    let mut energy_stored = vec![0.0; n_steps];
    let mut power_dissipated = vec![0.0; n_steps];

    let mut z0 = DVector::<f64>::zeros(n);
    for vs in &circuit.vsources {
        z0[vs.aux_idx] = vs.waveform.eval(0.0);
    }
    for is in &circuit.isources {
        let val = is.waveform.eval(0.0);
        if let Some(p) = is.node_p_idx { z0[p] += val; }
        if let Some(ni) = is.node_n_idx { z0[ni] -= val; }
    }
    let x0 = if uic {
        // t = 0+ com condições iniciais (espelha transient.rs/Python).
        // Capacitor sem fonte forçada vira fonte de tensão de valor ic (uma incógnita extra);
        // indutor vira fonte de corrente de valor ic; fontes independentes entram em t = 0.
        let forced = forced_capacitors(circuit);
        let sub_caps: Vec<usize> = (0..circuit.capacitors.len()).filter(|&i| !forced[i]).collect();
        let ns = n + sub_caps.len();
        let mut a0 = DMatrix::<f64>::zeros(ns, ns);
        let mut z_aug = DVector::<f64>::zeros(ns);

        for r in &circuit.resistors {
            stamp_conductance(&mut a0, r.node_p_idx, r.node_n_idx, 1.0 / r.resistance);
        }
        for vs in &circuit.vsources {
            let k = vs.aux_idx;
            if let Some(p) = vs.node_p_idx { a0[(p, k)] += 1.0; a0[(k, p)] += 1.0; }
            if let Some(ni) = vs.node_n_idx { a0[(ni, k)] -= 1.0; a0[(k, ni)] -= 1.0; }
            z_aug[k] = vs.waveform.eval(0.0);
        }
        for is in &circuit.isources {
            let val = is.waveform.eval(0.0);
            if let Some(p) = is.node_p_idx { z_aug[p] += val; }
            if let Some(ni) = is.node_n_idx { z_aug[ni] -= val; }
        }
        for vcvs in &circuit.vcvs {
            let k = vcvs.aux_idx;
            if let Some(p) = vcvs.node_p_idx { a0[(p, k)] += 1.0; a0[(k, p)] += 1.0; }
            if let Some(ni) = vcvs.node_n_idx { a0[(ni, k)] -= 1.0; a0[(k, ni)] -= 1.0; }
            if let Some(cp) = vcvs.ctrl_p_idx { a0[(k, cp)] -= vcvs.gain; }
            if let Some(cn) = vcvs.ctrl_n_idx { a0[(k, cn)] += vcvs.gain; }
        }
        for vccs in &circuit.vccs {
            let gm = vccs.gm;
            if let (Some(p), Some(cp)) = (vccs.node_p_idx, vccs.ctrl_p_idx) { a0[(p, cp)] -= gm; }
            if let (Some(p), Some(cn)) = (vccs.node_p_idx, vccs.ctrl_n_idx) { a0[(p, cn)] += gm; }
            if let (Some(ni), Some(cp)) = (vccs.node_n_idx, vccs.ctrl_p_idx) { a0[(ni, cp)] += gm; }
            if let (Some(ni), Some(cn)) = (vccs.node_n_idx, vccs.ctrl_n_idx) { a0[(ni, cn)] -= gm; }
        }
        for cccs in &circuit.cccs {
            let k_vctrl = cccs.vctrl_aux_idx;
            if let Some(p) = cccs.node_p_idx { a0[(p, k_vctrl)] += cccs.gain; }
            if let Some(ni) = cccs.node_n_idx { a0[(ni, k_vctrl)] -= cccs.gain; }
        }
        for ccvs in &circuit.ccvs {
            let k = ccvs.aux_idx;
            let k_vctrl = ccvs.vctrl_aux_idx;
            if let Some(p) = ccvs.node_p_idx { a0[(p, k)] += 1.0; a0[(k, p)] += 1.0; }
            if let Some(ni) = ccvs.node_n_idx { a0[(ni, k)] -= 1.0; a0[(k, ni)] -= 1.0; }
            a0[(k, k_vctrl)] += ccvs.transresistance;
        }
        for op in &circuit.opamps {
            let k = op.aux_idx;
            if let Some(out) = op.out_idx { a0[(out, k)] += 1.0; }
            if let Some(gnd) = op.gnd_idx { a0[(gnd, k)] -= 1.0; }
            if let Some(inp) = op.in_p_idx { a0[(k, inp)] += 1.0; }
            if let Some(inn) = op.in_n_idx { a0[(k, inn)] -= 1.0; }
        }
        // Indutor com corrente ic de node_p para node_n: injeta em node_n, drena de node_p
        for ind in &circuit.inductors {
            if let Some(ni) = ind.node_n_idx { z_aug[ni] += ind.ic; }
            if let Some(p) = ind.node_p_idx { z_aug[p] -= ind.ic; }
        }
        for cp in &circuit.coupled {
            if let Some(p) = cp.p1_idx { z_aug[p] -= cp.ic1; }
            if let Some(ni) = cp.n1_idx { z_aug[ni] += cp.ic1; }
            if let Some(p) = cp.p2_idx { z_aug[p] -= cp.ic2; }
            if let Some(ni) = cp.n2_idx { z_aug[ni] += cp.ic2; }
        }
        // Capacitor sem fonte forçada: fonte de tensão ic na incógnita extra n + j
        for (j, &ci) in sub_caps.iter().enumerate() {
            let c = &circuit.capacitors[ci];
            let k = n + j;
            if let Some(p) = c.node_p_idx { a0[(p, k)] += 1.0; a0[(k, p)] += 1.0; }
            if let Some(ni) = c.node_n_idx { a0[(ni, k)] -= 1.0; a0[(k, ni)] -= 1.0; }
            z_aug[k] = c.ic;
        }

        let x_aug = match a0.lu().solve(&z_aug) {
            Some(x) => x,
            None => return Err("Failed to solve transient system at t=0 (singular matrix).".to_string()),
        };

        let node_v = |o: Option<usize>| o.map(|i| x_aug[i]).unwrap_or(0.0);

        for (j, &ci) in sub_caps.iter().enumerate() {
            i_c_state[ci] = x_aug[n + j];
        }
        for (i, c) in circuit.capacitors.iter().enumerate() {
            if forced[i] {
                // tensão forçada pela fonte; corrente em t = 0 não recuperada (zero)
                v_c_state[i] = node_v(c.node_p_idx) - node_v(c.node_n_idx);
                i_c_state[i] = 0.0;
            }
        }
        for (i, ind) in circuit.inductors.iter().enumerate() {
            v_l_state[i] = node_v(ind.node_p_idx) - node_v(ind.node_n_idx);
        }
        for (i, cp) in circuit.coupled.iter().enumerate() {
            v_coupled_state[i] = [
                node_v(cp.p1_idx) - node_v(cp.n1_idx),
                node_v(cp.p2_idx) - node_v(cp.n2_idx),
            ];
        }

        x_aug.rows(0, n).into_owned()
    } else {
        // Assemble the same ideal DC model used by the Rust DC solvers.
        // Keep the transient companion matrix at its original size n.
        let mut windings = Vec::new();
        for ind in &circuit.inductors {
            windings.push(crate::types::RustInductorDC {
                node_p_idx: ind.node_p_idx, node_n_idx: ind.node_n_idx,
                aux_idx: n + windings.len(),
            });
        }
        for cp in &circuit.coupled {
            windings.push(crate::types::RustInductorDC {
                node_p_idx: cp.p1_idx, node_n_idx: cp.n1_idx,
                aux_idx: n + windings.len(),
            });
            windings.push(crate::types::RustInductorDC {
                node_p_idx: cp.p2_idx, node_n_idx: cp.n2_idx,
                aux_idx: n + windings.len(),
            });
        }
        let dc = crate::types::RustCircuit {
            n_nodes, size: n + windings.len(),
            resistors: circuit.resistors.clone(),
            vsources: circuit.vsources.iter().map(|vs| crate::types::RustVSource {
                voltage: vs.waveform.eval(0.0), node_p_idx: vs.node_p_idx,
                node_n_idx: vs.node_n_idx, aux_idx: vs.aux_idx,
            }).collect(),
            isources: circuit.isources.iter().map(|src| crate::types::RustISource {
                current: src.waveform.eval(0.0), node_p_idx: src.node_p_idx,
                node_n_idx: src.node_n_idx,
            }).collect(),
            vcvs: circuit.vcvs.clone(), vccs: circuit.vccs.clone(),
            cccs: circuit.cccs.clone(), ccvs: circuit.ccvs.clone(),
            opamps: circuit.opamps.clone(), inductors_dc: windings,
        };
        let x_dc = crate::mna_core::solve_mna(&dc, None).ok_or_else(||
            format!("DC operating point initialization failed: {}; specify uic=True for explicit initial conditions.",
                    crate::mna_core::dc_failure_reason(&dc))
        )?;
        for (i, c) in circuit.capacitors.iter().enumerate() {
            let vp = c.node_p_idx.map(|p| x_dc[p]).unwrap_or(0.0);
            let vn = c.node_n_idx.map(|ni| x_dc[ni]).unwrap_or(0.0);
            v_c_state[i] = vp - vn;
        }
        for i in 0..circuit.inductors.len() {
            i_l_state[i] = x_dc[n + i];
        }
        for i in 0..circuit.coupled.len() {
            let offset = n + circuit.inductors.len() + 2 * i;
            i_coupled_state[i] = [x_dc[offset], x_dc[offset + 1]];
        }
        x_dc.rows(0, n).into_owned()
    };

    if !x0.iter().all(|v| v.is_finite()) {
        return Err("Transient initialization produced non-finite values.".to_string());
    }

    // Store t=0 node voltages & currents
    for i in 0..n_nodes {
        node_voltages[i * n_steps] = x0[i];
    }
    for (i, _) in circuit.capacitors.iter().enumerate() {
        cap_currents[i * n_steps] = i_c_state[i];
    }
    for (i, _) in circuit.inductors.iter().enumerate() {
        ind_currents[i * n_steps] = i_l_state[i];
    }
    for (i, vs) in circuit.vsources.iter().enumerate() {
        vsource_currents[i * n_steps] = x0[vs.aux_idx];
    }
    for (i, currents) in i_coupled_state.iter().enumerate() {
        coupled_currents[(2 * i) * n_steps] = currents[0];
        coupled_currents[(2 * i + 1) * n_steps] = currents[1];
    }

    // Energy and power at t=0
    let mut e0_c = 0.0;
    for (i, c) in circuit.capacitors.iter().enumerate() {
        let vc = v_c_state[i];
        e0_c += 0.5 * c.capacitance * vc * vc;
    }
    let mut e0_l = 0.0;
    for (i, ind) in circuit.inductors.iter().enumerate() {
        let il = i_l_state[i];
        e0_l += 0.5 * ind.inductance * il * il;
    }
    for (cp, currents) in circuit.coupled.iter().zip(&i_coupled_state) {
        let [i1, i2] = *currents;
        let m = cp.k * (cp.l1 * cp.l2).sqrt();
        e0_l += 0.5 * cp.l1 * i1 * i1 + m * i1 * i2 + 0.5 * cp.l2 * i2 * i2;
    }
    energy_stored[0] = e0_c + e0_l;

    let mut p0_diss = 0.0;
    for r in &circuit.resistors {
        let vp = r.node_p_idx.map(|p| x0[p]).unwrap_or(0.0);
        let vn = r.node_n_idx.map(|ni| x0[ni]).unwrap_or(0.0);
        let vr = vp - vn;
        p0_diss += (vr * vr) / r.resistance;
    }
    power_dissipated[0] = p0_diss;


    // -----------------------------------------------------------------------
    // 4. Ultra-Fast Transient Time Stepping Loop
    // -----------------------------------------------------------------------
    let loop_start = Instant::now();
    let mut z = DVector::<f64>::zeros(n);
    let mut ieq_c = vec![0.0; circuit.capacitors.len()];
    let mut ieq_l = vec![0.0; circuit.inductors.len()];
    let mut ieq_coupled = vec![[0.0, 0.0]; circuit.coupled.len()];

    for k in 1..n_steps {
        let t_curr = time_vec[k];
        z.fill(0.0);

        // Sources at t_curr
        for vs in &circuit.vsources {
            z[vs.aux_idx] = vs.waveform.eval(t_curr);
        }
        for is in &circuit.isources {
            let val = is.waveform.eval(t_curr);
            if let Some(p) = is.node_p_idx { z[p] += val; }
            if let Some(ni) = is.node_n_idx { z[ni] -= val; }
        }

        // Capacitor companion currents
        for (i, c) in circuit.capacitors.iter().enumerate() {
            let ieq = if is_trap {
                g_c[i] * v_c_state[i] + i_c_state[i]
            } else {
                g_c[i] * v_c_state[i]
            };
            ieq_c[i] = ieq;
            if let Some(p) = c.node_p_idx { z[p] += ieq; }
            if let Some(ni) = c.node_n_idx { z[ni] -= ieq; }
        }

        // Inductor companion currents
        for (i, ind) in circuit.inductors.iter().enumerate() {
            let ieq = if is_trap {
                g_l[i] * v_l_state[i] + i_l_state[i]
            } else {
                i_l_state[i]
            };
            ieq_l[i] = ieq;
            if let Some(p) = ind.node_p_idx { z[p] -= ieq; }
            if let Some(ni) = ind.node_n_idx { z[ni] += ieq; }
        }

        // Coupled inductors companion currents
        for (i, cp) in circuit.coupled.iter().enumerate() {
            let geq = coupled_geq[i];
            let v_prev = v_coupled_state[i];
            let i_prev = i_coupled_state[i];
            let ieq = if is_trap {
                [
                    i_prev[0] + geq[0][0] * v_prev[0] + geq[0][1] * v_prev[1],
                    i_prev[1] + geq[1][0] * v_prev[0] + geq[1][1] * v_prev[1],
                ]
            } else {
                i_prev
            };
            ieq_coupled[i] = ieq;
            if let Some(p1) = cp.p1_idx { z[p1] -= ieq[0]; }
            if let Some(n1) = cp.n1_idx { z[n1] += ieq[0]; }
            if let Some(p2) = cp.p2_idx { z[p2] -= ieq[1]; }
            if let Some(n2) = cp.n2_idx { z[n2] += ieq[1]; }
        }

        // Solve step k: in-place forward/backward substitution
        if !lu.solve_mut(&mut z) {
            return Err(format!("Transient solve failed at step {k} (t={t_curr:.6e} s)"));
        }
        let x = &z;

        // Record node voltages
        for i in 0..n_nodes {
            node_voltages[i * n_steps + k] = x[i];
        }

        // Update capacitors
        let mut e_c_total = 0.0;
        for (i, c) in circuit.capacitors.iter().enumerate() {
            let vp = c.node_p_idx.map(|p| x[p]).unwrap_or(0.0);
            let vn = c.node_n_idx.map(|ni| x[ni]).unwrap_or(0.0);
            let v_curr = vp - vn;
            let i_curr = g_c[i] * v_curr - ieq_c[i];

            v_c_state[i] = v_curr;
            i_c_state[i] = i_curr;
            cap_currents[i * n_steps + k] = i_curr;
            e_c_total += 0.5 * c.capacitance * v_curr * v_curr;
        }

        // Update inductors
        let mut e_l_total = 0.0;
        for (i, ind) in circuit.inductors.iter().enumerate() {
            let vp = ind.node_p_idx.map(|p| x[p]).unwrap_or(0.0);
            let vn = ind.node_n_idx.map(|ni| x[ni]).unwrap_or(0.0);
            let v_curr = vp - vn;
            let i_curr = g_l[i] * v_curr + ieq_l[i];

            v_l_state[i] = v_curr;
            i_l_state[i] = i_curr;
            ind_currents[i * n_steps + k] = i_curr;
            e_l_total += 0.5 * ind.inductance * i_curr * i_curr;
        }

        // Update coupled pairs
        for (i, cp) in circuit.coupled.iter().enumerate() {
            let vp1 = cp.p1_idx.map(|p| x[p]).unwrap_or(0.0);
            let vn1 = cp.n1_idx.map(|ni| x[ni]).unwrap_or(0.0);
            let vp2 = cp.p2_idx.map(|p| x[p]).unwrap_or(0.0);
            let vn2 = cp.n2_idx.map(|ni| x[ni]).unwrap_or(0.0);
            let v1 = vp1 - vn1;
            let v2 = vp2 - vn2;
            let geq = coupled_geq[i];
            let ieq = ieq_coupled[i];
            let i1 = geq[0][0] * v1 + geq[0][1] * v2 + ieq[0];
            let i2 = geq[1][0] * v1 + geq[1][1] * v2 + ieq[1];
            v_coupled_state[i] = [v1, v2];
            i_coupled_state[i] = [i1, i2];
            coupled_currents[(2 * i) * n_steps + k] = i1;
            coupled_currents[(2 * i + 1) * n_steps + k] = i2;
            let m = cp.k * (cp.l1 * cp.l2).sqrt();
            e_l_total += 0.5 * cp.l1 * i1 * i1 + m * i1 * i2 + 0.5 * cp.l2 * i2 * i2;
        }

        // Update vsource currents (aux variables)
        for (i, vs) in circuit.vsources.iter().enumerate() {
            vsource_currents[i * n_steps + k] = x[vs.aux_idx];
        }

        // Power dissipation in resistors
        let mut p_diss = 0.0;
        for r in &circuit.resistors {
            let vp = r.node_p_idx.map(|p| x[p]).unwrap_or(0.0);
            let vn = r.node_n_idx.map(|ni| x[ni]).unwrap_or(0.0);
            let vr = vp - vn;
            p_diss += (vr * vr) / r.resistance;
        }

        energy_stored[k] = e_c_total + e_l_total;
        power_dissipated[k] = p_diss;
    }

    let step_loop_time_ns = loop_start.elapsed().as_nanos() as u64;

    Ok(TransientRunOutput {
        time_vec,
        node_voltages,
        cap_currents,
        ind_currents,
        coupled_currents,
        vsource_currents,
        energy_stored,
        power_dissipated,
        lu_factor_time_ns,
        step_loop_time_ns,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RustCapacitor, RustCoupledInductorPair, RustISourceTran,
        RustResistorStamped, RustVSourceTran, RustWaveform};

    fn coupled_fixture() -> RustTransientCircuit {
        RustTransientCircuit {
            n_nodes: 3, size: 4,
            resistors: vec![
                RustResistorStamped { resistance: 10.0, tolerance: 0.0,
                    node_p_idx: Some(0), node_n_idx: Some(1) },
                RustResistorStamped { resistance: 20.0, tolerance: 0.0,
                    node_p_idx: Some(2), node_n_idx: None },
            ],
            vsources: vec![RustVSourceTran { waveform: RustWaveform::DC(10.0),
                node_p_idx: Some(0), node_n_idx: None, aux_idx: 3 }],
            coupled: vec![RustCoupledInductorPair {
                l1: 0.02, l2: 0.005, k: 0.5, ic1: 0.2, ic2: -0.1,
                p1_idx: Some(1), n1_idx: None, p2_idx: Some(2), n2_idx: None,
            }],
            isources: vec![], vcvs: vec![], vccs: vec![], cccs: vec![],
            ccvs: vec![], opamps: vec![], capacitors: vec![], inductors: vec![],
        }
    }

    #[test]
    fn coupled_uic_preserves_currents_and_energy_at_zero() {
        let circuit = coupled_fixture();
        for method in [0, 1] {
            let out = run_transient(&circuit, 1e-4, 1e-6, method, true).unwrap();
            let steps = out.time_vec.len();
            assert_eq!(out.coupled_currents[0], 0.2);
            assert_eq!(out.coupled_currents[steps], -0.1);
            assert!((out.node_voltages[steps] - 8.0).abs() < 1e-12);
            assert!((out.node_voltages[2 * steps] - 2.0).abs() < 1e-12);
            assert!((out.energy_stored[0] - 0.000325).abs() < 1e-15);
            assert!(out.energy_stored.iter().all(|e| *e > 0.0));
        }
    }

    #[test]
    fn dc_coupled_windings_have_short_circuit_initial_currents() {
        let out = run_transient(&coupled_fixture(), 1e-4, 1e-6, 0, false).unwrap();
        let expected = 1.0;
        assert!((out.coupled_currents[0] - expected).abs() < 1e-12);
        assert_eq!(out.coupled_currents[out.time_vec.len()], 0.0);
        assert!((out.energy_stored[0] - 0.01 * expected * expected).abs() < 1e-14);
    }

    #[test]
    fn failed_dc_requires_explicit_uic() {
        let mut circuit = coupled_fixture();
        circuit.n_nodes = 1;
        circuit.size = 1;
        circuit.resistors.clear();
        circuit.vsources.clear();
        circuit.coupled.clear();
        circuit.capacitors.push(RustCapacitor { capacitance: 1e-6, ic: 1.0,
            node_p_idx: Some(0), node_n_idx: None });
        circuit.isources.push(RustISourceTran { waveform: RustWaveform::DC(0.001),
            node_p_idx: Some(0), node_n_idx: None });
        let error = run_transient(&circuit, 1e-3, 1e-5, 0, false).err().unwrap();
        assert!(error.contains("DC operating point initialization failed"));
        let out = run_transient(&circuit, 1e-3, 1e-5, 0, true).unwrap();
        for (v, t) in out.node_voltages.iter().zip(&out.time_vec) {
            assert!((v - (1.0 + 1000.0 * t)).abs() < 1e-12);
        }
    }
}
