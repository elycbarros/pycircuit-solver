"""
builtin_library.py — Pedagogical and standard subcircuit model library.

This library provides ready-to-use, academically proven macromodels from classic
textbooks (Alexander & Sadiku, Sedra & Smith, Boylestad) and standard industry
configurations for simulation without boilerplate.
"""

from __future__ import annotations

from circuit_solver.core.subcircuit import SubcircuitDefinition


BUILTIN_SUBCIRCUITS: list[SubcircuitDefinition] = [
    # -----------------------------------------------------------------------
    # 1. Classic Operational Amplifier 741 Macromodel
    # Alexander & Sadiku / Sedra & Smith
    # -----------------------------------------------------------------------
    SubcircuitDefinition(
        name="OPAMP_741",
        pins=["inp", "inn", "out"],
        default_params={"GAIN": 200000.0, "RIN": 2e6, "ROUT": 75.0},
        lines=[
            "* 741 Op-Amp Macromodel: High Rin, Controlled Voltage Source, and Output Impedance",
            "Rin inp inn {RIN}",
            "Egain int_out 0 inp inn {GAIN}",
            "Rout int_out out {ROUT}",
        ],
    ),
    # -----------------------------------------------------------------------
    # 2. Ideal Op-Amp (Antigravity extension via A/U element)
    # -----------------------------------------------------------------------
    SubcircuitDefinition(
        name="OPAMP_IDEAL",
        pins=["inp", "inn", "out"],
        lines=[
            "* Ideal Op-Amp using nullor virtual short-circuit",
            "A1 inp inn out 0",
        ],
    ),
    # -----------------------------------------------------------------------
    # 3. Parametric First-Order RC Filter
    # -----------------------------------------------------------------------
    SubcircuitDefinition(
        name="RC_LOWPASS",
        pins=["in", "out"],
        default_params={"R": 10000.0, "C": 1e-7},
        lines=[
            "* Parametric First-Order Lowpass Filter",
            "R1 in out {R}",
            "C1 out 0 {C}",
        ],
    ),
    # -----------------------------------------------------------------------
    # 4. Parametric Second-Order Sallen-Key Active Filter (Sedra & Smith Cap. 14)
    # -----------------------------------------------------------------------
    SubcircuitDefinition(
        name="SALLEN_KEY_LP",
        pins=["in", "out"],
        default_params={"R1": 10000.0, "R2": 10000.0, "C1": 1e-8, "C2": 1e-8},
        lines=[
            "* Sallen-Key Unity Gain Low-Pass Active Filter",
            "R1 in n1 {R1}",
            "R2 n1 n_in_p {R2}",
            "C2 n_in_p 0 {C2}",
            "C1 n1 out {C1}",
            "A1 n_in_p out out 0",
        ],
    ),
    # -----------------------------------------------------------------------
    # 5. CMOS Digital Inverter Cell (Microelectronic Circuits — Sedra/Smith)
    # -----------------------------------------------------------------------
    SubcircuitDefinition(
        name="CMOS_INVERTER",
        pins=["in", "out", "vdd", "gnd"],
        lines=[
            "* Complementary MOS Inverter",
            "Mp out in vdd vdd PMOS VTH=-1.0 KP=2e-3",
            "Mn out in gnd gnd NMOS VTH=1.0 KP=1e-3",
        ],
    ),
    # -----------------------------------------------------------------------
    # 6. BJT Differential Pair with Resistive Load
    # -----------------------------------------------------------------------
    SubcircuitDefinition(
        name="BJT_DIFF_PAIR",
        pins=["in1", "in2", "out1", "out2", "vcc", "v_tail"],
        default_params={"RC": 5000.0, "BETA": 100.0},
        lines=[
            "* Bipolar Differential Amplifier Stage",
            "Rc1 vcc out1 {RC}",
            "Rc2 vcc out2 {RC}",
            "Q1 out1 in1 tail NPN BETA={BETA}",
            "Q2 out2 in2 tail NPN BETA={BETA}",
            "Rtail tail v_tail 1000.0",
        ],
    ),
]
