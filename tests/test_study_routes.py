"""The five proposed study routes, with independent mathematical references."""
import math

import pytest

from circuit_solver import build_three_phase_source
from circuit_solver.engine.transient_rust import RUST_AVAILABLE
from scripts.validate_study_routes import (
    check_divider, check_first_order, check_transformer, check_three_phase, check_tolerances,
)

RUST = pytest.mark.skipif(not RUST_AVAILABLE, reason='Rust extension is optional')
BACKENDS = ['python', pytest.param('rust', marks=RUST)]


@pytest.mark.parametrize('solver', ['dense', 'sparse', pytest.param('rust_sparse', marks=RUST)])
def test_study_divider(solver):
    check_divider(solver)


@pytest.mark.parametrize('kind', ['RC', 'RL'])
@pytest.mark.parametrize('method', ['euler', 'trapezoidal'])
@pytest.mark.parametrize('backend', BACKENDS)
def test_study_first_order(kind, method, backend):
    check_first_order(kind, method, backend)


@pytest.mark.parametrize('k', [.5, -.5])
@pytest.mark.parametrize('backend', BACKENDS)
def test_study_transformer(k, backend):
    check_transformer(k, backend)


@pytest.mark.parametrize('sequence', ['ABC', 'ACB'])
def test_study_three_phase(sequence):
    check_three_phase(sequence)


@pytest.mark.parametrize('backend', BACKENDS)
def test_study_tolerances(backend):
    check_tolerances(backend)


@pytest.mark.parametrize('connection', ['Y', 'DELTA'])
@pytest.mark.parametrize('sequence', ['ABC', 'ACB'])
def test_three_phase_source_rms_to_peak(connection, sequence):
    sources = build_three_phase_source(connection=connection, sequence=sequence, V_rms=120)
    assert all(source.ac_mag == pytest.approx(120*math.sqrt(2)) for source in sources)
