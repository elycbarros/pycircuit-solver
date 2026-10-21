"""
test_validator.py — Testes para o validador topológico e conversor de sufixos de engenharia.
"""

import pytest

from circuit_solver.core.elements import Resistor, VoltageSource
from circuit_solver.core.validator import (
    CircuitValidationError,
    CircuitValidator,
    parse_value,
)


def test_parse_value_engineering_suffixes():
    assert parse_value("4.7k") == pytest.approx(4700.0)
    assert parse_value("4k7") == pytest.approx(4700.0)
    assert parse_value("10mA") == pytest.approx(0.01)
    assert parse_value("220u") == pytest.approx(220e-6)
    assert parse_value("1MEG") == pytest.approx(1e6)
    assert parse_value("100") == pytest.approx(100.0)
    assert parse_value(50) == pytest.approx(50.0)
    assert parse_value(1.25) == pytest.approx(1.25)


def test_missing_ground_raises():
    elements = [
        VoltageSource("V1", "1", "2", 10.0),
        Resistor("R1", "1", "2", 100.0),
    ]
    with pytest.raises(CircuitValidationError, match="Ground node '0' not found"):
        CircuitValidator(elements).validate()


def test_self_short_raises():
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "1", 100.0),
    ]
    with pytest.raises(CircuitValidationError, match="node_p == node_n"):
        CircuitValidator(elements).validate()


def test_negative_resistance_raises():
    with pytest.raises(ValueError, match="resistance must be > 0"):
        Resistor("R1", "1", "0", -50.0)


def test_floating_node_raises():
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        Resistor("R1", "1", "0", 100.0),
        # Node "2" is connected only to R2
        Resistor("R2", "1", "2", 200.0),
    ]
    with pytest.raises(CircuitValidationError, match="Floating .* node.* detected"):
        CircuitValidator(elements).validate()


def test_parallel_voltage_sources_conflict_raises():
    elements = [
        VoltageSource("V1", "1", "0", 10.0),
        VoltageSource("V2", "1", "0", 12.0),
    ]
    with pytest.raises(CircuitValidationError, match="Conflicting voltage sources"):
        CircuitValidator(elements).validate()
