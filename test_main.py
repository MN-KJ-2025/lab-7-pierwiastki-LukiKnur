import pickle
from typing import Any

import main
import numpy as np
import pytest

try:
    with open("expected", "rb") as f:
        expected = pickle.load(f)
except FileNotFoundError:
    print(
        "Error: The 'expected' file was not found. Please ensure it is in the correct directory."
    )
    expected = {
        "roots_20": [],
        "frob_a": [],
        "is_nonsingular": [],
    }


# --- Data Preparation ---

valid_roots_20 = [(c, res) for c, res in expected["roots_20"] if res is not None]
invalid_roots_20 = [(c, res) for c, res in expected["roots_20"] if res is None]

valid_frob_a = [(c, res) for c, res in expected["frob_a"] if res is not None]
invalid_frob_a = [(c, res) for c, res in expected["frob_a"] if res is None]

valid_is_nonsingular = [(A, res) for A, res in expected["is_nonsingular"] if res is not None]
invalid_is_nonsingular = [(A, res) for A, res in expected["is_nonsingular"] if res is None]


# --- Tests for roots_20 ---


@pytest.mark.parametrize("coef, expected_result", invalid_roots_20)
def test_roots_20_invalid_input(coef: Any, expected_result: None):
    """Tests if roots_20 correctly handles invalid input data by returning None."""
    actual = main.roots_20(coef)
    assert actual is None, f"For invalid input, expected None but got {actual}."


@pytest.mark.parametrize("coef, expected_result", valid_roots_20)
def test_roots_20_correct_solution(
    coef: np.ndarray, expected_result: tuple[np.ndarray, np.ndarray]
):
    """Tests if roots_20 correctly perturbs coefficients and finds roots."""
    # We need to set the seed to match the one used during data generation
    np.random.seed(42)
    actual_coef, actual_roots = main.roots_20(coef)
    expected_coef, expected_roots = expected_result

    assert actual_coef == pytest.approx(expected_coef), (
        "The perturbed coefficients do not match the expected ones."
    )
    # Sort roots before comparison as the order is not guaranteed
    assert np.sort(actual_roots) == pytest.approx(np.sort(expected_roots)), (
        "The calculated roots do not match the expected ones."
    )


# --- Tests for frob_a ---


@pytest.mark.parametrize("coef, expected_result", invalid_frob_a)
def test_frob_a_invalid_input(coef: Any, expected_result: None):
    """Tests if frob_a correctly handles invalid input data by returning None."""
    actual = main.frob_a(coef)
    assert actual is None, f"For invalid input, expected None but got {actual}."


@pytest.mark.parametrize("coef, expected_result", valid_frob_a)
def test_frob_a_correct_solution(coef: np.ndarray, expected_result: np.ndarray):
    """Tests if frob_a creates the correct Frobenius matrix for valid inputs."""
    actual_result = main.frob_a(coef)
    assert actual_result == pytest.approx(expected_result), (
        "The created Frobenius matrix is incorrect."
    )


# --- Tests for is_nonsingular ---


@pytest.mark.parametrize("A, expected_result", invalid_is_nonsingular)
def test_is_nonsingular_invalid_input(A: Any, expected_result: None):
    """Tests if is_nonsingular correctly handles invalid input data by returning None."""
    actual = main.is_nonsingular(A)
    assert actual is None, f"For invalid input, expected None but got {actual}."


@pytest.mark.parametrize("A, expected_result", valid_is_nonsingular)
def test_is_nonsingular_correct_solution(A: np.ndarray, expected_result: bool):
    """Tests if is_nonsingular correctly identifies non-singular matrices."""
    actual_result = main.is_nonsingular(A)
    assert actual_result == expected_result, (
        f"Expected singularity to be {expected_result}, but got {actual_result}."
    )