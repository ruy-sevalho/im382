import numpy as np
import pandas as pd
from typing import Callable

from nomeclature import (
    ANALYTICAL_DISPLACEMENT,
    ERROR_SQUARED,
    NUM_DISPLACEMENT,
    X_COORD,
)


def calc_approx_value(
    x_knots_global: np.ndarray,
    element_incidence_matrix: np.ndarray,
    knot_displacements: np.ndarray,
    ecsi_matrix: np.ndarray,
    ecsi_calc_pts: np.ndarray,
    factor: float = 1,
    result_name: str = "res",
):
    results = pd.DataFrame()

    for i, e in enumerate(element_incidence_matrix):
        x_element_global = np.interp(ecsi_calc_pts, [-1, 1], x_knots_global[i : i + 2])
        element_displacements = knot_displacements[e]
        res = np.array([factor * col @ element_displacements for col in ecsi_matrix.T])
        results = pd.concat(
            [results, pd.DataFrame({X_COORD: x_element_global, result_name: res})],
        )

    return results


def calc_approx_value_isop(
    p_knots_global: np.ndarray,
    element_incidence_matrix: np.ndarray,
    knot_displacements: np.ndarray,
    esci_matrix_: np.ndarray,
    factor: float = 1,
    result_name: str = "res",
):
    results = pd.DataFrame()

    for i, e in enumerate(element_incidence_matrix):
        p_element_global = esci_matrix_.T @ p_knots_global[e]
        element_displacements = knot_displacements[e]
        res = np.array([factor * col @ element_displacements for col in esci_matrix_.T])
        results = pd.concat(
            [results, pd.DataFrame({X_COORD: p_element_global, result_name: res})],
        )

    return results


def calc_strain_variable_det_j(
    p_knots_global: np.ndarray,
    element_incidence_matrix: np.ndarray,
    knot_displacements: np.ndarray,
    n_ecsi_matrix: np.ndarray,
    b_ecsi_matrix: np.ndarray,
    result_name: str = "res",
):
    results = pd.DataFrame()

    for i, e in enumerate(element_incidence_matrix):
        p_element_global = n_ecsi_matrix.T @ p_knots_global[e]
        det_j = b_ecsi_matrix.T @ p_knots_global[e]
        element_displacements = knot_displacements[e]
        res = np.array([col @ element_displacements for col in b_ecsi_matrix.T]) / det_j
        results = pd.concat(
            [results, pd.DataFrame({X_COORD: p_element_global, result_name: res})],
        )

    return results


def calc_error_squared(
    results: pd.DataFrame,
    analytical_solution: Callable[
        [
            np.ndarray,
        ],
        np.ndarray,
    ],
):
    results[ANALYTICAL_DISPLACEMENT] = results[X_COORD].apply(analytical_solution)
    results[ERROR_SQUARED] = (
        results[ANALYTICAL_DISPLACEMENT] - results[NUM_DISPLACEMENT]
    ) ** 2
    return results


def calc_l2_error_norm(results: pd.DataFrame):
    return np.trapz(results[ERROR_SQUARED], results[X_COORD]) ** 0.5
