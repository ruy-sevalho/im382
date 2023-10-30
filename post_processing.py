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
