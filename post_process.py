import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Callable
from bar_1d import EnergyNormsAndErrors

from nomeclature import (
    ANALYTICAL_DISPLACEMENT,
    ANALYTICAL_STRAIN,
    ERROR_DERIVATIVE_SQUARED,
    ERROR_SQUARED,
    NUM_DISPLACEMENT,
    X_COORD,
    STRAIN,
    NUM_STRAIN,
)


def calc_approx_value(
    x_knots_global: npt.NDArray[np.float64],
    element_incidence_matrix: npt.NDArray[np.float64],
    knot_displacements: npt.NDArray[np.float64],
    ecsi_matrix: npt.NDArray[np.float64],
    ecsi_calc_pts: npt.NDArray[np.float64],
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
    p_knots_global: npt.NDArray[np.float64],
    element_incidence_matrix: npt.NDArray[np.float64],
    knot_displacements: npt.NDArray[np.float64],
    esci_matrix: npt.NDArray[np.float64],
    factor: float = 1,
    result_name: str = "res",
):
    results = pd.DataFrame()

    for i, e in enumerate(element_incidence_matrix):
        p_element_global = esci_matrix.T @ p_knots_global[e]
        element_displacements = knot_displacements[e]
        res = np.array([factor * col @ element_displacements for col in esci_matrix.T])
        results = pd.concat(
            [results, pd.DataFrame({X_COORD: p_element_global, result_name: res})],
        )

    return results


def calc_strain_variable_det_j(
    p_knots_global: npt.NDArray[np.float64],
    element_incidence_matrix: npt.NDArray[np.float64],
    knot_displacements: npt.NDArray[np.float64],
    n_ecsi_matrix: npt.NDArray[np.float64],
    b_ecsi_matrix: npt.NDArray[np.float64],
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
            npt.NDArray[np.float64],
        ],
        npt.NDArray[np.float64],
    ],
    derivative_analytical_solution: Callable[
        [
            npt.NDArray[np.float64],
        ],
        npt.NDArray[np.float64],
    ],
):
    results[ANALYTICAL_DISPLACEMENT] = results[X_COORD].apply(analytical_solution)
    results[ANALYTICAL_STRAIN] = results[X_COORD].apply(derivative_analytical_solution)
    results[ERROR_SQUARED] = (
        results[ANALYTICAL_DISPLACEMENT] - results[NUM_DISPLACEMENT]
    ) ** 2
    results[ERROR_DERIVATIVE_SQUARED] = (
        results[ANALYTICAL_STRAIN] - results[NUM_STRAIN]
    ) ** 2
    return results


def calc_error_of_derivative_squared(
    results: pd.DataFrame,
    analytical_solution_derivative: Callable[
        [
            npt.NDArray[np.float64],
        ],
        npt.NDArray[np.float64],
    ],
):
    results[ANALYTICAL_STRAIN] = results[X_COORD].apply(analytical_solution_derivative)
    results[ERROR_DERIVATIVE_SQUARED] = (
        results[ANALYTICAL_STRAIN] - results[STRAIN]
    ) ** 2
    return results


def calc_h1_error_norm(results: pd.DataFrame):
    return np.trapz(results[ERROR_DERIVATIVE_SQUARED], results[X_COORD]) ** 0.5


def calc_l2_error_norm(results: pd.DataFrame):
    return np.trapz(results[ERROR_SQUARED], results[X_COORD]) ** 0.5


# def calc_l2_h1_error_norms_simplified(
#     analytical_solution_function: Callable[
#         [npt.NDArray[np.float64]], npt.NDArray[np.float64]
#     ],
#     analytical_derivative_solution_function: Callable[
#         [npt.NDArray[np.float64]], npt.NDArray[np.float64]
#     ],
#     p_knots_global_coords: npt.NDArray[np.float64],
#     knot_displacements: npt.NDArray[np.float64],
#     intergration_weights: npt.NDArray[np.float64],
#     n_ecsi: npt.NDArray[np.float64],
#     b_ecsi: npt.NDArray[np.float64],
#     incidence_matrix: npt.NDArray[np.float64],
#     det_j: float,
# ):
#     l2_error_norm = 0
#     l2_sol_norm = 0
#     h1_error_norm = 0
#     h1_sol_norm = 0

#     for e in incidence_matrix:
#         element_displacements = knot_displacements[e]
#         p_element = n_ecsi.T @ p_init_global_coords[e]
#         analitycal_displacement = analytical_solution_function(p_element)
#         analytical_derivative = analytical_derivative_solution_function(p_element)
#         num_displacement = n_ecsi.T @ element_displacements
#         num_derivative = b_ecsi.T @ element_displacements / det_j
#         l2_error_norm += np.sum(
#             (analitycal_displacement - num_displacement) ** 2
#             * intergration_weights
#             * det_j
#         )
#         l2_sol_norm += np.sum(num_displacement**2 * det_j * intergration_weights)
#         h1_error_norm += np.sum(
#             (analytical_derivative - num_derivative) ** 2 * det_j * intergration_weights
#         )
#         h1_sol_norm += np.sum(num_derivative**2 * det_j * intergration_weights)
#     return EnergyNormsAndErrors(
#         l2_error_norm=l2_error_norm**0.5,
#         l2_sol_norm=l2_sol_norm**0.5,
#         h1_error_norm=h1_error_norm**0.5,
#         h1_sol_norm=h1_sol_norm**0.5,
#     )


# def calc_l2_h1_error_norms(
#     analytical_solution_function: Callable[
#         [npt.NDArray[np.float64]], npt.NDArray[np.float64]
#     ],
#     analytical_derivative_solution_function: Callable[
#         [npt.NDArray[np.float64]], npt.NDArray[np.float64]
#     ],
#     p_init_global_coords: npt.NDArray[np.float64],
#     knot_displacements: npt.NDArray[np.float64],
#     intergration_weights: npt.NDArray[np.float64],
#     n_ecsi: npt.NDArray[np.float64],
#     b_ecsi: npt.NDArray[np.float64],
#     incidence_matrix: npt.NDArray[np.float64],
#     det_j: float,
# ):
#     l2_error_norm = 0
#     l2_sol_norm = 0
#     h1_error_norm = 0
#     h1_sol_norm = 0

#     for e in incidence_matrix:
#         element_displacements = knot_displacements[e]
#         p_element = n_ecsi.T @ p_init_global_coords[e]
#         analytical_displacement = analytical_solution_function(p_element)
#         analytical_derivative = analytical_derivative_solution_function(p_element)
#         num_displacement = n_ecsi.T @ element_displacements
#         num_derivative = b_ecsi.T @ element_displacements / det_j
#         h1_sol_norm += np.sum(num_derivative**2 * det_j * intergration_weights)
#         l2_sol_norm += np.sum(num_displacement**2 * det_j * intergration_weights)

#         for analytical_sol, analytical_strain, num_sol, num_strain, weight in zip(
#             analytical_displacement,
#             analytical_derivative,
#             num_displacement,
#             num_derivative,
#             intergration_weights,
#         ):
#             h1_error_norm += (analytical_strain - num_strain) ** 2 * det_j * weight
#             l2_error_norm += (analytical_sol - num_sol) ** 2 * weight * det_j
#             pass

#     return EnergyNormsAndErrors(
#         l2_error_norm=l2_error_norm**0.5,
#         l2_sol_norm=l2_sol_norm**0.5,
#         h1_error_norm=h1_error_norm**0.5,
#         h1_sol_norm=h1_sol_norm**0.5,
#     )


def calc_l2_h1_error_norms(
    analytical_solution_function: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    analytical_derivative_solution_function: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    p_init_global_coords: npt.NDArray[np.float64],
    knot_displacements: npt.NDArray[np.float64],
    intergration_weights: npt.NDArray[np.float64],
    n_ecsi: npt.NDArray[np.float64],
    b_ecsi: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    det_j: float,
):
    l2_error_norm = 0
    l2_sol_norm = 0
    h1_error_norm = 0
    h1_sol_norm = 0

    for e in incidence_matrix:
        element_displacements = knot_displacements[e]
        p_element = n_ecsi.T @ p_init_global_coords[e]
        analitycal_displacement = analytical_solution_function(p_element)
        analytical_derivative = analytical_derivative_solution_function(p_element)
        num_displacement = n_ecsi.T @ element_displacements
        num_derivative = b_ecsi.T @ element_displacements / det_j
        l2_error_norm += np.sum(
            (analitycal_displacement - num_displacement) ** 2
            * intergration_weights
            * det_j
        )
        l2_sol_norm += np.sum(num_displacement**2 * det_j * intergration_weights)
        h1_error_norm += np.sum(
            (analytical_derivative - num_derivative) ** 2 * det_j * intergration_weights
        )
        h1_sol_norm += np.sum(num_derivative**2 * det_j * intergration_weights)

    return EnergyNormsAndErrors(
        l2_error_norm=l2_error_norm**0.5,
        l2_sol_norm=l2_sol_norm**0.5,
        h1_error_norm=h1_error_norm**0.5,
        h1_sol_norm=h1_sol_norm**0.5,
    )
