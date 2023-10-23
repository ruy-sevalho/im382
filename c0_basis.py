from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache, partial
from typing import Callable
import numpy as np
import pandas as pd
from bar_1d import BarInput
from nomeclature import NUM_DISPLACEMENT, NUM_STRAIN, X_COORD

from polynomials import (
    IntegrationTypes,
    d_lagrange_poli,
    get_points_weights,
    lagrange_poli,
    quadrature_gauss_jacobi_n_pts,
)
from post_processing import calc_approx_value, calc_error_squared, calc_l2_error_norm


@dataclass
class C0BarResults:
    det_j: float
    x_knots_local: np.array
    x_knots_global: np.array
    element_stiffness_matrix: np.array
    incidence_matrix: np.array
    global_stiffness_matrix: np.array
    load_vector: np.array
    knots_displacements: np.array


def c0_bar(
    young_modulus: float,
    section_area: float,
    length: float,
    degree: int,
    n_elements: int,
    load_function: Callable[[float], float],
):
    """Returns the stiffness matrix, the load vector and knot displacement of a axially loaded bar"""
    stiffness = young_modulus * section_area
    det_j = calc_element_1D_jacobian(length / n_elements)
    x_knots_global = calc_x_knots_global(
        length=length, degree=degree, n_elements=n_elements
    )
    n_knots = x_knots_global.shape[0]
    x_knots_local = calc_x_knots_local(degree=degree)

    # relative to numerical integration of approx solution dericative to calculate de stuiffness matrix
    # intorder is corrected since we are intgreting phi1' * phi1' giving a 2*(P-1) order polynomial
    integration_points, integration_weights = get_points_weights(
        intorder=2 * (degree - 1),
    )
    b_esci_matrix_at_int_pts = d_lagrange_poli(
        degree=degree,
        calc_pts_coords=integration_points,
        placement_pts_coords=x_knots_local,
    )
    element_stiffness_matrix = calc_element_stiffness_matrix(
        stiffness=stiffness,
        b_esci_matrix=b_esci_matrix_at_int_pts,
        int_weights=integration_weights,
        det_j=det_j,
    )
    # end of numerical integration of stiffness matrix

    incidence_matrix = calc_incidence_matrix(n_elements=n_elements, degree=degree)
    global_stiffness_matrix = compose_global_matrix(
        element_stiffness_matrix=element_stiffness_matrix,
        incindence_matrix=incidence_matrix,
    )
    # get_point_degree intorder adjusted to better precision in numerical integration, since load function is trigonometric
    load_vector = calc_load_vector(
        x_knots=x_knots_global,
        element_incidence_matrix=incidence_matrix,
        test_function_local=partial(
            lagrange_poli,
            degree=degree,
            placement_pts_coords=x_knots_local,
        ),
        load_function=load_function,
        intorder=2 * degree + 2,
        det_j=det_j,
    )

    knots_displacements = np.zeros(n_knots)
    free_knots = np.arange(1, n_knots)
    knots_displacements[free_knots] = np.linalg.solve(
        global_stiffness_matrix[free_knots[:, np.newaxis], free_knots[np.newaxis, :]],
        load_vector[free_knots],
    )
    return C0BarResults(
        det_j=det_j,
        x_knots_local=x_knots_local,
        x_knots_global=x_knots_global,
        element_stiffness_matrix=element_stiffness_matrix,
        incidence_matrix=incidence_matrix,
        global_stiffness_matrix=global_stiffness_matrix,
        load_vector=load_vector,
        knots_displacements=knots_displacements,
    )


def calc_x_knots_global(
    length: float,
    degree: int,
    n_elements: int,
):
    n_knots_vertices = n_elements + 1

    x_knots = np.linspace(0, length, n_knots_vertices)
    for n in range(n_elements):
        x_knots = np.concatenate(
            (
                x_knots,
                np.linspace(x_knots[n], x_knots[n + 1], degree + 1)[1:-1],
            )
        )
    return x_knots


def calc_x_knots_local(degree: int) -> np.array:
    return np.concatenate(((-1.0, 1.0), np.linspace(-1, 1, degree + 1)[1:-1]))


def calc_element_stiffness_matrix(
    stiffness: float, b_esci_matrix: np.array, int_weights: np.array, det_j: float
):
    return (
        np.sum(
            np.outer(b_col, b_col) * w for b_col, w in zip(b_esci_matrix.T, int_weights)
        )
        * stiffness
        / det_j
    )


def compose_global_matrix(
    element_stiffness_matrix: np.array, incindence_matrix: np.array
):
    n_knots = incindence_matrix[-1, -1] + 1
    stiffness_matrix = np.zeros((n_knots, n_knots))
    for row in incindence_matrix:
        stiffness_matrix[
            row[:, np.newaxis],
            row[np.newaxis, :],
        ] += element_stiffness_matrix
    return stiffness_matrix


def calc_incidence_matrix(n_elements: int, degree: int):
    vertices = np.array([np.arange(i, i + 2) for i in range(n_elements)])
    interior_knots = np.reshape(
        np.arange(n_elements * (degree - 1)), (n_elements, degree - 1)
    )
    if interior_knots.any():
        return np.concatenate((vertices, interior_knots + n_elements + 1), axis=1)
    return vertices


def calc_load_vector(
    x_knots: np.array,
    element_incidence_matrix: np.array,
    test_function_local: Callable[[np.array], np.array],
    load_function: Callable[[float], float],
    intorder: int,
    det_j: float,
):
    ecsi_local_int_pts, weight_int_pts = get_points_weights(
        0, 0, intorder, IntegrationTypes.GJ, "x"
    )
    n_esci_matrix = test_function_local(calc_pts_coords=ecsi_local_int_pts)
    global_load_vector = np.zeros(element_incidence_matrix[-1, -1] + 1)
    for element_incidence in element_incidence_matrix:
        load_function_at_x = np.array(
            [
                load_function(x)
                for x in np.interp(
                    ecsi_local_int_pts, [-1, 1], x_knots[element_incidence[:2]]
                )
            ]
        )
        load_vector = det_j * np.array(
            [np.sum(row * weight_int_pts * load_function_at_x) for row in n_esci_matrix]
        )
        global_load_vector[element_incidence] += load_vector
    return global_load_vector


def calc_element_1D_jacobian(element_size: float):
    return element_size / 2


def calc_strain(
    degree: int,
    det_j: float,
    x_knots_global: np.array,
    x_knots_local: np.array,
    element_incidence_matrix: np.array,
    knot_displacements: np.array,
    ecsi_calc_pts: np.array = None,
):
    ecsi_calc_pts = ecsi_calc_pts or np.linspace(-1, 1, 21)
    b_ecsi = d_lagrange_poli(
        degree=degree, calc_pts_coords=ecsi_calc_pts, placement_pts_coords=x_knots_local
    )

    # results = pd.DataFrame()

    # for e in element_incidence_matrix:
    #     x_element_global = np.interp(ecsi_calc_pts, [-1, 1], x_knots_global[e[:2]])
    #     element_displacements = knot_displacements[e]
    #     strains = np.array(
    #         [1 / det_j * col @ element_displacements for col in b_ecsi.T]
    #     )
    #     results = pd.concat(
    #         [results, pd.DataFrame({"x": x_element_global, "strain": strains})]
    #     )
    results = calc_approx_value(
        x_knots_global=x_knots_global,
        element_incidence_matrix=element_incidence_matrix,
        knot_displacements=knot_displacements,
        esci_matrix_=b_ecsi,
        ecsi_calc_pts=ecsi_calc_pts,
        factor=1 / det_j,
        result_name=NUM_STRAIN,
    )
    return results


def calc_displacements(
    degree: int,
    x_knots_global: np.array,
    x_knots_local: np.array,
    element_incidence_matrix: np.array,
    knot_displacements: np.array,
    ecsi_calc_pts: np.array = None,
):
    ecsi_calc_pts = ecsi_calc_pts or np.linspace(-1, 1, 21)
    n_ecsi = lagrange_poli(
        degree=degree, calc_pts_coords=ecsi_calc_pts, placement_pts_coords=x_knots_local
    )
    results = pd.DataFrame()
    for e in element_incidence_matrix:
        x_element_global = np.interp(ecsi_calc_pts, [-1, 1], x_knots_global[e[:2]])
        element_displacements = knot_displacements[e]
        displacement = np.array([col @ element_displacements for col in n_ecsi.T])
        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {X_COORD: x_element_global, NUM_DISPLACEMENT: displacement}
                ),
            ]
        )

    return results


@dataclass
class C0BarAnalysis:
    input: BarInput
    displacement_analytical: Callable[[float], float]

    @cached_property
    def bar_result(self):
        return c0_bar(**asdict(self.input))

    @cached_property
    def results(self):
        results = calc_displacements(
            degree=self.input.degree,
            x_knots_global=self.bar_result.x_knots_global,
            x_knots_local=self.bar_result.x_knots_local,
            element_incidence_matrix=self.bar_result.incidence_matrix,
            knot_displacements=self.bar_result.knots_displacements,
        )
        results = pd.concat(
            [
                results,
                calc_strain(
                    degree=self.input.degree,
                    det_j=self.bar_result.det_j,
                    x_knots_global=self.bar_result.x_knots_global,
                    x_knots_local=self.bar_result.x_knots_local,
                    element_incidence_matrix=self.bar_result.incidence_matrix,
                    knot_displacements=self.bar_result.knots_displacements,
                )[NUM_STRAIN],
            ],
            axis=1,
        )

        return calc_error_squared(results, self.displacement_analytical)

    @cached_property
    def l2_error(self):
        return calc_l2_error_norm(self.results)

    @cached_property
    def energy_norm_aprox_sol(self):
        return (
            self.bar_result.knots_displacements
            @ self.bar_result.global_stiffness_matrix
            @ self.bar_result.knots_displacements
        ) ** 0.5
