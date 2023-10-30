from dataclasses import asdict, dataclass
from functools import cached_property, partial
from typing import Callable

import numpy as np
import pandas as pd
from bar_1d import BarInput
from polynomials import c1_basis

from c0_basis import (
    calc_element_1D_jacobian,
    calc_element_stiffness_matrix,
    compose_global_matrix,
    calc_load_vector,
)
from nomeclature import NUM_DISPLACEMENT, NUM_STRAIN
from polynomials import get_points_weights
from post_processing import calc_approx_value, calc_error_squared, calc_l2_error_norm


@dataclass
class C1BarResults:
    det_j: float
    x_knots_global: np.ndarray
    n_degrees_freedom: int
    element_stiffness_matrix: np.ndarray
    incidence_matrix: np.ndarray
    global_stiffness_matrix: np.ndarray
    load_vector: np.ndarray
    knots_displacements: np.ndarray


def c1_bar(
    young_modulus: float,
    section_area: float,
    length: float,
    degree: int,
    n_elements: int,
    load_function: Callable[[float], float],
):
    stiffness = young_modulus * section_area
    x_knots_global = calc_x_knots_global(length=length, n_elements=n_elements)
    det_j = calc_element_1D_jacobian(length / n_elements)

    n_knots_v = n_elements * 2 + 2
    n_degrees_total = n_knots_v + n_elements * (degree - 3)

    # relative to numerical integration of approx solution dericative to calculate de stuiffness matrix
    # intorder is corrected since we are intgreting phi1' * phi1' giving a 2*(P-1) order polynomial
    integration_points, integration_weights = get_points_weights(
        intorder=2 * (degree - 1),
    )
    # Hermite ploynomials and is derivatives calculated at the integration points
    (
        hs,
        d1hs,
        d2hs,
        d3hs,
    ) = c1_basis(
        degree=degree,
        calc_pts_coords=integration_points,
        element_size=length / n_elements,
    )

    element_stiffness_matrix = calc_element_stiffness_matrix(
        stiffness=stiffness,
        b_esci_matrix=d1hs,
        int_weights=integration_weights,
        det_j=det_j,
    )
    # End of numeric integratation - stiffness matrix

    incidence_matrix = calc_incidence_matrix(n_elements=n_elements, degree=degree)
    global_stiffness_matrix = compose_global_matrix(
        element_stiffness_matrix=element_stiffness_matrix,
        incindence_matrix=incidence_matrix,
    )
    load_vector = calc_load_vector(
        x_knots=x_knots_global,
        incidence_matrix=incidence_matrix,
        test_function_local=partial(
            c1_basis,
            degree=degree,
            element_size=length / n_elements,
            return_derivative_order=0,
        ),
        load_function=load_function,
        intorder=2 * degree + 2,
        det_j=det_j,
    )

    knots_displcaments = np.zeros(n_degrees_total)
    # first displacement = 0
    free_degrees = np.arange(1, n_degrees_total)
    # last strain = 0
    free_degrees = free_degrees[free_degrees != n_knots_v - 1]

    knots_displcaments[free_degrees] = np.linalg.solve(
        global_stiffness_matrix[
            free_degrees[:, np.newaxis], free_degrees[np.newaxis, :]
        ],
        load_vector[free_degrees],
    )

    # Calculo da reacao de apoio
    reaction = np.dot(global_stiffness_matrix[0], knots_displcaments) - load_vector[0]

    return C1BarResults(
        det_j=det_j,
        x_knots_global=x_knots_global,
        n_degrees_freedom=n_degrees_total,
        element_stiffness_matrix=element_stiffness_matrix,
        incidence_matrix=incidence_matrix,
        global_stiffness_matrix=global_stiffness_matrix,
        load_vector=load_vector,
        knots_displacements=knots_displcaments,
    )


def calc_x_knots_global(
    length: float,
    n_elements: int,
):
    return np.linspace(0, length, int((n_elements * 2 + 2) / 2))


def calc_x_knots_global_complete(
    length: float,
    n_elements: int,
    degree: int,
):
    x_knots = calc_x_knots_global(length=length, n_elements=n_elements)
    x_knots = np.concatenate(
        [x_knots[i], 0, x_knots[i+1], 0 for i in range(len(x_knots))]
    )
    if degree > 3:
        x_knots = np.concatenate([x_knots, np.zeros((degree-3)*n_elements)])
    return x_knots    
    


def calc_incidence_matrix(n_elements: int, degree: int):
    incidence_matrix = np.array(
        [np.arange(2 * i, 2 * i + 4) for i in range(n_elements)]
    )
    if degree > 3:
        incidence_matrix = np.concatenate(
            [
                incidence_matrix,
                np.reshape(
                    np.arange(n_elements * (degree - 3)),
                    (n_elements, degree - 3),
                )
                + n_elements * 2
                + 2,
            ],
            axis=1,
        )

    return incidence_matrix


@dataclass
class C1BarAnalysis:
    input: BarInput
    displacement_analytical: Callable[[float], float]

    @cached_property
    def bar_result(self):
        return c1_bar(**asdict(self.input))

    @cached_property
    def results(self):
        esci_calc_pts = np.linspace(-1, 1, 21)
        n_ecsi, b_ecsi, _, _ = c1_basis(
            degree=self.input.degree,
            calc_pts_coords=esci_calc_pts,
            element_size=self.input.length / self.input.n_elements,
        )
        results = calc_approx_value(
            x_knots_global=self.bar_result.x_knots_global,
            element_incidence_matrix=self.bar_result.incidence_matrix,
            knot_displacements=self.bar_result.knots_displacements,
            esci_matrix_=n_ecsi,
            ecsi_calc_pts=esci_calc_pts,
            factor=1,
            result_name=NUM_DISPLACEMENT,
        )
        results = pd.concat(
            [
                results,
                calc_approx_value(
                    x_knots_global=self.bar_result.x_knots_global,
                    element_incidence_matrix=self.bar_result.incidence_matrix,
                    knot_displacements=self.bar_result.knots_displacements,
                    esci_matrix_=b_ecsi,
                    ecsi_calc_pts=esci_calc_pts,
                    factor=self.bar_result.det_j,
                    result_name=NUM_STRAIN,
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
