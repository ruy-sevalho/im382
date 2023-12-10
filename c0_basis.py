from dataclasses import asdict, dataclass
from functools import cached_property, partial
from typing import Callable
import numpy as np
import numpy.typing as npt
import pandas as pd
from bar_1d import BarInput, BarResults
from nomeclature import NUM_DISPLACEMENT, NUM_STRAIN, X_COORD

from polynomials import (
    IntegrationTypes,
    d_lagrange_poli,
    get_points_weights,
    get_points_weights_degree,
    lagrange_poli,
)
from post_process import (
    calc_approx_value,
)
from pre_process import (
    calc_element_1D_jacobian,
    calc_element_stiffness_matrix,
    calc_load_vector,
    compose_global_matrix,
)


@dataclass
class C0BarResults:
    det_j: float
    ecsi_knots_local: npt.NDArray[np.float64]
    p_knots_global: npt.NDArray[np.float64]
    n_degrees_freedom: int
    element_stiffness_matrix: npt.NDArray[np.float64]
    incidence_matrix: npt.NDArray[np.float64]
    global_stiffness_matrix: npt.NDArray[np.float64]
    load_vector: npt.NDArray[np.float64]
    knots_displacements: npt.NDArray[np.float64]
    reaction: float
    integration_weights: npt.NDArray[np.float64]
    n_ecsi: npt.NDArray[np.float64]
    b_ecsi: npt.NDArray[np.float64]
    p_knots_global_complete: npt.NDArray[np.float64]
    n_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    b_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def calc_incidence_matrix(n_elements: int, degree: int):
    vertices = np.array([np.arange(i, i + 2) for i in range(n_elements)])
    interior_knots = np.reshape(
        np.arange(n_elements * (degree - 1)), (n_elements, degree - 1)
    )
    if interior_knots.any():
        return np.concatenate((vertices, interior_knots + n_elements + 1), axis=1)
    return vertices


def c0_bar(
    young_modulus: float,
    section_area: float,
    length: float,
    degree: int,
    n_elements: int,
    load_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ecsi_placement_coords_function: Callable[[int], npt.NDArray[np.float64]],
):
    """Returns the stiffness matrix, the load vector and knot displacement of a axially loaded bar"""
    stiffness = young_modulus * section_area
    det_j = calc_element_1D_jacobian(length / n_elements)
    ecsi_placement_coords = ecsi_placement_coords_function(degree=degree)
    p_knots_global = calc_p_knots_global(
        length=length,
        n_elements=n_elements,
        esci_placement_coords=ecsi_placement_coords,
    )
    n_knots = p_knots_global.shape[0]

    # relative to numerical integration of approx solution dericative to calculate de stuiffness matrix
    # intorder is corrected since we are intgreting phi1' * phi1' giving a 2*(P-1) order polynomial
    integration_points, integration_weights = get_points_weights(
        intorder=2 * degree + 1,
    )
    b_esci_matrix_at_int_pts = d_lagrange_poli(
        degree=degree,
        calc_pts_coords=integration_points,
        placement_pts_coords=ecsi_placement_coords,
    )
    n_ecsi = lagrange_poli(
        calc_pts_coords=integration_points,
        degree=degree,
        placement_pts_coords=ecsi_placement_coords,
    )
    element_stiffness_matrix = calc_element_stiffness_matrix(
        stiffness=stiffness,
        b_ecsi_matrix=b_esci_matrix_at_int_pts,
        int_weights=integration_weights,
        det_j=det_j,
    )
    # end of numerical integration of stiffness matrix

    incidence_matrix = calc_incidence_matrix(n_elements=n_elements, degree=degree)
    global_stiffness_matrix = compose_global_matrix(
        element_matrix=element_stiffness_matrix,
        incidence_matrix=incidence_matrix,
    )
    # get_point_degree intorder adjusted to better precision in numerical integration, since load function is trigonometric
    load_vector = calc_load_vector(
        x_knots=p_knots_global,
        incidence_matrix=incidence_matrix,
        test_function_local=partial(
            lagrange_poli,
            degree=degree,
            placement_pts_coords=ecsi_placement_coords,
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
    reaction = 0
    return C0BarResults(
        det_j=det_j,
        ecsi_knots_local=ecsi_placement_coords,
        p_knots_global=p_knots_global,
        n_degrees_freedom=n_knots,
        element_stiffness_matrix=element_stiffness_matrix,
        incidence_matrix=incidence_matrix,
        global_stiffness_matrix=global_stiffness_matrix,
        load_vector=load_vector,
        knots_displacements=knots_displacements,
        reaction=reaction,
        integration_weights=integration_weights,
        n_ecsi=n_ecsi,
        b_ecsi=b_esci_matrix_at_int_pts,
        p_knots_global_complete=p_knots_global,
        n_ecsi_function=partial(
            lagrange_poli, degree=degree, placement_pts_coords=ecsi_placement_coords
        ),
        b_ecsi_function=partial(
            d_lagrange_poli,
            degree=degree,
            placement_pts_coords=ecsi_placement_coords,
        ),
    )


def calc_p_knots_global(
    length: float,
    n_elements: int,
    esci_placement_coords: npt.NDArray[np.float64],
):
    # eliminating the vertices knots
    esci_placement_coords = esci_placement_coords[2:]
    n_knots_vertices = n_elements + 1
    x_knots = np.linspace(0, length, n_knots_vertices)
    if len(esci_placement_coords):
        x_knots_internal = np.reshape(
            np.array(
                [
                    np.interp(esci_placement_coords, [-1, 1], x_knots[i : i + 2])
                    for i in range(n_elements)
                ]
            ),
            n_elements * len(esci_placement_coords),
        )
        x_knots = np.concatenate(
            [x_knots, x_knots_internal],
        )
    return x_knots


def calc_ecsi_placement_coords_equal_dist(
    degree: int,
) -> npt.NDArray[np.float64]:
    return np.concatenate(((-1.0, 1.0), np.linspace(-1, 1, degree + 1)[1:-1]))


def calc_ecsi_placement_coords_gauss_lobato(
    degree: int,
) -> npt.NDArray[np.float64]:
    pts, _ = get_points_weights_degree(
        intorder=degree + 1, type_int=IntegrationTypes.GLJ
    )
    pts = np.concatenate([[-1, 1], pts[1:-1]])
    return pts


@dataclass
class C0BarModel:
    inputs: BarInput
    ecsi_placement_coords_function: Callable[
        [int], npt.NDArray[np.float64]
    ] = calc_ecsi_placement_coords_equal_dist

    @cached_property
    def bar_result(self):
        return c0_bar(
            **asdict(self.inputs),
            ecsi_placement_coords_function=self.ecsi_placement_coords_function
        )

    @cached_property
    def result_dataframe(self):
        esci_calc_pts = np.linspace(-1, 1, 21)
        n_ecsi = lagrange_poli(
            degree=self.inputs.degree,
            calc_pts_coords=esci_calc_pts,
            placement_pts_coords=self.bar_result.ecsi_knots_local,
        )
        b_ecsi = d_lagrange_poli(
            degree=self.inputs.degree,
            calc_pts_coords=esci_calc_pts,
            placement_pts_coords=self.bar_result.ecsi_knots_local,
        )
        results = calc_approx_value(
            x_knots_global=self.bar_result.p_knots_global,
            element_incidence_matrix=self.bar_result.incidence_matrix,
            knot_displacements=self.bar_result.knots_displacements,
            ecsi_matrix=n_ecsi,
            ecsi_calc_pts=esci_calc_pts,
            factor=1,
            result_name=NUM_DISPLACEMENT,
        )
        results = (
            pd.concat(
                [
                    results,
                    calc_approx_value(
                        x_knots_global=self.bar_result.p_knots_global,
                        element_incidence_matrix=self.bar_result.incidence_matrix,
                        knot_displacements=self.bar_result.knots_displacements,
                        ecsi_matrix=b_ecsi,
                        ecsi_calc_pts=esci_calc_pts,
                        factor=1 / self.bar_result.det_j,
                        result_name=NUM_STRAIN,
                    )[NUM_STRAIN],
                ],
                axis=1,
            )
            .sort_values(X_COORD)
            .drop_duplicates(X_COORD)
        )

        return results
