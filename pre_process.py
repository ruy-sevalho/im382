from typing import Callable
import numpy as np
import numpy.typing as npt

from polynomials import IntegrationTypes, get_points_weights


def calc_element_matrix(
    factor: float, ecsi_matrix: float, integration_weights: npt.NDArray[np.float64]
):
    return (
        np.sum(
            np.outer(b_col, b_col) * w
            for b_col, w in zip(ecsi_matrix.T, integration_weights)
        )
        * factor
    )


def calc_element_stiffness_matrix(
    stiffness: float,
    b_ecsi_matrix: npt.NDArray[np.float64],
    int_weights: npt.NDArray[np.float64],
    det_j: float,
):
    return calc_element_matrix(
        factor=stiffness / det_j,
        ecsi_matrix=b_ecsi_matrix,
        integration_weights=int_weights,
    )


def compose_global_matrix(
    element_matrix: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
):
    n_knots = incidence_matrix[-1, -1] + 1
    stiffness_matrix = np.zeros((n_knots, n_knots))
    for row in incidence_matrix:
        stiffness_matrix[
            row[:, np.newaxis],
            row[np.newaxis, :],
        ] += element_matrix
    return stiffness_matrix


def calc_load_vector(
    x_knots: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    test_function_local: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    load_function: Callable[[float], float],
    intorder: int,
    det_j: float,
):
    ecsi_local_int_pts, weight_int_pts = get_points_weights(
        0, 0, intorder, IntegrationTypes.GJ, "x"
    )
    n_esci_matrix = test_function_local(calc_pts_coords=ecsi_local_int_pts)
    global_load_vector = np.zeros(incidence_matrix[-1, -1] + 1)
    for i, element_incidence in enumerate(incidence_matrix):
        load_function_at_x = np.array(
            [
                load_function(x)
                for x in np.interp(ecsi_local_int_pts, [-1, 1], x_knots[i : i + 2])
            ]
        )
        load_vector = det_j * np.array(
            [np.sum(row * weight_int_pts * load_function_at_x) for row in n_esci_matrix]
        )
        global_load_vector[element_incidence] += load_vector
    return global_load_vector


def calc_load_vector_isop(
    collocation_pts: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    n_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    b_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    load_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    intorder: int,
):
    ecsi_local_int_pts, weight_int_pts = get_points_weights(
        0, 0, intorder, IntegrationTypes.GJ, "x"
    )
    n_ecsi = n_ecsi_function(calc_pts_coords=ecsi_local_int_pts)
    b_ecsi = b_ecsi_function(calc_pts_coords=ecsi_local_int_pts)
    global_load_vector = np.zeros(incidence_matrix[-1, -1] + 1)
    for i, element_incidence in enumerate(incidence_matrix):
        p_coords = n_ecsi.T @ collocation_pts[element_incidence]
        det_j = b_ecsi.T @ collocation_pts[element_incidence]
        load_at_x = load_function(p_coords)
        load_vector = (
            np.array([np.sum(row * weight_int_pts * load_at_x) for row in n_ecsi])
            * det_j
        )
        global_load_vector[element_incidence] += load_vector
    return global_load_vector


def calc_element_1D_jacobian(element_size: float):
    return element_size / 2


def calc_p_coords_at_int_pts(p_coords):
    return


def calc_external_load_vector(
    p_coords: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    n_ecsi: npt.NDArray[np.float64],
    integration_weights: npt.NDArray[np.float64],
    det_j: float,
    load_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
):
    load_vector = np.zeros(incidence_matrix[-1, -1] + 1)
    for e in incidence_matrix:
        load_vector[e] += (
            n_ecsi.T @ load_function(p=p_coords[e]) * integration_weights * det_j
        )
    return load_vector
