from functools import partial
from typing import Callable
import pytest as pt
import numpy as np
import numpy.typing as npt
from c0_basis import (
    c0_bar,
    calc_incidence_matrix,
    calc_p_knots_global,
    calc_ecsi_placement_coords_equal_dist,
)
from polynomials import lagrange_poli
from pre_process import (
    calc_element_stiffness_matrix,
    calc_load_vector,
    compose_global_matrix,
)


@pt.mark.parametrize(
    "length, n_elements, esci_internal_knots_coords, expected_knots",
    [
        [1, 5, np.array([-1, 1]), np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])],
        [1, 2, np.array([-1, 1, 0]), np.array([0.0, 0.5, 1.0, 0.25, 0.75])],
    ],
)
def test_x_knots_global(
    length: float,
    n_elements: int,
    esci_internal_knots_coords: npt.NDArray[np.float64],
    expected_knots: npt.NDArray[np.float64],
):
    assert calc_p_knots_global(
        length=length,
        n_elements=n_elements,
        esci_placement_coords=esci_internal_knots_coords,
    ) == pt.approx(expected_knots)


@pt.mark.parametrize(
    "degree, expected_knots",
    (
        (1, np.array((-1.0, 1.0))),
        (2, np.array((-1.0, 1.0, 0.0))),
        (4, np.array((-1.0, 1.0, -0.5, 0.0, 0.5))),
    ),
)
def test_x_knots_local(degree: int, expected_knots: np.array):
    assert calc_ecsi_placement_coords_equal_dist(degree) == pt.approx(expected_knots)


@pt.mark.parametrize(
    "young_modulus, section_area, length, poly_degree, n_elements, load_function, x_knots_local_function, expected_stiffness_matrix, expected_load_vector, expected_knot_displacements, expected_element_stiffness_matrix, expected_incidence_matrix",
    (
        (
            100000000000.0,
            0.0001,
            1.0,
            3,
            4,
            lambda x: 1000 * np.sin(np.pi / 2 * x),
            calc_ecsi_placement_coords_equal_dist,
            np.array(
                [
                    [
                        1.48e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        -1.30e07,
                        2.96e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        -1.30e07,
                        2.96e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        -1.30e07,
                        2.96e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        -1.89e08,
                        5.40e07,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -1.30e07,
                        1.48e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                    ],
                    [
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                    ],
                ]
            ),
            np.array(
                [
                    1.61433421,
                    23.7964582,
                    43.97012135,
                    57.44973212,
                    31.09157098,
                    7.42861349,
                    29.0100683,
                    42.73635623,
                    61.03223018,
                    71.53787614,
                    83.76278827,
                    89.44840289,
                    93.74122116,
                ]
            ),
            np.array(
                [
                    0.00000000e00,
                    1.55095752e-05,
                    2.86579581e-05,
                    3.74434268e-05,
                    4.05284731e-05,
                    5.29001438e-06,
                    1.04894770e-05,
                    2.02641490e-05,
                    2.46720405e-05,
                    3.21532506e-05,
                    3.50985096e-05,
                    3.91473113e-05,
                    4.01815487e-05,
                ]
            ),
            np.array(
                [
                    [1.48e08, -1.30e07, -1.89e08, 5.40e07],
                    [-1.30e07, 1.48e08, 5.40e07, -1.89e08],
                    [-1.89e08, 5.40e07, 4.32e08, -2.97e08],
                    [5.40e07, -1.89e08, -2.97e08, 4.32e08],
                ]
            ),
            np.array([[0, 1, 5, 6], [1, 2, 7, 8], [2, 3, 9, 10], [3, 4, 11, 12]]),
        ),
    ),
)
def test_c0_bar(
    young_modulus: float,
    section_area: float,
    length: float,
    poly_degree: int,
    n_elements: int,
    load_function: Callable[
        [
            float,
        ],
        float,
    ],
    x_knots_local_function: Callable[
        [
            int,
        ],
        float,
    ],
    expected_stiffness_matrix: np.array,
    expected_load_vector: np.array,
    expected_knot_displacements: np.array,
    expected_element_stiffness_matrix: np.array,
    expected_incidence_matrix: np.array,
):
    res = c0_bar(
        young_modulus=young_modulus,
        section_area=section_area,
        length=length,
        degree=poly_degree,
        n_elements=n_elements,
        load_function=load_function,
        ecsi_placement_coords_function=x_knots_local_function,
    )
    assert res.element_stiffness_matrix == pt.approx(expected_element_stiffness_matrix)
    assert res.incidence_matrix == pt.approx(expected_incidence_matrix)
    assert res.global_stiffness_matrix == pt.approx(expected_stiffness_matrix)
    assert res.load_vector == pt.approx(expected_load_vector, rel=0.01)
    assert res.knots_displacements == pt.approx(expected_knot_displacements, rel=1e-5)


@pt.mark.parametrize(
    "stiffness, b_esci_matrix, int_weights, det_j, expected_matrix",
    (
        (
            10000000.0,
            np.array(
                [
                    [-1.82142125, 0.0625, -0.07857875],
                    [0.07857875, -0.0625, 1.82142125],
                    [2.22142125, -1.6875, 0.47857875],
                    [-0.47857875, 1.6875, -2.22142125],
                ]
            ),
            np.array([0.55555556, 0.88888889, 0.55555556]),
            0.125,
            np.array(
                [
                    [1.48e08, -1.30e07, -1.89e08, 5.40e07],
                    [-1.30e07, 1.48e08, 5.40e07, -1.89e08],
                    [-1.89e08, 5.40e07, 4.32e08, -2.97e08],
                    [5.40e07, -1.89e08, -2.97e08, 4.32e08],
                ]
            ),
        ),
    ),
)
def test_element_stiffness_matrix(
    stiffness: float,
    b_esci_matrix: np.array,
    int_weights: np.array,
    det_j: float,
    expected_matrix: np.array,
):
    assert calc_element_stiffness_matrix(
        stiffness=stiffness,
        b_ecsi_matrix=b_esci_matrix,
        int_weights=int_weights,
        det_j=det_j,
    ) == pt.approx(expected_matrix)


@pt.mark.parametrize(
    "n_elements, degree, expected_matrix",
    (
        (2, 1, np.array([[0, 1], [1, 2]])),
        (2, 3, np.array([[0, 1, 3, 4], [1, 2, 5, 6]])),
    ),
)
def test_incidence_matrix(n_elements: int, degree: int, expected_matrix: np.array):
    assert (
        calc_incidence_matrix(n_elements=n_elements, degree=degree) == expected_matrix
    ).all()


@pt.mark.parametrize(
    "element_stiffeness_matrix, incidence_matrix, expected_global_stiffeness",
    (
        (
            np.array(
                [
                    [1.48e08, -1.30e07, -1.89e08, 5.40e07],
                    [-1.30e07, 1.48e08, 5.40e07, -1.89e08],
                    [-1.89e08, 5.40e07, 4.32e08, -2.97e08],
                    [5.40e07, -1.89e08, -2.97e08, 4.32e08],
                ]
            ),
            np.array([[0, 1, 5, 6], [1, 2, 7, 8], [2, 3, 9, 10], [3, 4, 11, 12]]),
            np.array(
                [
                    [
                        1.48e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        -1.30e07,
                        2.96e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        -1.30e07,
                        2.96e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        -1.30e07,
                        2.96e08,
                        -1.30e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        -1.89e08,
                        5.40e07,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -1.30e07,
                        1.48e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                    ],
                    [
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                        0.00e00,
                        0.00e00,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -1.89e08,
                        5.40e07,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        4.32e08,
                        -2.97e08,
                    ],
                    [
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        5.40e07,
                        -1.89e08,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        0.00e00,
                        -2.97e08,
                        4.32e08,
                    ],
                ]
            ),
        ),
    ),
)
def test_assembly_global_stiffeness(
    element_stiffeness_matrix: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    expected_global_stiffeness: npt.NDArray[np.float64],
):
    assert compose_global_matrix(
        element_matrix=element_stiffeness_matrix,
        incidence_matrix=incidence_matrix,
    ) == pt.approx(expected_global_stiffeness)


@pt.mark.parametrize(
    "x_knots, element_incidence_matrix, test_function_local, load_function, intorder, det_j, expected_load_vector",
    (
        (
            np.array(
                [
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    0.08333333,
                    0.16666667,
                    0.33333333,
                    0.41666667,
                    0.58333333,
                    0.66666667,
                    0.83333333,
                    0.91666667,
                ]
            ),
            np.array([[0, 1, 5, 6], [1, 2, 7, 8], [2, 3, 9, 10], [3, 4, 11, 12]]),
            partial(
                lagrange_poli,
                degree=3,
                placement_pts_coords=np.array([-1.0, 1.0, -0.33333333, 0.33333333]),
            ),
            lambda x: 1000 * np.sin(np.pi / 2 * x),
            2 * (3 - 1),
            0.125,
            np.array(
                [
                    1.61433421,
                    23.7964582,
                    43.97012135,
                    57.44973212,
                    31.09157098,
                    7.42861349,
                    29.0100683,
                    42.73635623,
                    61.03223018,
                    71.53787614,
                    83.76278827,
                    89.44840289,
                    93.74122116,
                ]
            ),
        ),
    ),
)
def test_load_vector(
    x_knots: npt.NDArray[np.float64],
    element_incidence_matrix: npt.NDArray[np.float64],
    test_function_local: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    load_function: Callable[[npt.NDArray[np.float64]], float],
    intorder: int,
    det_j: float,
    expected_load_vector: npt.NDArray[np.float64],
):
    assert calc_load_vector(
        x_knots=x_knots,
        incidence_matrix=element_incidence_matrix,
        test_function_local=test_function_local,
        load_function=load_function,
        intorder=intorder,
        det_j=det_j,
    ) == pt.approx(expected_load_vector)
