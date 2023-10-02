import numpy as np

from polynomials import IntegrationTypes, get_points_weights


def c0_bar(
    young_modulus: float,
    section_area: float,
    length: float,
    poly_degree: int,
    n_elements: int,
):
    x_knots = x_knots_global(
        length=length, poly_degree=poly_degree, n_elements=n_elements
    )
    integration_points, integration_weights = get_points_weights(
        0, 0, poly_degree, IntegrationTypes.GJ, "x"
    )

    return


def x_knots_global(
    length: float,
    poly_degree: int,
    n_elements: int,
):
    n_knots_vertices = n_elements + 1
    # n_knots_internal = n_elements * (poly_degree)
    # n_knots_total = n_knots_vertices + n_knots_internal
    # element_size = length / n_elements

    nodal_coords = np.linspace(0, length, n_knots_vertices)
    for n in range(n_elements):
        nodal_coords = np.concatenate(
            (
                nodal_coords,
                np.linspace(nodal_coords[n], nodal_coords[n + 1], poly_degree + 1)[
                    1:-1
                ],
            )
        )
    return nodal_coords


def c0_bar_stiffnes_matrix(
    young_modulus: float,
    section_area: float,
    lenght: float,
    poly_degree: int,
    n_elements: int,
):
    axial_stiff = young_modulus * section_area
    n_knots_vertices = n_elements + 1
    n_knots_internal = n_elements * (poly_degree - 1)
    n_knots_total = n_knots_vertices + n_knots_internal
    element_size = lenght / n_elements
    det_jacob = element_size / 2

    return None


def x_knots_local(degree: int) -> np.array:
    return np.concatenate(((-1.0, 1.0), np.linspace(-1, 1, degree + 1)[1:-1]))
