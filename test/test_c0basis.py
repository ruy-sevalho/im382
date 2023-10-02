import pytest as pt
import numpy as np
from c0_basis import x_knots_global, x_knots_local


@pt.mark.parametrize(
    "length, poly_degree, n_elements, expected_knots",
    [[1, 1, 5, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]], [1, 2, 2, [0.0, 0.5, 1.0, 0.25, 0.75]]],
)
def test_x_knots_global(
    length: float, poly_degree: int, n_elements: int, expected_knots
):
    assert x_knots_global(
        length=length, poly_degree=poly_degree, n_elements=n_elements
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
    assert x_knots_local(degree) == pt.approx(expected_knots)
