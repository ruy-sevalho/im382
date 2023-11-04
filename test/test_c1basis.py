import numpy as np
import numpy.typing as npt
from polynomials import c1_basis
import pytest as pt


@pt.mark.parametrize(
    "degree, element_size, coords, shape_f, shape_f_1d, shape_f_2d, shape_f_3d,",
    [
        (
            3,
            1,
            np.linspace(-1, 1, 5),
            np.array(
                [
                    [1, 0.8438, 0.5, 0.1562, 0],
                    [0, 0.1406, 0.125, 0.0469, 0],
                    [0, 0.1562, 0.5, 0.8438, 1],
                    [0, -0.0469, -0.125, -0.1406, 0],
                ]
            ),
            np.array(
                [
                    [0, -0.5625, -0.75, -0.5625, 0],
                    [0.5, 0.0938, -0.125, -0.1562, 0],
                    [0, 0.5625, 0.75, 0.5625, 0],
                    [0, -0.1562, -0.125, 0.0938, 0.5],
                ]
            ),
            np.array(
                [
                    [-1.5, -0.75, 0, 0.75, 1.5],
                    [-1, -0.625, -0.25, 0.125, 0.5],
                    [1.5, 0.75, 0, -0.75, -1.5],
                    [-0.5, -0.125, 0.25, 0.625, 1],
                ]
            ),
            np.array(
                [
                    [1.5, 1.5, 1.5, 1.5, 1.5],
                    [0.75, 0.75, 0.75, 0.75, 0.75],
                    [-1.5, -1.5, -1.5, -1.5, -1.5],
                    [0.75, 0.75, 0.75, 0.75, 0.75],
                ]
            ),
        )
    ],
)
def test_c1basis(
    degree: float,
    element_size: float,
    coords: npt.NDArray,
    shape_f,
    shape_f_1d,
    shape_f_2d,
    shape_f_3d,
):
    assert np.array(
        c1_basis(degree=degree, calc_pts_coords=coords, element_size=element_size)
    ) == pt.approx(np.array((shape_f, shape_f_1d, shape_f_2d, shape_f_3d)), rel=0.001)
