import numpy as np
import pytest as pt
from nptyping import Double, Int, NDArray, Shape
from polynomials import (
    IntegrationTypes,
    OneDArray,
    get_points_weights,
    lagrange_poli,
    d_lagrange_poli,
)


@pt.mark.parametrize(
    "alfa, beta, intorder, type_int, coordinate, x_expected, w_expected",
    [
        (
            0,
            0,
            2,
            IntegrationTypes.GJ,
            "x",
            np.array((-0.5774, 0.5774)),
            np.array((1.0, 1.0)),
        )
    ],
)
def test_get_points_weight(
    alfa: Double,
    beta: Double,
    intorder: Int,
    type_int: IntegrationTypes,
    coordinate,
    x_expected: NDArray[Shape["Any"], Double],
    w_expected: NDArray[Shape["Any"], Double],
):
    assert np.array(
        get_points_weights(
            alfa=alfa,
            beta=beta,
            intorder=2 * intorder - 1,  # why?
            type_int=type_int,
            coordinate=coordinate,
        )
    ) == pt.approx(np.array((x_expected, w_expected)), rel=0.001)


@pt.mark.parametrize(
    "degree, pi_coords, pc_coords, expected_phi",
    [
        (
            2,
            np.array((-0.5774, 0.5774)),
            np.array((-1.0, 1.0, 0.0)),
            np.array(((0.4553, -0.1220), (-0.1220, 0.4553), (0.6667, 0.666))),
        )
    ],
)
def test_lagrange_poli(
    degree: Int,
    pi_coords: OneDArray,
    pc_coords: OneDArray,
    expected_phi,
):
    assert lagrange_poli(
        degree=degree, pi_coords=pi_coords, pc_coords=pc_coords
    ) == pt.approx(expected_phi, rel=0.001)


@pt.mark.parametrize(
    "degree, pi_coords, pc_coords, expected_dphi",
    [
        (
            2,
            np.array((-0.5774, 0.5774)),
            np.array((-1.0, 1.0, 0.0)),
            np.array(((-1.0774, 0.0774), (-0.0774, 1.0774), (1.1547, -1.1547))),
        )
    ],
)
def test_d_lagrange_poli(
    degree: Int,
    pi_coords: OneDArray,
    pc_coords: OneDArray,
    expected_dphi,
):
    assert d_lagrange_poli(
        degree=degree,
        pi_coords=pi_coords,
        pc_coords=pc_coords,
    ) == pt.approx(expected_dphi, rel=0.001)
