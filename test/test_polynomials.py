from functools import partial
import numpy as np
import sympy as sp
import pytest as pt
from polynomials import (
    IntegrationTypes,
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
    alfa: float,
    beta: float,
    intorder: int,
    type_int: IntegrationTypes,
    coordinate,
    x_expected: np.ndarray,
    w_expected: np.ndarray,
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


# Test indicates that gauss-jacobi quadrature fails for low point count and a function composed with sin
@pt.mark.parametrize("degree", (i for i in range(2, 11)))
def test_gauss_langrange_quadrature_on_polynomial_composed_trig(degree: int):
    def f(x):
        return 1000 * np.sin(np.pi / 2 * x)

    x_zeros = np.linspace(-1, 1, degree + 1)
    lagrange_poli_pre = partial(
        lagrange_poli, degree=degree, placement_pts_coords=x_zeros
    )
    x_int_trapz = np.linspace(-1, 1, 2001)
    langrage_poli_x_int_trapz = lagrange_poli_pre(calc_pts_coords=x_int_trapz)
    x_quadrature_int_pts, int_weights = get_points_weights(intorder=2 * degree + 2)
    lagrange_poli_quadrature = lagrange_poli_pre(calc_pts_coords=x_quadrature_int_pts)
    quadrature_values = np.array(
        tuple(
            tuple(
                f(int_pt) * int_weight * lagrange_poly_value
                for int_pt, int_weight, lagrange_poly_value in zip(
                    x_quadrature_int_pts, int_weights, lagrange_poly
                )
            )
            for lagrange_poly in lagrange_poli_quadrature
        )
    )
    quadrature_integral = np.array(tuple(np.sum(row) for row in quadrature_values))
    y = np.array(
        tuple(
            tuple(
                langrage_poli_x_int_trap_value * f(x)
                for langrage_poli_x_int_trap_value, x in zip(
                    langrage_poli_x_int_trapz_row, x_int_trapz
                )
            )
            for langrage_poli_x_int_trapz_row in langrage_poli_x_int_trapz
        )
    )
    num_integral = np.array([np.trapz(y_, x_int_trapz) for y_ in y])
    res = (num_integral - quadrature_integral) / num_integral
    assert quadrature_integral == pt.approx(num_integral, rel=1e-3)


@pt.mark.parametrize("power", (i for i in range(3, 11)))
def test_gauss_langrange_quadrature_on_polynomial(power: int):
    x: sp.Symbol = sp.symbols("x")
    f_symb = x
    for i in range(2, power):
        f_symb += x**i
    def_integral = sp.integrate(f_symb, (x, -1, 1)).evalf()
    f = sp.lambdify(x, f_symb)
    int_pts, int_weights = get_points_weights(intorder=power)
    num_integral = sum(
        (f(int_pt) * int_weight for int_pt, int_weight in zip(int_pts, int_weights))
    )
    assert num_integral == pt.approx(def_integral)


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
    degree: int,
    pi_coords: np.ndarray,
    pc_coords: np.ndarray,
    expected_phi,
):
    assert lagrange_poli(
        degree=degree, calc_pts_coords=pi_coords, placement_pts_coords=pc_coords
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
    degree: int,
    pi_coords: np.ndarray,
    pc_coords: np.ndarray,
    expected_dphi,
):
    assert d_lagrange_poli(
        degree=degree,
        calc_pts_coords=pi_coords,
        placement_pts_coords=pc_coords,
    ) == pt.approx(expected_dphi, rel=0.001)
