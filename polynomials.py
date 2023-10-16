from typing import Any
import math
from enum import Enum
import numpy as np
from nptyping import NDArray, Double, Shape, Int

OneDArray = NDArray[Shape["Any"], Double]


def jacobi_polynomials(r: OneDArray, m: Int, alfa: Double, beta: Double) -> OneDArray:
    # Two initial terms of the recurrence relation
    Pn = 1
    Pn1 = 0.5 * (alfa - beta + (alfa + beta + 2) * r)

    # General recurrence relation
    for n in range(1, m):
        a1n = 2 * (n + 1) * (n + alfa + beta + 1) * (2 * n + alfa + beta)
        a2n = (2 * n + alfa + beta + 1) * (alfa**2 - beta**2)
        a3n = (
            (2 * n + alfa + beta)
            * (2 * n + alfa + beta + 1)
            * (2 * n + alfa + beta + 2)
        )
        a4n = 2 * (n + alfa) * (n + beta) * (2 * n + alfa + beta + 2)

        Pn2 = (1 / a1n) * ((a2n + a3n * r) * Pn1 - a4n * Pn)
        Pn = Pn1
        Pn1 = Pn2

    if m > 1:
        Pm = Pn2
    elif m == 1:
        Pm = Pn1
    else:
        Pm = Pn
    return Pm


def D_jacobi_polynomials(r: OneDArray, m: Int, alfa: Double, beta: Double) -> OneDArray:
    # Jacobi polynomials calculated at point r
    Pn = 1
    Pn1 = 0.5 * (alfa - beta + (alfa + beta + 2) * r)

    for n in range(1, m + 1):
        a1n = 2 * (n + 1) * (n + alfa + beta + 1) * (2 * n + alfa + beta)
        a2n = (2 * n + alfa + beta + 1) * (alfa**2 - beta**2)
        a3n = (
            (2 * n + alfa + beta)
            * (2 * n + alfa + beta + 1)
            * (2 * n + alfa + beta + 2)
        )
        a4n = 2 * (n + alfa) * (n + beta) * (2 * n + alfa + beta + 2)

        Pn2 = (1 / a1n) * ((a2n + a3n * r) * Pn1 - a4n * Pn)

        b1n = (2 * n + alfa + beta) * (1 - r**2)
        b2n = n * (alfa - beta - (2 * n + alfa + beta) * r)
        b3n = 2 * (n + alfa) * (n + beta)

        DPn1 = (1 / b1n) * (b2n * Pn1 + b3n * Pn)

        Pn = Pn1
        Pn1 = Pn2

    if m == 0:
        DPm = np.zeros(len(r))
    else:
        DPm = DPn1

    return DPm


def jacobi_root(alfa: Double, beta: Double, m: Int):
    # Initialize the vector of roots and the precision for the Newton-Raphson method
    x = np.zeros(m)
    epsilon = 1e-16

    for k in range(m):
        r = -np.cos(np.pi * (2 * k + 1) / (2 * m))

        if k > 0:
            r = (r + x[k - 1]) / 2

        delta = 1

        while abs(delta) > epsilon:
            s = 0

            for i in range(k):
                s += 1 / (r - x[i])
            Pm = jacobi_polynomials(r, m, alfa, beta)
            DPm = D_jacobi_polynomials(r, m, alfa, beta)

            delta = -Pm / (DPm - Pm * s)
            r += delta

        x[k] = r

    return x


class IntegrationTypes(Enum):
    GJ = "GJ"
    GHJ = "GHJ"
    GLJ = "GLJ"


def gauss_lobato_jacobi_weights(x, alfa, beta):
    m = len(x)
    w = np.zeros(m)
    c = 2 ** (alfa + beta + 1)
    g1 = alfa + m
    g2 = beta + m
    g3 = alfa + beta + m + 1

    gama1 = math.factorial(g1 - 1)
    gama2 = math.factorial(g2 - 1)
    gama3 = math.factorial(g3 - 1)
    M = math.factorial(m - 1)

    C1 = c * gama1 * gama2
    C2 = (m - 1) * M * gama3
    C3 = beta + 1
    C4 = alfa + 1

    r = x[0]
    Pm = jacobi_polynomials(r, m - 1, alfa, beta)
    w[0] = C3 * (C1 / C2) * (Pm ** (-2))

    r = x[m - 1]
    Pm = jacobi_polynomials(r, m - 1, alfa, beta)
    w[m - 1] = C4 * (C1 / C2) * (Pm ** (-2))

    for k in range(1, m - 1):
        r = x[k]
        Pm = jacobi_polynomials(r, m - 1, alfa, beta)
        w[k] = (C1 / C2) * (Pm ** (-2))


def gauss_radau_jacobi_weights(x, alfa, beta):
    m = len(x)
    w = np.zeros(m)
    c = 2 ** (alfa + beta)
    g1 = alfa + m
    g2 = beta + m
    g3 = alfa + beta + m + 1

    gama1 = math.factorial(g1 - 1)
    gama2 = math.factorial(g2 - 1)
    gama3 = math.factorial(g3 - 1)
    M = math.factorial(m - 1)

    r = x[0]
    Pm = jacobi_polynomials(r, m - 1, alfa, beta)
    C1 = c * gama1 * gama2 * (1 - r)
    C2 = M * (beta + m) * gama3
    C3 = beta + 1
    w[0] = C3 * (C1 / C2) * (Pm ** (-2))

    for k in range(1, m):
        r = x[k]
        Pm = jacobi_polynomials(r, m - 1, alfa, beta)
        C1 = c * gama1 * gama2 * (1 - r)
        w[k] = (C1 / C2) * (Pm ** (-2))

    return w


def gauss_jacobi_weights(x, alfa, beta):
    m = len(x)
    w = np.zeros(m)
    c = 2 ** (alfa + beta + 1)
    g1 = alfa + m + 1
    g2 = beta + m + 1
    g3 = alfa + beta + m + 1

    gama1 = math.factorial(g1 - 1)
    gama2 = math.factorial(g2 - 1)
    gama3 = math.factorial(g3 - 1)
    M = math.factorial(m)

    C1 = c * gama1 * gama2

    for k in range(m):
        r = x[k]
        DPm = D_jacobi_polynomials(r, m, alfa, beta)
        C2 = M * gama3 * (1 - r**2)
        w[k] = (C1 / C2) * (DPm ** (-2))

    return w


def quadrature_gauss_jacobi(alfa: Double, beta: Double, intorder: Int):
    n_points = math.ceil(0.5 * (intorder + 1))
    x = jacobi_root(alfa, beta, n_points)
    w = gauss_jacobi_weights(x, alfa, beta)
    return x, w


def quadrature_gauss_jacobi_n_pts(n_points: Int, alfa: Double = 0, beta: Double = 0):
    x = jacobi_root(alfa, beta, n_points)
    w = gauss_jacobi_weights(x, alfa, beta)
    return x, w


def quadrature_gauss_radau_jacobi(alfa: Double, beta: Double, intorder: Int):
    n_points = math.ceil(0.5 * (intorder + 2))
    x = np.zeros(n_points)
    x[0] = -1.0
    x[1:] = jacobi_root(alfa, beta + 1, n_points - 1)
    w = gauss_radau_jacobi_weights(x, alfa, beta)
    return x, w


def quadrature_gauss_lobato_jacobi(alfa: Double, beta: Double, intorder: Int):
    n_points = math.ceil(0.5 * (intorder + 3))

    # Initialize arrays for nodes (x) and weights (w)
    x = np.zeros(n_points)
    w = np.zeros(n_points)

    # Set the first node
    x[0] = -1.0

    # Calculate the intermediate nodes using jacobi_root
    x[1 : n_points - 1] = jacobi_root(alfa + 1, beta + 1, n_points - 2)

    # Set the last node
    x[n_points - 1] = 1.0

    # Calculate the weights using jacobi_weight
    w = gauss_lobato_jacobi_weights(x, alfa, beta)

    return x, w


def get_points_weights(
    alfa: Double = 0,
    beta: Double = 0,
    intorder: Int = 1,
    type_int: IntegrationTypes = IntegrationTypes.GJ,
    coordinate: str = "x",
) -> tuple[OneDArray, OneDArray]:
    table = {
        IntegrationTypes.GJ: quadrature_gauss_jacobi,
        IntegrationTypes.GHJ: quadrature_gauss_radau_jacobi,
        IntegrationTypes.GLJ: quadrature_gauss_lobato_jacobi,
    }
    return table[type_int](alfa=alfa, beta=beta, intorder=intorder)


def lagrange_poli(
    degree: Int,
    calc_pts_coords: OneDArray,
    placement_pts_coords: OneDArray,
):
    # Initialization and variable allocation
    m = degree + 1
    n = len(calc_pts_coords)
    i = np.arange(m)
    b = np.zeros(m)
    phi = np.zeros((m, n))

    # Calculate the denominators of Lagrange polynomials
    for k in range(m):
        indices = np.where(i != k)
        b[k] = np.prod(placement_pts_coords[k] - placement_pts_coords[indices])

    # Calculate the polynomials
    for j in range(n):
        for k in range(m):
            indices = np.where(i != k)
            phi[k, j] = (
                np.prod(calc_pts_coords[j] - placement_pts_coords[indices]) / b[k]
            )
            pass

    return phi


def d_lagrange_poli(
    degree: Int, calc_pts_coords: OneDArray, placement_pts_coords: OneDArray
):
    # Initialization and variable allocation
    n = len(calc_pts_coords)
    m = degree + 1
    d_phi = np.zeros((m, n))

    # Calculate the first derivatives
    for i in range(m):
        for k in range(m):
            if k != i:
                phi = np.ones(n)
                for j in range(m):
                    if j != i and j != k:
                        phi *= (np.array(calc_pts_coords) - placement_pts_coords[j]) / (
                            placement_pts_coords[i] - placement_pts_coords[j]
                        )
                d_phi[i, :] += phi / (placement_pts_coords[i] - placement_pts_coords[k])

    return d_phi
