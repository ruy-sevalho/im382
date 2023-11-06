import math
from enum import Enum
import numpy as np
import numpy.typing as npt


def jacobi_polynomials(
    r: npt.NDArray[np.float64], m: int, alfa: float, beta: float
) -> npt.NDArray[np.float64]:
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


def D_jacobi_polynomials(
    r: npt.NDArray[np.float64], m: int, alfa: float, beta: float
) -> npt.NDArray[np.float64]:
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


def jacobi_root(alfa: float, beta: float, m: int):
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


def quadrature_gauss_jacobi(alfa: float, beta: float, intorder: int):
    n_points = math.ceil(0.5 * (intorder + 1))
    x = jacobi_root(alfa, beta, n_points)
    w = gauss_jacobi_weights(x, alfa, beta)
    return x, w


def quadrature_gauss_jacobi_n_pts(n_points: int, alfa: float = 0, beta: float = 0):
    x = jacobi_root(alfa, beta, n_points)
    w = gauss_jacobi_weights(x, alfa, beta)
    return x, w


def quadrature_gauss_radau_jacobi(alfa: float, beta: float, intorder: int):
    n_points = math.ceil(0.5 * (intorder + 2))
    x = np.zeros(n_points)
    x[0] = -1.0
    x[1:] = jacobi_root(alfa, beta + 1, n_points - 1)
    w = gauss_radau_jacobi_weights(x, alfa, beta)
    return x, w


def quadrature_gauss_radau_jacobi_n_pts(alfa: float, beta: float, intorder: int):
    n_points = intorder
    x = np.zeros(n_points)
    x[0] = -1.0
    x[1:] = jacobi_root(alfa, beta + 1, n_points - 1)
    w = gauss_radau_jacobi_weights(x, alfa, beta)
    return x, w


def quadrature_gauss_lobato_jacobi_n_pts(alfa: float, beta: float, intorder: int = 1):
    n_points = intorder

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


def quadrature_gauss_lobato_jacobi(alfa: float, beta: float, intorder: int):
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
    alfa: float = 0,
    beta: float = 0,
    intorder: int = 1,
    type_int: IntegrationTypes = IntegrationTypes.GJ,
    coordinate: str = "x",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    table = {
        IntegrationTypes.GJ: quadrature_gauss_jacobi,
        IntegrationTypes.GHJ: quadrature_gauss_radau_jacobi,
        IntegrationTypes.GLJ: quadrature_gauss_lobato_jacobi,
    }
    return table[type_int](alfa=alfa, beta=beta, intorder=intorder)


def get_points_weights_degree(
    alfa: float = 0,
    beta: float = 0,
    intorder: int = 1,
    type_int: IntegrationTypes = IntegrationTypes.GJ,
    coordinate: str = "x",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    table = {
        IntegrationTypes.GJ: quadrature_gauss_jacobi_n_pts,
        IntegrationTypes.GHJ: quadrature_gauss_radau_jacobi_n_pts,
        IntegrationTypes.GLJ: quadrature_gauss_lobato_jacobi_n_pts,
    }
    return table[type_int](alfa=alfa, beta=beta, intorder=intorder)


def lagrange_poli(
    calc_pts_coords: npt.NDArray[np.float64],
    degree: int,
    placement_pts_coords: npt.NDArray[np.float64],
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
    calc_pts_coords: npt.NDArray[np.float64],
    placement_pts_coords: npt.NDArray[np.float64],
    degree: int,
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


def c1_basis(
    calc_pts_coords: npt.NDArray[np.float64],
    degree: int,
    element_size: float,
    return_derivative_order: int | None = None,
) -> (
    npt.NDArray[np.float64]
    | tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
):
    n_integ_pts = len(calc_pts_coords)

    # Jacobi polynomials and derivatives on the integ points calculated using
    # the recurrence relations
    alfa = 1
    beta = 1
    J = np.zeros((degree + 1, n_integ_pts))
    dJ = np.zeros((degree + 1, n_integ_pts))
    d2J = np.zeros((degree + 1, n_integ_pts))
    d3J = np.zeros((degree + 1, n_integ_pts))
    Ja3b3 = np.zeros((degree + 1, n_integ_pts))
    Ja4b4 = np.zeros((degree + 1, n_integ_pts))
    for j in range(degree + 1):
        J[j, :] = jacobi_polynomials(calc_pts_coords, j, alfa, beta)
        Ja3b3[j, :] = jacobi_polynomials(calc_pts_coords, j, alfa + 2, beta + 2)
        Ja4b4[j, :] = jacobi_polynomials(calc_pts_coords, j, alfa + 3, beta + 3)
        dJ[j, :] = D_jacobi_polynomials(calc_pts_coords, j, alfa, beta)

    # Second derivatives of Jacobi polynomials on the integ points
    for j in range(2, degree):
        d2J[j + 1, :] = (
            0.25 * (alfa + j + beta + 1) * (alfa + j + beta + 2) * Ja3b3[j - 1, :]
        )

    # Third derivatives of Jacobi polynomials on the integ points
    for j in range(3, degree):
        d3J[j + 1, :] = (
            0.125
            * (alfa + j + beta + 1)
            * (alfa + j + beta + 2)
            * (alfa + j + beta + 3)
            * Ja4b4[j - 2, :]
        )

    Jcsi = np.zeros((degree + 1, n_integ_pts))
    dJcsi = np.zeros((degree + 1, n_integ_pts))
    d2Jcsi = np.zeros((degree + 1, n_integ_pts))
    d3Jcsi = np.zeros((degree + 1, n_integ_pts))

    Jcsi[0, :] = 0.5 * (1 - calc_pts_coords)
    Jcsi[1, :] = 0.5 * (1 + calc_pts_coords)
    dJcsi[0, :] = -0.5
    dJcsi[1, :] = 0.5
    d2Jcsi[0:2, :] = 0
    d3Jcsi[0:2, :] = 0
    for i in range(2, degree + 1):
        Jcsi[i, :] = (1 + calc_pts_coords) * (1 - calc_pts_coords) * J[i - 2, :]
        dJcsi[i, :] = (
            -2.0 * calc_pts_coords * J[i - 2, :]
            + (1 + calc_pts_coords) * (1 - calc_pts_coords) * dJ[i - 2, :]
        )
        d2Jcsi[i, :] = (
            -2.0 * J[i - 2, :]
            - 2.0 * calc_pts_coords * dJ[i - 2, :]
            - 2.0 * calc_pts_coords * dJ[i - 2, :]
            + (1 + calc_pts_coords) * (1 - calc_pts_coords) * d2J[i - 2, :]
        )
        d3Jcsi[i, :] = (
            -4.0 * dJ[i - 2, :]
            - 2.0 * calc_pts_coords * d2J[i - 2, :]
            - 2.0 * dJ[i - 2, :]
            - 4.0 * calc_pts_coords * d2J[i - 2, :]
            + (1 + calc_pts_coords) * (1 - calc_pts_coords) * d3J[i - 2, :]
        )

    # Hermite cubic polynomials and their derivatives
    H = np.zeros((degree + 1, n_integ_pts))
    d1H = np.zeros((degree + 1, n_integ_pts))
    d2H = np.zeros((degree + 1, n_integ_pts))
    d3H = np.zeros((degree + 1, n_integ_pts))

    H[0, :] = 0.5 - 0.75 * calc_pts_coords + 0.25 * calc_pts_coords**3
    H[1, :] = (
        (
            0.25
            - 0.25 * calc_pts_coords
            - 0.25 * calc_pts_coords**2
            + 0.25 * calc_pts_coords**3
        )
        * element_size
        / 2
    )
    H[2, :] = 0.5 + 0.75 * calc_pts_coords - 0.25 * calc_pts_coords**3
    H[3, :] = (
        (
            -0.25
            - 0.25 * calc_pts_coords
            + 0.25 * calc_pts_coords**2
            + 0.25 * calc_pts_coords**3
        )
        * element_size
        / 2
    )

    d1H[0, :] = -0.75 + 0.75 * calc_pts_coords**2
    d1H[1, :] = (
        (-0.25 - 0.50 * calc_pts_coords + 0.75 * calc_pts_coords**2)
        * element_size
        / 2
    )
    d1H[2, :] = 0.75 - 0.75 * calc_pts_coords**2
    d1H[3, :] = (
        (-0.25 + 0.50 * calc_pts_coords + 0.75 * calc_pts_coords**2)
        * element_size
        / 2
    )

    d2H[0, :] = 1.50 * calc_pts_coords
    d2H[1, :] = (-0.50 + 1.50 * calc_pts_coords) * element_size / 2
    d2H[2, :] = -1.50 * calc_pts_coords
    d2H[3, :] = (0.50 + 1.50 * calc_pts_coords) * element_size / 2

    d3H[0, :] = 1.50
    d3H[1, :] = 1.50 * element_size / 2
    d3H[2, :] = -1.50
    d3H[3, :] = 1.50 * element_size / 2

    cont = 4
    for j in range(2, degree - 1):
        # Internal functions
        H[cont] = d1H[0, :] * Jcsi[j, :]

        # 1st derivative
        d1H[j + 2, :] = d2H[0, :] * Jcsi[j, :] + d1H[0, :] * dJcsi[j, :]

        # 2nd derivative
        d2H[j + 2, :] = (
            d3H[0, :] * Jcsi[j, :]
            + 2 * d2H[0, :] * dJcsi[j, :]
            + d1H[0, :] * d2Jcsi[j, :]
        )

        # 3rd derivative
        d3H[j + 2, :] = 3 * d3H[0, :] * dJcsi[j, :] + 3 * d2H[0, :]
        cont = cont + 1
    table = {None: (H, d1H, d2H, d3H), 0: H, 1: d1H, 2: d2H, 3: d3H}
    return table[return_derivative_order]
