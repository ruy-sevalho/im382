from polynomials import D_jacobi_polynomials, jacobi_polynomials


import numpy as np
from nptyping import Double, Int, NDArray, Shape


def c1_basis(P: Int, Coords: NDArray[Shape["1"], Double], he: Double):
    Ninteg = len(Coords)

    # Jacobi polynomials and derivatives on the integ points calculated using
    # the recurrence relations
    alfa = 1
    beta = 1
    J = np.zeros((P + 1, Ninteg))
    dJ = np.zeros((P + 1, Ninteg))
    d2J = np.zeros((P + 1, Ninteg))
    d3J = np.zeros((P + 1, Ninteg))
    Ja3b3 = np.zeros((P + 1, Ninteg))
    Ja4b4 = np.zeros((P + 1, Ninteg))
    for j in range(P + 1):
        J[j, :] = jacobi_polynomials(Coords, j, alfa, beta)
        Ja3b3[j, :] = jacobi_polynomials(Coords, j, alfa + 2, beta + 2)
        Ja4b4[j, :] = jacobi_polynomials(Coords, j, alfa + 3, beta + 3)
        dJ[j, :] = D_jacobi_polynomials(Coords, j, alfa, beta)

    # Second derivatives of Jacobi polynomials on the integ points
    for j in range(2, P):
        d2J[j + 1, :] = (
            0.25 * (alfa + j + beta + 1) * (alfa + j + beta + 2) * Ja3b3[j - 1, :]
        )

    # Third derivatives of Jacobi polynomials on the integ points
    for j in range(3, P):
        d3J[j + 1, :] = (
            0.125
            * (alfa + j + beta + 1)
            * (alfa + j + beta + 2)
            * (alfa + j + beta + 3)
            * Ja4b4[j - 2, :]
        )

    Jcsi = np.zeros((P + 1, Ninteg))
    dJcsi = np.zeros((P + 1, Ninteg))
    d2Jcsi = np.zeros((P + 1, Ninteg))
    d3Jcsi = np.zeros((P + 1, Ninteg))

    Jcsi[0, :] = 0.5 * (1 - Coords)
    Jcsi[1, :] = 0.5 * (1 + Coords)
    dJcsi[0, :] = -0.5
    dJcsi[1, :] = 0.5
    d2Jcsi[0:2, :] = 0
    d3Jcsi[0:2, :] = 0
    for i in range(2, P + 1):
        Jcsi[i, :] = (1 + Coords) * (1 - Coords) * J[i - 2, :]
        dJcsi[i, :] = (
            -2.0 * Coords * J[i - 2, :] + (1 + Coords) * (1 - Coords) * dJ[i - 2, :]
        )
        d2Jcsi[i, :] = (
            -2.0 * J[i - 2, :]
            - 2.0 * Coords * dJ[i - 2, :]
            - 2.0 * Coords * dJ[i - 2, :]
            + (1 + Coords) * (1 - Coords) * d2J[i - 2, :]
        )
        d3Jcsi[i, :] = (
            -4.0 * dJ[i - 2, :]
            - 2.0 * Coords * d2J[i - 2, :]
            - 2.0 * dJ[i - 2, :]
            - 4.0 * Coords * d2J[i - 2, :]
            + (1 + Coords) * (1 - Coords) * d3J[i - 2, :]
        )

    # Hermite cubic polynomials and their derivatives
    H = np.zeros((P + 1, Ninteg))
    d1H = np.zeros((P + 1, Ninteg))
    d2H = np.zeros((P + 1, Ninteg))
    d3H = np.zeros((P + 1, Ninteg))

    H[0, :] = 0.5 - 0.75 * Coords + 0.25 * Coords**3
    H[1, :] = (0.25 - 0.25 * Coords - 0.25 * Coords**2 + 0.25 * Coords**3) * he / 2
    H[2, :] = 0.5 + 0.75 * Coords - 0.25 * Coords**3
    H[3, :] = (-0.25 - 0.25 * Coords + 0.25 * Coords**2 + 0.25 * Coords**3) * he / 2

    d1H[0, :] = -0.75 + 0.75 * Coords**2
    d1H[1, :] = (-0.25 - 0.50 * Coords + 0.75 * Coords**2) * he / 2
    d1H[2, :] = 0.75 - 0.75 * Coords**2
    d1H[3, :] = (-0.25 + 0.50 * Coords + 0.75 * Coords**2) * he / 2

    d2H[0, :] = 1.50 * Coords
    d2H[1, :] = (-0.50 + 1.50 * Coords) * he / 2
    d2H[2, :] = -1.50 * Coords
    d2H[3, :] = (0.50 + 1.50 * Coords) * he / 2

    d3H[0, :] = 1.50
    d3H[1, :] = 1.50 * he / 2
    d3H[2, :] = -1.50
    d3H[3, :] = 1.50 * he / 2

    cont = 4
    for j in range(2, P - 1):
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
    return H, d1H, d2H, d3H
