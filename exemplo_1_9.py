import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from polynomials import (
    IntegrationTypes,
    get_points_weights,
    lagrange_poli,
    d_lagrange_poli,
)


def c0_bar(E: float, L: float, A: float, P: int, Nel: int):
    EA = E * A
    Nnosv = Nel + 1  # numero de nos de vertice
    Nnosi = Nel * (P - 1)  # numero de nos interno
    Nnos = Nnosv + Nnosi  # numero total de nos
    he = L / Nel  # tamanho do elemento
    detJ = he / 2  # determinante do Jacobiano

    # Coordenadas nodais
    xn = np.zeros(Nnos)
    xn[:Nnosv] = np.linspace(0, L, Nnosv)  # nos de vertice

    no_int = Nnosv  # indice dos nos internos
    dhe = he / P  # incremento de coordenadas dos nos internos

    for e in range(Nel):
        for i in range(P - 1):
            xn[no_int] = xn[e] + (i + 1) * dhe
            no_int += 1

    csi, wi = get_points_weights(0, 0, 2 * (P - 1), IntegrationTypes.GJ, "x")

    # Numero de pontos de integracao
    Nint = len(csi)

    # Coordenadas nodais do elemento padrao
    csipc = np.linspace(-1, 1, P + 1)
    csipcop = csipc.copy()
    csipc[1] = csipcop[-1]
    csipc[2:] = csipcop[1:-1]

    # Valores dos polinomios de Lagrange e derivadas primeira nos pontos de integracao
    Necsi = lagrange_poli(P, csi, csipc)
    Becsi = d_lagrange_poli(P, csi, csipc)

    # Matriz de rigidez dos elementos
    Ke = np.zeros((P + 1, P + 1))
    for n in range(Nint):
        Ke += EA * np.outer(Becsi[:, n], Becsi[:, n]) / detJ * wi[n]

    # Alocacao da matriz e dos vetores de deslocamento e carregamento globais
    Kg = np.zeros((Nnos, Nnos))
    Ug = np.zeros(Nnos)
    Fg = np.zeros(Nnos)

    # Matriz de incidencia dos elementos
    IncidEls = np.zeros((Nel, P + 1), dtype=int)

    # Processo de montagem ou superposicao da matriz de rigidez e do vetor de forcas globais
    ni = Nnosv
    for e in range(Nel):
        IncidEls[e, :2] = [e, e + 1]
        for n in range(2, P + 1):
            IncidEls[e, n] = ni
            ni += 1
        Kg[IncidEls[e, :][:, np.newaxis], IncidEls[e, :][np.newaxis, :]] += Ke

        # Vetor de forca nodal equivalente devido a carga distribuida senoidal
        fe = np.zeros(P + 1)
        for i in range(Nint):
            xei = 0.5 * (1 - csi[i]) * xn[e] + 0.5 * (1 + csi[i]) * xn[e + 1]
            fe += 1000 * np.sin(np.pi / 2 * xei) * Necsi[:, i] * detJ * wi[i]
        Fg[IncidEls[e, :][np.newaxis, :]] += fe

    # Aplicacao das condicoes de contorno
    NosLivres = np.arange(1, Nnos)

    # Calculo dos deslocamentos dos nos livres
    Ug[NosLivres] = np.linalg.solve(Kg[np.ix_(NosLivres, NosLivres)], Fg[NosLivres])

    # Calculo da reacao de apoio
    R = np.dot(Kg[0], Ug) - Fg[0]

    # Calculo das deformacoes e tensoes elemento nas coordenadas locais do elemento
    EpsilonEls = np.zeros((Nel, P + 1))
    SigmaEls = np.zeros((Nel, P + 1))

    csip = np.linspace(-1, 1, P + 1)
    Becsip = d_lagrange_poli(degree=P, pi_coords=csip, pc_coords=csipc)

    # Solucao analitica do deslocamento axial, deformacao longitudinal e tensao normal
    xi = np.arange(0, L + he / 5, he / 5)
    EpsilonNum = 2000 / (np.pi * EA) * np.cos(np.pi / 2 * xi)
    SigmaNum = E * EpsilonNum

    x = sp.symbols("x", real=True)
    ua = 4 / (sp.pi**2 * EA) * 1000 * sp.sin(sp.pi / 2 * x)
    uE = sp.sqrt(sp.integrate(EA * sp.diff(ua) ** 2, (x, 0, L)))

    # Norma de energia da solucao aprox
    unE = np.sqrt(Ug @ Kg @ Ug)

    # Norma de energia do erro
    ErroEn = uE - unE

    def displacement_analytical(x):
        return 4000 * np.sin(np.pi * x / 2) / (np.pi**2 * EA)

    elements_error_l2 = np.zeros(Nel)
    xs_element_local = np.linspace(0, he, 6)

    csipc, wcp = get_points_weights(0, 0, P + 1, IntegrationTypes.GJ, "x")

    # Coordenadas nodais do elemento padrao
    csipc = np.linspace(-1, 1, P + 1)
    csipcop = csipc.copy()
    csipc[1] = csipcop[-1]
    csipc[2:] = csipcop[1:-1]

    phip = lagrange_poli(P, csip, csipc)
    xs_element_local = csipc
    for e in range(Nel):
        xe = (1 - xs_element_local / he) * xn[e] + xs_element_local / he * xn[e + 1]
        element_displacement_analytical = np.array(
            [displacement_analytical(x) for x in xe]
        )
        phip = lagrange_poli(P, csip, csipc)
        element_displacement_aprox = np.zeros(P + 1)

        # solucao aproximada no elemento
        for k in range(P + 1):
            Ne = phip[k]
            ue = Ug[IncidEls[e]]
            element_displacement_aprox[k] = np.dot(Ne, ue)

        elements_error_l2[e] = np.trapz(
            y=np.array(
                [
                    (analytical_sol - aprox_sol) ** 2
                    for analytical_sol, aprox_sol in zip(
                        element_displacement_analytical, element_displacement_aprox
                    )
                ]
            ),
            x=xe,
        )
    error_l2 = np.sqrt(np.sum(elements_error_l2))

    return ErroEn.evalf(), error_l2
