import collections
from itertools import islice
from dataclasses import dataclass
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from polynomials import IntegrationTypes, get_points_weights, c1_basis
from exemplo_1_9 import c0_bar

# from ex_1_def import E, L, EA, A


def c1_bar(E: float, L: float, A: float, P: int, Nel: int):
    # Material
    EA = E * A

    # Malha
    Nnosv = Nel * 2 + 2
    Nnosi = Nel * (P - 3)
    Nnos = Nnosv + Nnosi
    he = L / Nel
    detJ = he / 2

    # Coordenadas nodais
    xn = np.linspace(0, L, int(Nnosv / 2))

    # Pontos e pesos para integracao numerica
    csi, wi = get_points_weights(0, 0, 2 * (P - 1), IntegrationTypes.GJ, "x")
    Nint = len(csi)

    # Valores dos polinomios de Hermite e suas derivadas nos pontos de integracao
    (
        hs,
        d1hs,
        d2hs,
        d3hs,
    ) = c1_basis(degree=P, coords=csi, element_size=he)

    Ke = np.zeros((P + 1, P + 1))
    for n in range(Nint):
        Ke += EA * np.outer(d1hs[:, n], d1hs[:, n]) / detJ * wi[n]

    # Alocacao da matriz e dos vetores de deslocamento e carregamento globais
    Kg = np.zeros((Nnos, Nnos))
    Ug = np.zeros(Nnos)
    Fg = np.zeros(Nnos)

    # Matriz de incidencia dos elementos
    IncidEls = np.zeros((Nel, P + 1), dtype=int)

    # Processo de montagem ou superposicao da matriz de rigidez e do vetor de forcas globais
    ni = Nnosv
    for e in range(Nel):
        start = 2 * e
        IncidEls[e, :4] = np.arange(0, 4) + start
        for n in range(4, P + 1):
            IncidEls[e, n] = ni
            ni += 1
        Kg[IncidEls[e, :][:, np.newaxis], IncidEls[e, :][np.newaxis, :]] += Ke

        # Vetor de forca nodal equivalente devido a carga distribuida senoidal
        fe = np.zeros(P + 1)
        for i in range(Nint):
            xei = 0.5 * (1 - csi[i]) * xn[e] + 0.5 * (1 + csi[i]) * xn[e + 1]
            fe += 1000 * np.sin(np.pi / 2 * xei) * hs[:, i] * detJ * wi[i]
        Fg[IncidEls[e, :][np.newaxis, :]] += fe

    # Aplicacao das condicoes de contorno
    NosLivres = np.arange(1, Nnos)
    NosLivres = NosLivres[NosLivres != Nnosv - 1]

    # Calculo dos deslocamentos dos nos livres
    m = Kg[NosLivres[:, np.newaxis], NosLivres[np.newaxis, :]]

    Ug[NosLivres] = np.linalg.solve(
        Kg[NosLivres[:, np.newaxis], NosLivres[np.newaxis, :]], Fg[NosLivres]
    )

    # Calculo da reacao de apoio
    R = np.dot(Kg[0], Ug) - Fg[0]

    # Calculo das deformacoes e tensoes elemento nas coordenadas locais do elemento
    EpsilonEls = np.zeros((Nel, 2))
    SigmaEls = np.zeros((Nel, 2))
    csip = np.array((-1, 1))
    (
        hs,
        d1hs,
        d2hs,
        d3hs,
    ) = c1_basis(degree=P, coords=csip, element_size=he)

    for e in range(Nel):
        # vetor de deslocamentos nodais do elemento
        ue = Ug[IncidEls[e]]

        for k in range(2):
            Becsi = d1hs[:, k]
            EpsilonEls[e, k] = 2 / he * np.dot(Becsi, ue)
            SigmaEls[e, k] = E * EpsilonEls[e, k]

    # Solucao analitica do deslocamento axial, deformacao longitudinal e tensao normal
    xi = np.arange(0, L + he / 5, he / 5)
    EpsilonNum = 2000 / (np.pi * EA) * np.cos(np.pi / 2 * xi)
    SigmaNum = E * EpsilonNum

    # Calculo das normas L2 e de energia do erro usando o sistema de coordenadas xb do elemento
    phip = hs
    une = np.zeros(P + 1)
    for e in range(Nel):
        # mapeamento das coordenadas do elemento
        xe = (1 - csip) / 2 * xn[e] + (1 + csip) / 2 * xn[e + 1]

        # solucao analitica no elemento
        uae = 4 / (np.pi**2 * EA) * 1000 * np.sin(np.pi / 2 * xe)

        # solucao aproximada no elemento !!!!!!!!!!!
        for k in range(2):
            Ne = phip[:, k]
            ue = Ug[IncidEls[e]]
            une[k] = np.dot(Ne, ue)

    # Norma de energia da solucao exata
    def ua(x):
        return 4 / (np.pi**2 * EA) * 1000 * np.sin(np.pi / 2 * x)

    x = sp.symbols("x", real=True)
    ua = 4 / (sp.pi**2 * EA) * 1000 * sp.sin(sp.pi / 2 * x)
    uE = sp.sqrt(sp.integrate(EA * sp.diff(ua) ** 2, (x, 0, L)))

    def displacement_analytical(x):
        return 4000 * np.sin(np.pi * x / 2) / (np.pi**2 * EA)

    elements_error_l2 = np.zeros(Nel)
    xs_element_local = np.linspace(0, he, 6)
    (
        hs,
        d1hs,
        d2hs,
        d3hs,
    ) = c1_basis(degree=P, coords=xs_element_local, element_size=he)

    for e in range(Nel):
        xe = (1 - xs_element_local / he) * xn[e] + xs_element_local / he * xn[e + 1]
        element_displacement_analytical = np.array(
            [displacement_analytical(x) for x in xe]
        )
        element_displacement_aprox = hs.T @ Ug[IncidEls[e]]
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

    # Norma de energia da solucao exata
    unE = np.sqrt(Ug @ Kg @ Ug)

    # Norma de energia do erro
    ErroEn = uE - unE
    return ErroEn.evalf(), error_l2
    # return Kg, Ug, Fg


degrees = (3, 4, 5, 6)
n_elements = (4, 8, 16, 32)

# Comprimento da barra
L = 1.0

# Modulo de Young do material
E = 100e9  # [N/m^2]

# Area da secao transversal
A = 1.0e-4  # [m^2]

# Kg, Ug, Fg = one_d_c1_bar(E=E, L=L, A=A, P=5, Nel=2)

c1_error_h_refinement = np.array(
    tuple(
        (c1_bar(E, L, A, 3, n) for n in n_elements),
    )
)

c1_error_p_refinement = np.array(
    tuple(
        (c1_bar(E, L, A, p, 4) for p in degrees),
    )
)
c0_error_h_refinement = np.array(
    tuple(
        (c0_bar(E, L, A, 3, n) for n in n_elements),
    )
)

c0_error_p_refinement = np.array(
    tuple(
        (c0_bar(E, L, A, p, 4) for p in degrees),
    )
)
titles = (
    "C1 Energy Norm Error - p ref",
    "C1 L2 Norm Error - p ref",
    "C1 Energy Norm Error - h ref",
    "C1 L2 Norm Error - h ref",
    "C0 Energy Norm Error - p ref",
    "C0 L2 Norm Erro r - p ref",
    "C0 Energy Norm Error - h ref",
    "C0 L2 Norm Error - h ref",
)
y_labels = ("Energy norm error (log scale)", "L2 norm error (log scale)") * 4

values = (
    c1_error_p_refinement[:, 0],
    c1_error_p_refinement[:, 1],
    c1_error_h_refinement[:, 0],
    c1_error_h_refinement[:, 1],
    c0_error_p_refinement[:, 0],
    c0_error_p_refinement[:, 1],
    c0_error_h_refinement[:, 0],
    c0_error_h_refinement[:, 1],
)
x = (
    "Degree (log scale)",
    "Degree (log scale)",
    "Num elementos (log scale)",
    "Num elementos (log scale)",
) * 2

for i, (title, value) in enumerate(zip(titles, values)):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.loglog(n_elements, value)
    axs.set_xlabel(x[i])
    axs.set_ylabel(y_labels[i])
    axs.set_title(title)
plt.show()
