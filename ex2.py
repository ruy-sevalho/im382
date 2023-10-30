import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

from bar_1d import BarInput, BarInputNonLiner
from c0_basis import calc_ecsi_placement_coords_gauss_lobato
from lame import calc_lambda, calc_mu
from energy_norm import calc_energy_norm
from newton_raphson import (
    ConvergenceCriteria,
    NewtonRaphsonConvergenceParam,
    C0BarAnalysisInput,
    C1BarAnalysisInput,
    BarAnalysis,
)
from nomeclature import NUM_DISPLACEMENT, X_COORD


def pe(x: float):
    print(f"{x:e}")


young_modulus = 210e9
poisson = 0.3
lame_lambda = calc_lambda(young_modulus=young_modulus, poisson=poisson)
lame_mu = calc_mu(young_modulus=young_modulus, poisson=poisson)
section_area = 1.0e-3
length = 4.0
degree = 3
n_elements = 10
x = sy.symbols("x")
p, lambda_, mu, cp, A = sy.symbols("p lambda mu C_p A")
energy_norm = calc_energy_norm(
    mu=lame_mu, lambda_=lame_lambda, area=section_area, length=length
)
displacement_analytical_symb = 0.2 * p + 0.1 * sy.sin(p)  # type: ignore
displacement_analytical_num = sy.lambdify(p, displacement_analytical_symb)
grad_p = 1 + sy.diff(displacement_analytical_symb, p)
cauchy_green = grad_p**2
stress_piolla_kirchhoff_2: sy.Expr = (
    mu * (1 - 1 / cp) + lambda_ * sy.ln(sy.sqrt(cp)) / cp
)
stress_piolla_kirchhoff_2 = stress_piolla_kirchhoff_2.subs(cp, cauchy_green)
stress_piolla_kirchhoff_2_num = sy.lambdify(
    p,
    stress_piolla_kirchhoff_2.subs(
        {A: section_area, lambda_: lame_lambda, mu: lame_mu, p: length}
    ),
)

stress_piolla_kirchhoff_1 = stress_piolla_kirchhoff_2 * grad_p
normal_1 = A * stress_piolla_kirchhoff_1
normal_1 = normal_1.simplify()
dist_load = -sy.diff(normal_1, p)
dist_load_num = sy.lambdify(
    p, dist_load.subs({lambda_: lame_lambda, mu: lame_mu, A: section_area})
)
load_at_end = normal_1.subs(
    {A: section_area, lambda_: lame_lambda, mu: lame_mu, p: length}
)


def stress_pk_2_manual(p: float):
    term = 0.1 * np.cos(p) + 1.2
    num = np.log(term)
    den = term**2
    return lame_lambda * num / den - lame_mu / (den - 1)


def stress_1_pk_manual(p: float):
    return stress_pk_2_manual(p) * (1.2 + 0.1 * np.cos(p))


def normal_force_manual(p: float):
    return stress_1_pk_manual(p) * section_area


bar = BarInputNonLiner(
    young_modulus=young_modulus,
    section_area=section_area,
    length=length,
    degree=degree,
    n_elements=n_elements,
    load_function=dist_load_num,
    poisson=poisson,
    load_at_end=load_at_end,
)

convernge_criteria = NewtonRaphsonConvergenceParam(
    n_load_steps=10,
    max_iterations=100,
    convergence_criteria=ConvergenceCriteria.FORCE,
    precision=1e-7,
)

analysis = BarAnalysis(
    inputs=C0BarAnalysisInput(
        bar_input=bar,
        ecsi_placement_pts_function=calc_ecsi_placement_coords_gauss_lobato,
    ),
    convergence_crit=convernge_criteria,
)

analysis_c1 = BarAnalysis(
    inputs=C1BarAnalysisInput(bar_input=bar),
    convergence_crit=convernge_criteria,
)


res = analysis.bar_result
# res_c1 = analysis_c1.bar_result

ax: plt.Axes
fig, ax = plt.subplots()

col_pts = analysis.pre_process.collocation_pts
col_pts_c1 = analysis_c1.pre_process.collocation_pts
x_knots_c1 = analysis_c1.pre_process.x_knots_global
col_pts_sorted = np.sort(col_pts)
element_incidence = analysis.pre_process.incidence_matrix
element_incidence_c1 = analysis_c1.pre_process.incidence_matrix
col_pt_el1 = col_pts[element_incidence[1]]
col_pt_el1_c1 = col_pts_c1[element_incidence_c1[1]]


def check_p_conversion():
    return


# ecsi_pts = analysis.pre_process.ecsi_placement_pts
ecsi_pts_test = np.linspace(-1, 1, 11)
n_ecsi = analysis.n_ecsi(ecsi_pts_test)
b_ecsi = analysis.b_ecsi(ecsi_pts_test)
b_ecsi_c1 = analysis_c1.b_ecsi(ecsi_pts_test)
n_ecsi_c1 = analysis_c1.n_ecsi(ecsi_pts_test)
ecsi_pts_test_linear_interp = np.interp(ecsi_pts_test, [-1, 1], col_pt_el1[:2])
ecsi_pts_test_linear_interp_c1 = np.interp(ecsi_pts_test, [-1, 1], x_knots_c1[:2])
p = n_ecsi.T @ col_pt_el1
j = b_ecsi.T @ col_pt_el1
j_c1 = b_ecsi_c1.T @ col_pt_el1_c1
p1 = n_ecsi_c1.T @ col_pt_el1_c1
res_df = analysis.result_df()

ax.plot(
    col_pts_sorted,
    [displacement_analytical_num(pt) for pt in col_pts_sorted],
    label="analytical",
)
ax.plot(res_df[X_COORD], res_df[NUM_DISPLACEMENT], label="num")
