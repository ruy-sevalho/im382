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
    C0BarAnalysis,
)
from nomeclature import NUM_DISPLACEMENT, X_COORD


young_modulus = 210e9
poisson = 0.3
lame_lambda = calc_lambda(young_modulus=young_modulus, poisson=poisson)
lame_mu = calc_mu(young_modulus=young_modulus, poisson=poisson)
section_area = 1.0e-3
length = 4.0
degree = 1
n_elements = 2
x = sy.symbols("x")
p, lambda_, mu, cp, A = sy.symbols("p lambda mu C_p A")
energy_norm = calc_energy_norm(
    mu=lame_mu, lambda_=lame_lambda, area=section_area, length=length
)
displacement_analytical_symb = 0.2 * p + 0.1 * sy.sin(p)
displacement_analytical_num = sy.lambdify(p, displacement_analytical_symb)
grad_p = 1 + sy.diff(displacement_analytical_symb, p)
cauchy_green = grad_p**2
stress_piolla_kirchhoff_2: sy.Expr = (
    mu * (1 - 1 / cp) + lambda_ * sy.ln(sy.sqrt(cp)) / cp
)
stress_piolla_kirchhoff_2 = stress_piolla_kirchhoff_2.subs(cp, cauchy_green)
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

bar = BarInputNonLiner(
    young_modulus=young_modulus,
    section_area=section_area,
    length=length,
    degree=degree,
    n_elements=n_elements,
    load_function=dist_load_num,
    poisson=poisson,
)

convernge_criteria = NewtonRaphsonConvergenceParam(
    n_load_steps=10,
    max_iterations=100,
    convergence_criteria=ConvergenceCriteria.FORCE,
    precision=1e-7,
)

analysis = C0BarAnalysis(
    convergence_crit=convernge_criteria,
    bar_input=bar,
    x_knots_local_function=calc_ecsi_placement_coords_gauss_lobato,
)

res = analysis.bar_result

ax: plt.Axes
fig, ax = plt.subplots()

col_pts = analysis.pre_process.collocation_pts

ax.plot(col_pts, [displacement_analytical_num(pt) for pt in col_pts])


# load_at_end_2 = dist_load_2.subs(
#     {A: section_area, lambda_: lame_lambda, mu: lame_mu, p: length}
# )

# dist_load = -sy.integrate(normal_1, p)
# analytical_energy = (
#     sy.integrate(

#         young_modulus * section_area * sy.diff(displacement_analytical_symb, x) ** 2,

#         (x, 0, length),
#     )

#     ** 0.5

# ).evalf()


# bar_input_h_study = BarInput(
#     young_modulus=young_modulus,
#     section_area=section_area,

#     length=length,

#     degree=1,

#     n_elements=4,
#     load_function=load_function,
# )


# bar_input_p_study = BarInput(
#     young_modulus=young_modulus,
#     section_area=section_area,

#     length=length,

#     degree=2,

#     n_elements=2,
#     load_function=load_function,
# )


# n_elements_cases = (4, 6, 8, 10, 12)

# degrees = (2, 3, 4, 5, 6)

# degrees_freedom = (5, 7, 9, 11, 13)

# h_study = dict()
# p_study = dict()


# for n_elements in n_elements_cases:

#     h_study[n_elements] = C0BarAnalysis(

#         input=replace(bar_input_h_study, n_elements=n_elements),
#         displacement_analytical=displacement_analytical,
#     )


# for degree in degrees:

#     p_study[degree] = C0BarAnalysis(

#         input=replace(bar_input_p_study, degree=degree),
#         displacement_analytical=displacement_analytical,
#     )


# l2_error_h = tuple(case.l2_error for case in h_study.values())

# energy_h_ = tuple(case.energy_norm_aprox_sol for case in h_study.values())

# energy_diff_h = tuple(

#     analytical_energy - case.energy_norm_aprox_sol for case in h_study.values()
# )


# l2_error_p = tuple(case.l2_error for case in p_study.values())
# energy_diff_p = tuple(

#     analytical_energy - case.energy_norm_aprox_sol for case in p_study.values()
# )


# fig, ax = plt.subplots()

# energy_h = (0.1405, 0.1419, 0.1421)

# errors_h = (0.0018, 4.5764e-04, 2.0332e-04)

# errors_p = (7.3568e-05, 8.0004e-10, 4.4409e-16)

# ax.loglog(degrees_freedom, energy_diff_h, label="ref h")

# ax.loglog(degrees_freedom, energy_diff_p, label="ref p")
# fig.legend()


# fig2, ax2 = plt.subplots()


# ax2.loglog(degrees_freedom, l2_error_h, label="ref h")

# ax2.loglog(degrees_freedom, l2_error_p, label="ref p")
# fig.legend()

# fig, (strain_plot, disp_plot) = plt.subplots(2, sharex=True)

# disp_plot.set_xlabel("x(m)")
# strain_plot.plot(

#     results[X_COORD],

#     [strain_analytical(xs) for xs in results[X_COORD]],

#     layel="analitical",
# )

# strain_plot.plot(results[X_COORD], results[NUM_STRAIN], label="num")

# strain_plot.set_ylabel("strain")
# disp_plot.plot(

#     results[X_COORD],

#     [displacement_analytical(xs) for xs in results.x],

#     label="analytical",
# )

# disp_plot.plot(results[X_COORD], results[NUM_DISPLACEMENT], label="num")

# disp_plot.set_ylabel("displacement")
# fig.legend()
