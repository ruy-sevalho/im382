from dataclasses import asdict, replace
import math
from matplotlib import pyplot as plt
import numpy as np
import sympy as sy

from bar_1d import BarInput
from c1_basis import C1BarAnalysis, c1_bar

young_modulus = 100e9
section_area = 1.0e-4
length = 1.0
degree = 5
n_elements = 3
x = sy.symbols("x")


def load_function(x: float):
    return 1000 * np.sin(np.pi / 2 * x)


def displacement_analytical(x):
    return 4000 * np.sin(np.pi * x / 2) / (np.pi**2 * young_modulus * section_area)


def strain_analytical(x):
    return 2000 * np.cos(np.pi * x / 2) / (np.pi * young_modulus * section_area)


displacement_analytical_symb = (
    4000 * sy.sin(sy.pi * x / 2) / (sy.pi**2 * young_modulus * section_area)
)

analytical_energy = (
    sy.integrate(
        young_modulus * section_area * sy.diff(displacement_analytical_symb, x) ** 2,
        (x, 0, length),
    )
    ** 0.5
).evalf()

bar_input_h_study = BarInput(
    young_modulus=young_modulus,
    section_area=section_area,
    length=length,
    degree=3,
    n_elements=4,
    load_function=load_function,
)

bar_input_p_study = BarInput(
    young_modulus=young_modulus,
    section_area=section_area,
    length=length,
    degree=3,
    n_elements=4,
    load_function=load_function,
)


n_elements_cases = (4, 8, 16, 32)
degrees = (3, 4, 5, 6)
degrees_freedom = (5, 7, 9, 11, 13)
h_study: dict[int, C1BarAnalysis] = dict()
p_study: dict[int, C1BarAnalysis] = dict()

for n_elements in n_elements_cases:
    h_study[n_elements] = C1BarAnalysis(
        input=replace(bar_input_h_study, n_elements=n_elements),
        displacement_analytical=displacement_analytical,
    )

for degree in degrees:
    p_study[degree] = C1BarAnalysis(
        input=replace(bar_input_p_study, degree=degree),
        displacement_analytical=displacement_analytical,
    )

degrees_freedom_h = tuple(case.bar_result.n_degrees_total for case in h_study.values())
l2_error_h = tuple(case.l2_error for case in h_study.values())
energy_h_ = tuple(case.energy_norm_aprox_sol for case in h_study.values())
energy_diff_h = tuple(
    analytical_energy - case.energy_norm_aprox_sol for case in h_study.values()
)

degrees_freedom_p = tuple(case.bar_result.n_degrees_total for case in p_study.values())
l2_error_p = tuple(case.l2_error for case in p_study.values())
energy_diff_p = tuple(
    analytical_energy - case.energy_norm_aprox_sol for case in p_study.values()
)


fig, ax = plt.subplots()
energy_h = (0.1405, 0.1419, 0.1421)
errors_h = (0.0018, 4.5764e-04, 2.0332e-04)
errors_p = (7.3568e-05, 8.0004e-10, 4.4409e-16)
ax.loglog(degrees_freedom_h, energy_diff_h, label="ref h")
ax.loglog(degrees_freedom_p, energy_diff_p, label="ref p")
fig.legend()

fig2, ax2 = plt.subplots()

ax2.loglog(degrees_freedom_h, l2_error_h, label="ref h")
ax2.loglog(degrees_freedom_p, l2_error_p, label="ref p")
fig.legend()
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
