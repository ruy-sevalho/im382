from dataclasses import replace
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
from bar_1d import BarInput
from c0_basis import (
    C0BarAnalysis,
    calc_ecsi_placement_coords_gauss_lobato,
)
from c1_basis import C1BarAnalysis
from nomeclature import NUM_DISPLACEMENT, X_COORD, NUM_STRAIN


young_modulus = 100e9
section_area = 1.0e-4
length = 1.0
poly_degree = 1
n_elements = 2
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
    n_elements=2,
    load_function=load_function,
)


n_elements_cases = (4, 6, 8, 10)
degrees = (3, 4, 5, 6)

h_study_c0 = dict()
p_study_c0 = dict()
h_study_c1 = dict()
p_study_c1 = dict()

for n_elements in n_elements_cases:
    h_study_c0[n_elements] = C0BarAnalysis(
        input=replace(bar_input_h_study, n_elements=n_elements),
        displacement_analytical=displacement_analytical,
        ecsi_placement_coords_function=calc_ecsi_placement_coords_gauss_lobato,
    )
    h_study_c1[n_elements] = C1BarAnalysis(
        input=replace(bar_input_h_study, n_elements=n_elements),
        displacement_analytical=displacement_analytical,
    )


for degree in degrees:
    p_study_c0[degree] = C0BarAnalysis(
        input=replace(
            bar_input_p_study,
            degree=degree,
        ),
        displacement_analytical=displacement_analytical,
        ecsi_placement_coords_function=calc_ecsi_placement_coords_gauss_lobato,
    )
    p_study_c1[degree] = C1BarAnalysis(
        input=replace(bar_input_p_study, degree=degree),
        displacement_analytical=displacement_analytical,
    )


degrees_freedom_h = tuple(
    case.bar_result.n_degrees_freedom for case in h_study_c0.values()
)
l2_error_h_c0 = tuple(case.l2_error for case in h_study_c0.values())
l2_error_h_c1 = tuple(case.l2_error for case in h_study_c1.values())
energy_diff_h_c0 = tuple(
    analytical_energy - case.energy_norm_aprox_sol for case in h_study_c0.values()
)
energy_diff_h_c1 = tuple(
    analytical_energy - case.energy_norm_aprox_sol for case in h_study_c1.values()
)

degrees_freedom_p = tuple(
    case.bar_result.n_degrees_freedom for case in p_study_c0.values()
)
l2_error_p_c0 = tuple(case.l2_error for case in p_study_c0.values())
l2_error_p_c1 = tuple(case.l2_error for case in p_study_c1.values())
energy_diff_p_c0 = tuple(
    analytical_energy - case.energy_norm_aprox_sol for case in p_study_c0.values()
)
energy_diff_p_c1 = tuple(
    analytical_energy - case.energy_norm_aprox_sol for case in p_study_c1.values()
)


ax: plt.Axes
fig, ax = plt.subplots()
ax.set_title("h ref")
ax.set_xlabel("degrees of freedom")
ax.set_ylabel("energy diff")
ax.loglog(degrees_freedom_h, energy_diff_h_c0, label="c0")
ax.loglog(degrees_freedom_h, energy_diff_h_c1, label="c1")
fig.legend()

ax2: plt.Axes
fig2, ax2 = plt.subplots()
ax2.set_title("p ref")
ax2.set_xlabel("degrees of freedom")
ax2.set_ylabel("energy diff")
ax2.loglog(degrees_freedom_p, energy_diff_p_c0, label="c0")
ax2.loglog(degrees_freedom_p, energy_diff_p_c1, label="c1")
fig2.legend()

ax3: plt.Axes
fig3, ax3 = plt.subplots()
ax3.set_title("h ref")
ax3.set_xlabel("degrees of freedom")
ax3.set_ylabel("l2 error")
ax3.loglog(degrees_freedom_h, l2_error_h_c0, label="c0")
ax3.loglog(degrees_freedom_h, l2_error_h_c1, label="c1")
fig3.legend()

ax4: plt.Axes
fig4, ax4 = plt.subplots()
ax4.set_title("p ref")
ax4.set_xlabel("degrees of freedom")
ax4.set_ylabel("l2 error")
ax4.loglog(degrees_freedom_p, l2_error_p_c0, label="c0")
ax4.loglog(degrees_freedom_p, l2_error_p_c1, label="c1")
fig4.legend()


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

# bar_results = [x.bar_result for x in p_study.values()]
# results = [x.results for x in p_study.values()]
# results_gl = [x.results for x in p_study_gauss_lobato.values()]

# fig_disp, disp_plot = plt.subplots()

# disp_plot.set_xlabel("x(m)")
# disp_plot.set_ylabel("u(m)")
# disp_plot.plot(
#     results[0][X_COORD],
#     [displacement_analytical(xs) for xs in results[0][X_COORD]],
#     label="analytical",
# )


# def plot_result(ax, result_df, key: str, name: str = "disp"):
#     ax.plot(result_df[X_COORD], result_df[key], label=name)


# def plot_results(ax, result_dfs, key: str, name: str = "disp"):
#     for i, r in enumerate(result_dfs):
#         plot_result(disp_plot, r, key, f"{name} {i+2}")


# plot_results(disp_plot, result_dfs=results, key=NUM_DISPLACEMENT, name="std")

# plot_results(disp_plot, result_dfs=results_gl, key=NUM_DISPLACEMENT, name="gl")

# fig_disp.legend()

# fig_strain, strain_plot = plt.subplots()

# for i, r in enumerate(results):
#     plot_result(strain_plot, r, NUM_STRAIN, f"{i+2}")

# fig_strain.legend()
