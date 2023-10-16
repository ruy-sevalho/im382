from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt
from c0_basis import C0BarInput, c0_bar, calc_displacements, calc_strain


young_modulus = 100e9
section_area = 1.0e-4
length = 1.0
poly_degree = 2
n_elements = 5


def load_function(x: float):
    return 1000 * np.sin(np.pi / 2 * x)


def displacement_analytical(x):
    return 4000 * np.sin(np.pi * x / 2) / (np.pi**2 * young_modulus * section_area)


def strain_analytical(x):
    return 2000 * np.cos(np.pi * x / 2) / (np.pi * young_modulus * section_area)


bar_input = C0BarInput(
    young_modulus=young_modulus,
    section_area=section_area,
    length=length,
    poly_degree=poly_degree,
    n_elements=n_elements,
    load_function=load_function,
)

# bar_result = c0_bar(
#     young_modulus=young_modulus,
#     section_area=section_area,
#     length=length,
#     poly_degree=poly_degree,
#     n_elements=n_elements,
#     load_function=load_function,
# )

bar_result = bar_input.calc_results()

strain = calc_strain(
    degree=bar_input.poly_degree,
    det_j=bar_result.det_j,
    x_knots_global=bar_result.x_knots_global,
    x_knots_local=bar_result.x_knots_local,
    element_incidence_matrix=bar_result.incidence_matrix,
    knot_displacements=bar_result.knots_displacements,
)

strain = calc_strain(
    degree=bar_input.poly_degree,
    det_j=bar_result.det_j,
    x_knots_global=bar_result.x_knots_global,
    x_knots_local=bar_result.x_knots_local,
    element_incidence_matrix=bar_result.incidence_matrix,
    knot_displacements=bar_result.knots_displacements,
)

displacements = calc_displacements(
    degree=bar_input.poly_degree,
    x_knots_global=bar_result.x_knots_global,
    x_knots_local=bar_result.x_knots_local,
    element_incidence_matrix=bar_result.incidence_matrix,
    knot_displacements=bar_result.knots_displacements,
)

fig, (strain_plot, disp_plot) = plt.subplots(2, sharex=True)
disp_plot.set_xlabel("x(m)")
strain_plot.plot(
    strain.x, [strain_analytical(xs) for xs in displacements.x], label="analitical"
)
strain_plot.plot(strain.x, strain.strain, label="num")
strain_plot.set_ylabel("strain")
disp_plot.plot(
    displacements.x,
    [displacement_analytical(xs) for xs in displacements.x],
    label="analytical",
)
disp_plot.plot(displacements.x, displacements.displacement, label="num")
disp_plot.set_ylabel("displacement")
fig.legend()
