import numpy as np
import matplotlib.pyplot as plt

from truss2d import TrussInputs
from newton_raphson import ConvergenceCriteria, NewtonRaphsonConvergenceParam
from truss_large_deformation_plastic import Analysis


coords = np.array([[0, 0], [10, 0]])  ## mm
incidence = np.array([[0, 1]])
boundary_conditions = np.array([[0, 0], [0, 1], [1, 1]])
loads = np.array([[1, 0, 21000000]])  # N
truss = TrussInputs(
    coords=coords,
    incidences=incidence,
    boundary_conditions=boundary_conditions,
    loads=loads,
    young_modulus=70e9,  # N / mm2
    isotropic_hardening=85e6,  # kN / mm2
    yield_stress=56e6,  # N / mm2
    section_area=3e-4,  # mm2
)
convergence_criteria = NewtonRaphsonConvergenceParam(
    n_load_steps=1000,
    max_iterations=400,
    precision=1e-3,
    convergence_criteria=ConvergenceCriteria.DISPLACEMENT,
)


analysis = Analysis(truss=truss, convergence_crit=convergence_criteria)
res = analysis.results_rewton_raphson_plastic
ps = res.plastic_strains
stress = res.stresses
disp = res.displacements
loads = res.loads
ax: plt.Axes
fig, ax = plt.subplots()
fig.set_dpi(600)

ax.plot(disp[:, 0], loads[:, 0])


# ax: plt.Axes
# fig, ax = plt.subplots()
# fig.set_dpi(600)
# x, y = coords[:, 0], coords[:, 1]
# [plt.text(i, j, f"{n}", size=2) for n, (i, j) in enumerate(zip(x, y))]
# deformed_coords = analysis.deformed_shape(100000)
# for e in incidence:
#     ax.plot(
#         coords[e][:, 0],
#         coords[e][:, 1],
#         linewidth=0.2,
#         color="blue",
#         linestyle="dashed",
#     )
#     ax.plot(
#         deformed_coords[e][:, 0], deformed_coords[e][:, 1], linewidth=0.2, color="red"
#     )
