import numpy as np
import matplotlib.pyplot as plt

from truss2d import TrussInputs
from newton_raphson import NewtonRaphsonConvergenceParam
from truss_large_deformation_plastic import Analysis


coords = np.array([[0, 0], [0, 120], [24, 120], [120, 120]])  ## mm
incidence = np.array([[0, 1], [1, 2], [2, 3]])
boundary_conditions = np.array([[0, 0], [0, 1], [3, 0], [3, 1]])
loads = np.array([[2, 1, -100]])  # N
truss = TrussInputs(
    coords=coords,
    incidences=incidence,
    boundary_conditions=boundary_conditions,
    loads=loads,
    young_modulus=210000,  # N / mm2
    isotropic_hardening=1800,  # kN / mm2
    yield_stress=25000,  # N / mm2
    section_area=1,  # mm2
)
convergence_criteria = NewtonRaphsonConvergenceParam(
    n_load_steps=100, max_iterations=100
)

ax: plt.Axes
fig, ax = plt.subplots()
x, y = coords[:, 0], coords[:, 1]
ax.scatter(x, y)
[plt.text(i, j, f"{n}") for n, (i, j) in enumerate(zip(x, y))]

for i, e in enumerate(incidence):
    ax.plot(coords[e][:, 0], coords[e][:, 1], label=i)
fig.legend()

analysis = Analysis(truss=truss, newton_raphson_convergence_crit=convergence_criteria)
res = analysis.results_rewton_raphson_plastic
