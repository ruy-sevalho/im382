import numpy as np
import matplotlib.pyplot as plt

from truss2d import TrussInputs
from newton_raphson import NewtonRaphsonConvergenceParam
from truss_large_deformation_plastic import Analysis


coords = np.array([[0, 0], [100, 100]])  ## mm
incidence = np.array([[0, 1]])
boundary_conditions = np.array([[0, 0], [0, 1], [1, 0]])
loads = np.array([[1, 1, -100]])  # N
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


analysis = Analysis(truss=truss, convergence_crit=convergence_criteria)
res = analysis.results

ax: plt.Axes
fig, ax = plt.subplots()
fig.set_dpi(600)
x, y = coords[:, 0], coords[:, 1]
[plt.text(i, j, f"{n}", size=2) for n, (i, j) in enumerate(zip(x, y))]
deformed_coords = analysis.deformed_shape()
for e in incidence:
    ax.plot(
        coords[e][:, 0],
        coords[e][:, 1],
        linewidth=0.2,
        color="blue",
        linestyle="dashed",
    )
    ax.plot(
        deformed_coords[e][:, 0], deformed_coords[e][:, 1], linewidth=0.2, color="red"
    )
