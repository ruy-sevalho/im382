import numpy as np

from truss2d import TrussInputs
from newton_raphson import NewtonRaphsonConvergenceParam
from large_deformation_plasticity import Analysis


coords = np.array([[0, 0], [100, 100]])  ## mm
incidence = np.array([[0, 1]])
boundary_conditions = np.array([[0, 0], [0, 1], [1, 0]])
loads = np.array([[1, 1, -1]])  # N
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
analysis.results
