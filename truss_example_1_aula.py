import numpy as np

from truss2d import TrussInputs

coords = np.array([[0, 0], [0, 4], [3, 0], [3, 4], [6, 0], [6, 4]])
incidence = np.array(
    [[0, 1], [2, 3], [4, 5], [0, 2], [2, 4], [1, 3], [3, 5], [0, 3], [4, 3]]
)
# groups = np.array([3, 4, 2])
# materials = np.array(
#     [
#         [210e9, 120e6, 80e6],
#         [80e9, 70e6, 60e6],
#         [70e9, 85e6, 56e6],
#     ]
# )
boundary_conditions = np.array(
    [
        [0, 0],
        [0, 1],
        [2, 1],
        [4, 0],
        [4, 1],
    ]
)
loads = np.array(
    [
        [1, 1, -5000],
        [3, 1, -10000],
        [5, 1, -5000],
        [5, 0, 3000],
    ]
)
truss = TrussInputs(
    coords=coords,
    incidences=incidence,
    boundary_conditions=boundary_conditions,
    loads=loads,
    young_modulus=70e9,
    isotropic_hardening=1000 / 9,
    yield_stress=4,
    section_area=3e-4,
)
truss.run
