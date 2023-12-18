import numpy as np
import matplotlib.pyplot as plt


from truss2d import TrussInputs
from truss_linear import LinearTruss

x = 100 * 3**0.5
coords = np.array([[0, 0], [x, 0], [2 * x, 0], [x, 100]])
ax: plt.Axes
fig, ax = plt.subplots()

for coord in coords:
    ax.plot(coord[0], coord[1])

incidence = np.array(
    [
        [0, 3],
        [1, 3],
        [2, 3],
    ]
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
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ]
)
loads = np.array(
    [
        [3, 1, 6.4],
    ]
)
truss = TrussInputs(
    coords=coords,
    incidences=incidence,
    boundary_conditions=boundary_conditions,
    loads=loads,
    young_modulus=1000,
    isotropic_hardening=1000 / 9,
    yield_stress=5,
    section_area=1,
)
analysis = LinearTruss(truss=truss)
analysis.run
