from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt

from nomeclature import H1_ERROR_NORM, H1_SOL_NORM, L2_ERROR_NORM, L2_SOL_NORM
import pandas as pd


@dataclass
class BarInput:
    young_modulus: float
    section_area: float
    length: float
    degree: int
    n_elements: int
    load_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclass
class BarInputNonLiner(BarInput):
    poisson: float
    load_at_end: float


@dataclass
class BarInputDynamics:
    young_modulus: float
    poisson: float
    density: float
    section_area: float
    length: float
    degree: int
    n_elements: int


class BarResults(Protocol):
    det_j: float
    p_knots_global: npt.NDArray[np.float64]
    n_degrees_freedom: int
    element_stiffness_matrix: npt.NDArray[np.float64]
    incidence_matrix: npt.NDArray[np.float64]
    global_stiffness_matrix: npt.NDArray[np.float64]
    load_vector: npt.NDArray[np.float64]
    knots_displacements: npt.NDArray[np.float64]
    reaction: npt.NDArray[np.float64]
    integration_weights: npt.NDArray[np.float64]
    n_ecsi: npt.NDArray[np.float64]
    b_ecsi: npt.NDArray[np.float64]
    p_knots_global_complete: npt.NDArray[np.float64]
    n_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    b_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclass
class EnergyNormsAndErrors:
    l2_error_norm: float
    l2_sol_norm: float
    h1_error_norm: float
    h1_sol_norm: float

    @cached_property
    def df(self):
        return pd.DataFrame(
            {
                L2_ERROR_NORM: [self.l2_error_norm],
                L2_SOL_NORM: [self.l2_sol_norm],
                H1_ERROR_NORM: [self.h1_error_norm],
                H1_SOL_NORM: [self.h1_sol_norm],
            }
        )


@dataclass
class ErrorDif:
    l2: float
    h1: float
