from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt


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
