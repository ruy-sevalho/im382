from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass
class BarInput:
    young_modulus: float
    section_area: float
    length: float
    degree: int
    n_elements: int
    load_function: Callable[[float], float]


@dataclass
class BarInputNonLiner(BarInput):
    poisson: float
    load_at_end: float
