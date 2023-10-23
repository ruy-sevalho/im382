from dataclasses import asdict, dataclass
from typing import Callable


@dataclass
class BarInput:
    young_modulus: float
    section_area: float
    length: float
    degree: int
    n_elements: int
    load_function: Callable[[float], float]
