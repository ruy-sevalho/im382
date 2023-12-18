from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Protocol
import numpy as np
import numpy.typing as npt
import pandas as pd
from bar_1d import BarInput
from bar_1d import BarResults
from nomeclature import (
    X_COORD,
    ANALYTICAL_DISPLACEMENT,
    ANALYTICAL_STRAIN,
)

from post_process import (
    calc_l2_error_norm,
    calc_l2_h1_error_norms,
)


class BarModel(Protocol):
    inputs: BarInput

    @property
    def bar_result(self) -> BarResults:
        ...

    @property
    def result_dataframe(self) -> pd.DataFrame:
        ...


@dataclass
class BarAnalysis:
    model: BarModel
    displacement_analytical: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]
    displacement_derivative_analytical: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]

    @cached_property
    def bar_result(self):
        return self.model.bar_result

    @cached_property
    def results(self):
        df = self.model.result_dataframe
        df[ANALYTICAL_DISPLACEMENT] = df[X_COORD].apply(self.displacement_analytical)
        df[ANALYTICAL_STRAIN] = df[X_COORD].apply(
            self.displacement_derivative_analytical
        )
        return df

    @cached_property
    def energy_norm_aprox_sol(self):
        return (
            self.bar_result.knots_displacements
            @ self.bar_result.global_stiffness_matrix
            @ self.bar_result.knots_displacements
        ) ** 0.5

    @cached_property
    def h1_error(self):
        return self.error_norms.h1_error_norm

    @cached_property
    def l2_error(self):
        return self.error_norms.l2_error_norm

    @cached_property
    def error_norms(self):
        return calc_l2_h1_error_norms(
            analytical_solution_function=self.displacement_analytical,
            analytical_derivative_solution_function=self.displacement_derivative_analytical,
            p_init_global_coords=self.bar_result.p_knots_global_complete,
            knot_displacements=self.bar_result.knots_displacements,
            integration_weights=self.bar_result.integration_weights,
            n_ecsi=self.bar_result.n_ecsi,
            b_ecsi=self.bar_result.b_ecsi,
            incidence_matrix=self.bar_result.incidence_matrix,
            det_j=self.bar_result.det_j,
        )
