import numpy as np
import pandas as pd


def calc_approx_value(
    x_knots_global: np.array,
    element_incidence_matrix: np.array,
    knot_displacements: np.array,
    esci_matrix_: np.array,
    ecsi_calc_pts: np.array = None,
    factor: float = 1,
    result_name: str = "res",
):
    results = pd.DataFrame()

    for e in element_incidence_matrix:
        x_element_global = np.interp(ecsi_calc_pts, [-1, 1], x_knots_global[e[:2]])
        element_displacements = knot_displacements[e]
        res = np.array([factor * col @ element_displacements for col in esci_matrix_.T])
        results = pd.concat(
            [results, pd.DataFrame({"x": x_element_global, result_name: res})]
        )

    return results
