from truss2d import calc_element_lengths

import pytest as pt
import numpy as np


@pt.mark.parametrize(
    "coords, incidence, expected",
    [
        [
            np.array([[0, 0], [0, 4], [3, 0], [3, 4], [6, 0], [6, 4]]),
            np.array(
                [[0, 1], [2, 3], [4, 5], [0, 2], [2, 4], [1, 3], [3, 5], [0, 3], [4, 3]]
            ),
            np.array(
                [
                    [4, 1, 0],
                    [4, 1, 0],
                    [4, 1, 0],
                    [3, 0, 1],
                    [3, 0, 1],
                    [3, 0, 1],
                    [3, 0, 1],
                    [5, 0.8, 0.6],
                    [5, 0.8, -0.6],
                ]
            ),
        ]
    ],
)
def test_calc_element_lengths(coords, incidence, expected):
    assert calc_element_lengths(coords=coords, incidence=incidence) == pt.approx(
        expected
    )
