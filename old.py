# def calc_element_stiffness_matrix(
#     stiffness: float,
#     b_esci_matrix: npt.NDArray[Any,   np.dtype[np.float64]],
#     int_weights: npt.NDArray[Any,   np.dtype[np.float64]],
#     # integration_points: npt.NDArray[Any,   np.dtype[np.float64]],
#     size: float,
# ):
#     placement_pts = np.array([-1, 0, 1, 0])
#     n = b_esci_matrix.shape[0]
#     if n > 4:
#         placement_pts = np.concatenate([placement_pts, np.zeros(n - 4)])
#     det_js = b_esci_matrix.T @ placement_pts  # * size / 2
#     return (
#         np.sum(
#             np.outer(b_col, b_col) * w / det_j
#             for b_col, w, det_j in zip(b_esci_matrix.T, int_weights, det_js)
#         )
#         * stiffness
#     )

# def calc_load_vector_variable_det_j(
#     collocation_pts: npt.NDArray[Any,   np.dtype[np.float64]],
#     incidence_matrix: npt.NDArray[Any,   np.dtype[np.float64]],
#     n_ecsi_function: Callable[[npt.NDArray[Any,   np.dtype[np.float64]]], npt.NDArray[Any,   np.dtype[np.float64]]],
#     b_ecsi_function: Callable[[npt.NDArray[Any,   np.dtype[np.float64]]], npt.NDArray[Any,   np.dtype[np.float64]]],
#     load_function: Callable[[npt.NDArray[Any,   np.dtype[np.float64]]], npt.NDArray[Any,   np.dtype[np.float64]]],
#     intorder: int,
# ):
#     ecsi_local_int_pts, weight_int_pts = get_points_weights(
#         0, 0, intorder, IntegrationTypes.GJ, "x"
#     )
#     n_ecsi = n_ecsi_function(calc_pts_coords=ecsi_local_int_pts)  # type: ignore
#     b_ecsi = b_ecsi_function(calc_pts_coords=ecsi_local_int_pts)  # type: ignore
#     global_load_vector = np.zeros(incidence_matrix[-1, -1] + 1)
#     for i, element_incidence in enumerate(incidence_matrix):
#         pts = collocation_pts[element_incidence]
#         p_coords = n_ecsi.T @ pts
#         det_j = b_ecsi.T @ pts
#         load_at_x = load_function(p_coords)
#         load_vector = np.array(
#             [np.sum(row * weight_int_pts * load_at_x * det_j) for row in n_ecsi]
#         )
#         global_load_vector[element_incidence] += load_vector
#     return global_load_vector
