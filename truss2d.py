from dataclasses import dataclass, asdict
from functools import cached_property

import numpy as np

from type_alias import Array


@dataclass
class TrussInputs:
    coords: Array
    incidences: Array
    boundary_conditions: Array
    loads: Array
    young_modulus: float
    isotropic_hardening: float
    yield_stress: float
    section_area: float


@dataclass
class LinearTruss:
    truss: TrussInputs

    @cached_property
    def run(self):
        return calc_truss(**asdict(self.truss))


def calc_truss(
    coords: Array,
    incidences: Array,
    boundary_conditions: Array,
    loads: Array,
    young_modulus: float,
    isotropic_hardening: float,
    yield_stress: float,
    section_area: float,
):
    load_steps = np.array([5 / 6.4, 1])
    lengths = calc_element_lengths(coords=coords, incidences=incidences)
    n_nodes = coords.shape[0]
    nodal_dofs = 2
    n_elements = incidences.shape[0]
    nodal_dofs_mapping = np.ones((nodal_dofs, n_nodes), dtype=int)
    for bc in boundary_conditions:
        nodal_dofs_mapping[bc[1], bc[0]] = 0

    n_free_dofs = np.sum(nodal_dofs_mapping.flatten())
    free_dofs_array = np.arange(n_free_dofs)
    total_dofs = n_nodes * nodal_dofs
    rest_dofs = total_dofs - n_free_dofs

    free_node_numbering = 0
    restricted_node_numbering = n_free_dofs
    for i in range(n_nodes):
        for j in range(nodal_dofs):
            if nodal_dofs_mapping[j, i] == 1:
                nodal_dofs_mapping[j, i] = free_node_numbering
                free_node_numbering += 1
            elif nodal_dofs_mapping[j, i] == 0:
                nodal_dofs_mapping[j, i] = restricted_node_numbering
                restricted_node_numbering += 1

    global_load = assembly_global_load(
        loads=loads, total_dofs=total_dofs, nodal_dofs_control=nodal_dofs_mapping
    )
    plastic_strain = np.zeros(n_elements)
    alpha = np.zeros(n_elements)

    global_stiffness = assembly_global_stiffness(
        element_lengths=lengths,
        incidences=incidences,
        nodal_dofs_mapping=nodal_dofs_mapping,
        young_modulus=young_modulus,
        section_area=section_area,
        total_dofs=total_dofs,
    )
    global_stiffness_free = global_stiffness[
        free_dofs_array[:, np.newaxis], free_dofs_array[np.newaxis, :]
    ]

    load_not_finished = True
    step = 0
    while load_not_finished:
        step_global_load = global_load * load_steps[step]
        step_global_load_free = step_global_load[:n_free_dofs]

        displacements = np.linalg.solve(global_stiffness_free, step_global_load_free)
        load_not_finished = False
    return


def calc_strain(element_length: Array):
    return 1 / element_length[0] * ()


def calc_element_lengths(coords: Array, incidences: Array):
    lengths = np.zeros((incidences.shape[0], 3))
    lengths[:, 0] = (
        (coords[incidences[:, 0], 0] - coords[incidences[:, 1], 0]) ** 2
        + (coords[incidences[:, 0], 1] - coords[incidences[:, 1], 1]) ** 2
    ) ** 0.5
    lengths[:, 1] = (
        coords[incidences[:, 1], 1] - coords[incidences[:, 0], 1]
    ) / lengths[:, 0]
    lengths[:, 2] = (
        coords[incidences[:, 1], 0] - coords[incidences[:, 0], 0]
    ) / lengths[:, 0]
    return lengths


def assembly_global_load(loads: Array, total_dofs: int, nodal_dofs_control: Array):
    global_load = np.zeros(total_dofs)
    for load in loads:
        elem_eqs = nodal_dofs_control[int(load[1]), int(load[0])]
        global_load[elem_eqs] = load[2]
    return global_load


def calc_element_stiffness(
    element_length: Array, elastoplastic_modulus: float, section_area: float
) -> Array:
    cc = element_length[2] ** 2
    ss = element_length[1] ** 2
    cs = element_length[1] * element_length[2]
    return (
        elastoplastic_modulus
        * section_area
        * np.array(
            [
                [cc, cs, -cc, -cs],
                [cs, ss, -cs, -ss],
                [-cc, -cs, cc, cs],
                [-cs, -ss, cs, ss],
            ]
        )
        / element_length[0]
    )


def assembly_rotation_matrix(cos: float, sin: float):
    return np.array([[cos, sin, 0, 0], [0, 0, cos, sin]])


def calc_rotation_matrix(element_coords: Array):
    delta_x = element_coords[1][0] - element_coords[0][0]
    delta_y = element_coords[1][1] - element_coords[0][1]
    length = (delta_x**2 + delta_y**2) ** 0.5
    return assembly_rotation_matrix(delta_x / length, delta_y / length)


def assembly_global_stiffness(
    element_lengths: Array,
    incidences: Array,
    nodal_dofs_mapping: Array,
    young_modulus: float,
    section_area: float,
    total_dofs: float,
):
    global_stiffness = np.zeros((total_dofs, total_dofs))
    for element_length, incidence in zip(element_lengths, incidences):
        element_stiffness = calc_element_stiffness(
            element_length=element_length,
            elastoplastic_modulus=young_modulus,
            section_area=section_area,
        )
        elem_eqs = nodal_dofs_mapping[:, incidence]
        test_m = np.reshape(np.arange(16, (4, 4)))
        z = np.zeros((4, 4))
        z[
            elem_eqs.T.flatten()[:, np.newaxis], elem_eqs.T.flatten()[np.newaxis, :]
        ] += test_m
        global_stiffness[
            elem_eqs.T.flatten()[:, np.newaxis], elem_eqs.T.flatten()[np.newaxis, :]
        ] += element_stiffness

    return global_stiffness
