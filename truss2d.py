from dataclasses import dataclass
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
    section_area: float
    poisson: float = 0.3
    isotropic_hardening: float = None
    yield_stress: float = None

    @cached_property
    def pre_process(self):
        return pre_process(
            coords=self.coords,
            incidences=self.incidences,
            boundary_conditions=self.boundary_conditions,
            loads=self.loads,
        )


@dataclass
class TrussPreProcess:
    n_elements: int
    n_free_dofs: int
    free_dofs_array: Array
    nodal_dofs_mapping: Array
    global_load: Array
    total_dofs: int


def pre_process(
    coords: Array,
    incidences: Array,
    boundary_conditions: Array,
    loads: Array,
):
    n_nodes = coords.shape[0]
    nodal_dofs = 2
    n_elements = incidences.shape[0]
    nodal_dofs_mapping = np.ones((nodal_dofs, n_nodes), dtype=int)
    for bc in boundary_conditions:
        nodal_dofs_mapping[bc[1], bc[0]] = 0
    n_free_dofs = np.sum(nodal_dofs_mapping.flatten())
    free_dofs_array = np.arange(n_free_dofs)
    total_dofs = n_nodes * nodal_dofs
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
    return TrussPreProcess(
        n_elements=n_elements,
        n_free_dofs=n_free_dofs,
        free_dofs_array=free_dofs_array,
        nodal_dofs_mapping=nodal_dofs_mapping,
        global_load=global_load,
        total_dofs=total_dofs,
    )


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


def assembly_rotation_matrix(cos: float, sin: float):
    return np.array([[cos, sin, 0, 0], [0, 0, cos, sin]])


def calc_rotation_matrix(element_coords: Array):
    delta_x = element_coords[1][0] - element_coords[0][0]
    delta_y = element_coords[1][1] - element_coords[0][1]
    length = (delta_x**2 + delta_y**2) ** 0.5
    return assembly_rotation_matrix(delta_x / length, delta_y / length)


def reduce_dimension(element_coords: Array):
    r = calc_rotation_matrix(element_coords=element_coords)
    return r @ element_coords.flatten()


def reduce_dimension_alt(element_coords: Array):
    delta_x = element_coords[1][0] - element_coords[0][0]
    delta_y = element_coords[1][1] - element_coords[0][1]
    length = (delta_x**2 + delta_y**2) ** 0.5
    return length


def calc_deformed_coords(
    initial_coords: Array,
    nodal_dofs_mapping: Array,
    displacements: Array,
    factor: float = 1,
):
    return initial_coords + displacements[nodal_dofs_mapping.T] * factor
