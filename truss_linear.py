from dataclasses import asdict, dataclass
from functools import cached_property
from truss2d import TrussInputs, assembly_global_load, calc_element_lengths, pre_process
from type_alias import Array


import numpy as np


@dataclass
class LinearTruss:
    truss: TrussInputs

    @cached_property
    def run(self):
        return calc_truss(
            coords=self.truss.coords,
            incidences=self.truss.incidences,
            young_modulus=self.truss.young_modulus,
            section_area=self.truss.section_area,
            nodal_dofs_mapping=self.truss.pre_process.nodal_dofs_mapping,
            total_dofs=self.truss.pre_process.total_dofs,
            free_dofs_array=self.truss.pre_process.free_dofs_array,
            global_load=self.truss.pre_process.global_load,
            n_free_dofs=self.truss.pre_process.n_free_dofs,
        )


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
        global_stiffness[
            elem_eqs.T.flatten()[:, np.newaxis], elem_eqs.T.flatten()[np.newaxis, :]
        ] += element_stiffness

    return global_stiffness


def calc_truss(
    coords: Array,
    incidences: Array,
    young_modulus: float,
    section_area: float,
    nodal_dofs_mapping: Array,
    total_dofs: int,
    free_dofs_array: Array,
    global_load: Array,
    n_free_dofs: int,
):
    lengths = calc_element_lengths(coords=coords, incidences=incidences)
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
    displacements = np.linalg.solve(global_stiffness_free, global_load[:n_free_dofs])

    return displacements


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
