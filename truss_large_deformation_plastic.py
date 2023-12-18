from dataclasses import asdict, dataclass, field
from functools import cached_property
from logging import warning
import numpy as np
from c0_basis import calc_ecsi_placement_coords_gauss_lobato
from newton_raphson import ConvergenceCriteria, NewtonRaphsonConvergenceParam
from polynomials import d_lagrange_poli, get_points_weights
from truss2d import (
    TrussInputs,
    assembly_global_load,
    calc_rotation_matrix,
)
from type_alias import Array


@dataclass
class Analysis:
    truss: TrussInputs
    convergence_crit: NewtonRaphsonConvergenceParam = field(
        default_factory=NewtonRaphsonConvergenceParam
    )

    @cached_property
    def results(self):
        return newton_raphson_platic(
            **asdict(self.truss),
            **asdict(self.convergence_crit),
            n_elements=self.truss.pre_process.n_elements,
            nodal_dofs_mapping=self.truss.pre_process.nodal_dofs_mapping,
            total_dofs=self.truss.pre_process.total_dofs,
            free_dofs_array=self.truss.pre_process.free_dofs_array,
            global_load=self.truss.pre_process.global_load,
            n_free_dofs=self.truss.pre_process.n_free_dofs,
        )

    def deformed_shape(self, scale: float = 1):
        return calc_deformed_coords(
            initial_coords=self.truss.coords,
            nodal_dofs_mapping=self.results.nodal_dofs_mapping,
            displacements=self.results.displacements,
            factor=scale,
        )


@dataclass
class ResultsNewtonRaphson:
    displacements: Array
    plastic_strains: Array
    alphas: Array
    crit_disp_list: Array
    crit_residue_list: Array
    crit_comb_list: Array
    crit_disp_per_step: Array
    crit_residue_per_step: Array
    crit_comb_per_step: Array


def newton_raphson_platic(
    section_area: float,
    young_modulus: float,
    isotropic_hardening: float,
    yield_stress: float,
    coords: Array,
    incidences: Array,
    boundary_conditions: Array,
    loads: Array,
    n_load_steps: int,
    n_elements: int,
    nodal_dofs_mapping: Array,
    total_dofs: int,
    free_dofs_array: Array,
    global_load: Array,
    n_free_dofs: int,
    max_iterations: int,
    convergence_criteria: ConvergenceCriteria,
    precision: float,
    print: bool = False,
):
    degree = 1

    # Element approximation
    integration_points, integration_weights = get_points_weights(
        intorder=2 * degree,
    )
    n_integration_pts = integration_points.shape[0]
    placement_pts_coords = calc_ecsi_placement_coords_gauss_lobato(degree=degree)
    b_ecsi = d_lagrange_poli(
        calc_pts_coords=integration_points,
        placement_pts_coords=placement_pts_coords,
        degree=degree,
    )

    load_step_vector = global_load / n_load_steps

    # Convergences criteria
    crit_disp = 1
    crit_disp_list = []
    crit_residue = 1
    crit_residue_list = []
    crit_comb = 1
    crit_comb_list = []

    crit_disp_per_step = []
    crit_residue_per_step = []
    crit_comb_per_step = []

    conv_measure = 1

    # Plasticity model
    plastic_strains = np.zeros((n_elements, n_integration_pts))
    alphas = np.zeros((n_elements, n_integration_pts))

    # Solution
    displacements = np.zeros(total_dofs)
    displacements_increment = np.zeros(total_dofs)
    stresses = np.zeros((n_load_steps, n_elements, integration_points.shape[0]))

    # Newton Raphson
    total_iter_count = 0
    iter_per_load_step = np.zeros(n_load_steps)
    for step in range(n_load_steps):
        load_step_counter = 0
        residue_init = (step + 1) * load_step_vector

        if step == 0:
            (
                tangent_stiffness_matrix,
                internal_load_vector,
                plastic_strains,
                alphas,
                stresses[step],
            ) = assemble_stiff_matrix_and_internal_force_vector(
                b_ecsi=b_ecsi,
                integration_weights=integration_weights,
                element_incidences=incidences,
                displacements=displacements,
                element_coords=coords,
                nodal_dofs_mapping=nodal_dofs_mapping,
                total_dofs=total_dofs,
                young_modulus=young_modulus,
                isotropic_hardening=isotropic_hardening,
                yield_stress=yield_stress,
                plastic_strains=plastic_strains,
                alphas=alphas,
                section_area=section_area,
            )
        residue = residue_init - internal_load_vector

        # Newton-Raphson iterations
        while load_step_counter <= max_iterations and conv_measure > precision:
            load_step_counter += 1  # increment NR iteration counter.
            total_iter_count += 1  # increment total number of NR iterations.

            tangent_stiffness_matrix_free_nodes = tangent_stiffness_matrix[
                free_dofs_array[:, np.newaxis], free_dofs_array[np.newaxis, :]
            ]
            displacements_increment[free_dofs_array] = np.linalg.solve(
                tangent_stiffness_matrix_free_nodes, residue[free_dofs_array]
            )
            displacements[free_dofs_array] += displacements_increment[free_dofs_array]

            if load_step_counter == 1:
                init_comb_norm = np.sqrt(
                    np.abs(
                        np.dot(
                            residue[free_dofs_array],
                            displacements_increment[free_dofs_array],
                        )
                    )
                )
                init_res_norm = np.linalg.norm(residue[free_dofs_array])

            (
                tangent_stiffness_matrix,
                internal_load_vector,
                plastic_strains,
                alphas,
                stresses[step],
            ) = assemble_stiff_matrix_and_internal_force_vector(
                b_ecsi=b_ecsi,
                integration_weights=integration_weights,
                element_incidences=incidences,
                displacements=displacements,
                element_coords=coords,
                nodal_dofs_mapping=nodal_dofs_mapping,
                total_dofs=total_dofs,
                young_modulus=young_modulus,
                isotropic_hardening=isotropic_hardening,
                yield_stress=yield_stress,
                plastic_strains=plastic_strains,
                alphas=alphas,
                section_area=section_area,
            )
            residue = residue_init - internal_load_vector

            crit_comb = np.sqrt(
                np.abs(
                    np.dot(
                        residue[free_dofs_array],
                        displacements_increment[free_dofs_array],
                    )
                )
            )
            crit_residue = np.linalg.norm(residue[free_dofs_array])

            if init_comb_norm:
                crit_comb = crit_comb / init_comb_norm
            if init_res_norm:
                crit_residue = crit_residue / init_res_norm

            crit_disp = np.linalg.norm(displacements_increment[free_dofs_array])
            disp_norm = np.linalg.norm(displacements[free_dofs_array])
            if disp_norm:
                crit_disp = crit_disp / disp_norm

            table = {
                ConvergenceCriteria.WORK: crit_comb,
                ConvergenceCriteria.DISPLACEMENT: crit_disp,
                ConvergenceCriteria.FORCE: crit_residue,
            }
            conv_measure = table[convergence_criteria]
            crit_disp_list.append(crit_disp)
            crit_residue_list.append(crit_residue)
            crit_comb_list.append(crit_comb)
            if print:
                print(f"Load step: {step}, interation: {load_step_counter}")

        iter_per_load_step[step - 1] = load_step_counter
        crit_disp_per_step.append(crit_disp)
        crit_residue_per_step.append(crit_residue)
        crit_comb_per_step.append(crit_comb)
        if load_step_counter > max_iterations:
            warning(f"No conversion in load step {step}")
        else:
            conv_measure = 1

    return ResultsNewtonRaphson(
        displacements=displacements,
        plastic_strains=plastic_strains,
        alphas=alphas,
        crit_disp_list=np.array(crit_disp_list),
        crit_residue_list=np.array(crit_residue_list),
        crit_comb_list=np.array(crit_comb_list),
        crit_disp_per_step=np.array(crit_comb_per_step),
        crit_residue_per_step=np.array(crit_residue_per_step),
        crit_comb_per_step=np.array(crit_comb_per_step),
    )


def assemble_stiff_matrix_and_internal_force_vector(
    b_ecsi: Array,
    integration_weights: Array,
    element_incidences: Array,
    displacements: Array,
    element_coords: Array,
    nodal_dofs_mapping: Array,
    total_dofs: float,
    young_modulus: float,
    isotropic_hardening: float,
    yield_stress: float,
    plastic_strains: Array,
    alphas: Array,
    section_area: float,
):
    degree = 1
    stresses = np.zeros((element_incidences.shape[0], integration_weights.shape[0]))
    global_tangent_stiffness = np.zeros((total_dofs, total_dofs))
    global_internal_force = np.zeros(total_dofs)
    for e, (element_incidence, plastic_strain, alpha) in enumerate(
        zip(element_incidences, plastic_strains, alphas)
    ):
        element_stiffness = np.zeros((degree + 1, degree + 1))
        element_internal_force = np.zeros(degree + 1)
        p_2d_coords = element_coords[element_incidence]
        element_mapped_index = nodal_dofs_mapping[:, element_incidence]
        element_displacements = displacements[element_mapped_index.T]
        x_2d_coords = p_2d_coords + element_displacements
        # p_1d_coords = reduce_dimension(p_2d_coords)
        # x_1d_coords = reduce_dimension(x_2d_coords)
        for i, (integration_weight, b_ecsi_col) in enumerate(
            zip(integration_weights, b_ecsi.T)
        ):
            # jacobian_p = b_ecsi_col @ p_1d_coords
            # assert jacobian_p == abs(p_1d_coords[1] - p_1d_coords[0]) / 2
            # dif_p = (jacobian_p - reduce_dimension_alt(p_2d_coords) / 2) / jacobian_p
            # assert dif_p < 1e-10
            # jacobian_x = b_ecsi_col @ x_1d_coords
            # assert jacobian_x == abs(x_1d_coords[1] - x_1d_coords[0]) / 2
            # dif_x = (jacobian_x - reduce_dimension_alt(x_2d_coords) / 2) / jacobian_x
            # assert dif_x < 1e-10

            jacobian_p = reduce_dimension_alt(p_2d_coords) / 2
            jacobian_x = reduce_dimension_alt(x_2d_coords) / 2
            deformation_gradient = jacobian_x / jacobian_p
            linear_part_disp_grad = b_ecsi_col / jacobian_p
            strain_increment = np.log(deformation_gradient)

            (
                stresses[e, i],
                plastic_strain[i],
                alpha[i],
                elastoplastic_modulus,
            ) = return_mapping_isotropic_hardening(
                young_modulus=young_modulus,
                isotropic_hardening=isotropic_hardening,
                yield_stress=yield_stress,
                strain_increment=strain_increment,
                plastic_strain=plastic_strain[i],
                alpha=alpha[i],
            )

            non_linear_part_disp_grad = linear_part_disp_grad * deformation_gradient
            piolla_kirc_stress = stresses[e, i] / deformation_gradient
            element_stiffness += (
                (
                    np.outer(
                        linear_part_disp_grad,
                        piolla_kirc_stress * linear_part_disp_grad,
                    )
                    + np.outer(
                        non_linear_part_disp_grad,
                        elastoplastic_modulus * non_linear_part_disp_grad,
                    )
                )
                * integration_weight
                * section_area
                * jacobian_p
            )

            element_internal_force += (
                non_linear_part_disp_grad
                * piolla_kirc_stress
                * integration_weight
                * section_area
                * jacobian_p
            )
        rotation_matrix = calc_rotation_matrix(x_2d_coords)
        corrected_element_stiffness = (
            rotation_matrix.T @ element_stiffness @ rotation_matrix
        )
        corrected_internal_force = rotation_matrix.T @ element_internal_force
        global_tangent_stiffness[
            element_mapped_index.T.flatten()[:, np.newaxis],
            element_mapped_index.T.flatten()[np.newaxis, :],
        ] += corrected_element_stiffness
        global_internal_force[
            element_mapped_index.T.flatten()
        ] += corrected_internal_force

    return (
        global_tangent_stiffness,
        global_internal_force,
        plastic_strains,
        alphas,
        stresses,
    )


def reduce_dimension(element_coords: Array):
    r = calc_rotation_matrix(element_coords=element_coords)
    return r @ element_coords.flatten()


def reduce_dimension_alt(element_coords: Array):
    delta_x = element_coords[1][0] - element_coords[0][0]
    delta_y = element_coords[1][1] - element_coords[0][1]
    length = (delta_x**2 + delta_y**2) ** 0.5
    return length


def return_mapping_isotropic_hardening(
    young_modulus: float,
    isotropic_hardening: float,
    yield_stress: float,
    strain_increment: float,
    plastic_strain: float,
    alpha: float,
):
    trial_stress = young_modulus * (strain_increment - plastic_strain)
    trial_f = abs(trial_stress) - (yield_stress + isotropic_hardening * alpha)
    if trial_f <= 0:  # Elastic
        new_stress = trial_stress
        new_plastic_strain = plastic_strain
        new_alpha = alpha
        elastoplastic_modulus = young_modulus
    else:
        corrected_modulus = young_modulus + isotropic_hardening
        delta_lambda = trial_f / corrected_modulus
        new_stress = (
            1 - delta_lambda * young_modulus / abs(trial_stress)
        ) * trial_stress
        new_plastic_strain = plastic_strain + delta_lambda * np.sign(trial_stress)
        new_alpha = alpha + delta_lambda
        elastoplastic_modulus = young_modulus * isotropic_hardening / corrected_modulus
    return new_stress, new_plastic_strain, new_alpha, elastoplastic_modulus


def calc_deformed_coords(
    initial_coords: Array,
    nodal_dofs_mapping: Array,
    displacements: Array,
    factor: float = 1,
):
    return initial_coords + displacements[nodal_dofs_mapping.T] * factor
