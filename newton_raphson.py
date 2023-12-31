from dataclasses import dataclass, field
from typing import Protocol
from enum import Enum
from functools import cached_property, partial
from logging import warning
from typing import Callable
import numpy as np
import numpy.typing as npt
from bar_1d import EnergyNormsAndErrors

from bar_1d import BarInputNonLiner
from c0_basis import (
    calc_element_1D_jacobian,
    calc_p_knots_global,
    calc_ecsi_placement_coords_equal_dist,
    calc_load_vector,
    calc_approx_value,
)
from c0_basis import calc_incidence_matrix as calc_incidence_matrix_c0
from c1_basis import calc_incidence_matrix as calc_incidence_matrix_c1
from c1_basis import calc_p_knots_global as calc_knots_global_c1
from c1_basis import calc_p_knots_global_complete as calc_collocation_pts_c1
from nomeclature import (
    NUM_DISPLACEMENT,
)
from polynomials import d_lagrange_poli, get_points_weights, lagrange_poli, c1_basis


class ConvergenceCriteria(Enum):
    WORK = "work"
    DISPLACEMENT = "displacement"
    FORCE = "force"


@dataclass
class NewtonRaphsonResults:
    displacements: npt.NDArray[np.float64]
    crit_disp_list: npt.NDArray[np.float64]
    crit_residue_list: npt.NDArray[np.float64]
    crit_comb_list: npt.NDArray[np.float64]
    crit_disp_per_step: npt.NDArray[np.float64]
    crit_residue_per_step: npt.NDArray[np.float64]
    crit_comb_per_step: npt.NDArray[np.float64]


def newton_raphson(
    n_load_steps: int,
    max_iterations: int,
    convergence_criteria: ConvergenceCriteria,
    precision: float,
    young_modulus: float,
    poisson: float,
    area: float,
    n_elements: int,
    n_degrees_freedom: int,
    incidence_matrix: npt.NDArray[np.float64],
    collocation_pts: npt.NDArray[np.float64],
    int_weights: npt.NDArray[np.float64],
    load_vector: npt.NDArray[np.float64],
    b_ecsi: npt.NDArray[np.float64],
    det_j: float,
):
    """Newton Raphson Method for 1D bar"""

    free_nodes = np.arange(1, n_degrees_freedom)

    # Initialize convergences criteria
    crit_disp = 1
    crit_disp_list = np.array([])
    crit_residue = 1
    crit_residue_list = []
    crit_comb = 1
    crit_comb_list = []

    crit_disp_per_step = []
    crit_residue_per_step = []
    crit_comb_per_step = []

    conv_measure = 1

    disp = np.zeros(n_degrees_freedom)
    disp_increment = np.zeros(n_degrees_freedom)
    load_step_vector = load_vector / n_load_steps
    total_iter_count = 0
    iter_per_load_step = np.zeros(n_load_steps)

    for step in range(n_load_steps):
        load_step_counter = 0
        residue_init = (step + 1) * load_step_vector

        if step == 0:
            (
                tangent_stiffness_matrix,
                internal_load_vector,
            ) = assemble_global_non_linear_stiff_matrix(
                collocation_pts=collocation_pts,
                incidence_matrix=incidence_matrix,
                displacements=disp,
                b_ecsi=b_ecsi,
                int_weights=int_weights,
                young_modulus=young_modulus,
                poisson=poisson,
                area=area,
                det_j=det_j,
            )
        residue = residue_init - internal_load_vector

        # Newton-Raphson iterations
        while load_step_counter <= max_iterations and conv_measure > precision:
            load_step_counter += 1  # increment NR iteration counter.
            total_iter_count += 1  # increment total number of NR iterations.

            # Calculate increment in displacement
            disp_increment[free_nodes] = np.linalg.solve(
                tangent_stiffness_matrix[
                    free_nodes[:, np.newaxis], free_nodes[np.newaxis, :]
                ],
                residue[free_nodes],
            )

            # Increment the primary variable.
            disp[free_nodes] += disp_increment[free_nodes]

            # Compute the initial convergence criteria for the first NR iteration of each load step
            if load_step_counter == 1:
                init_comb_norm = np.sqrt(
                    np.abs(
                        np.dot(
                            residue[free_nodes],
                            disp_increment[free_nodes],
                        )
                    )
                )
                init_res_norm = np.linalg.norm(residue[free_nodes])

            # Update tangent stiffness matrix and internal and residue load
            (
                tangent_stiffness_matrix,
                internal_load_vector,
            ) = assemble_global_non_linear_stiff_matrix(
                collocation_pts=collocation_pts,
                incidence_matrix=incidence_matrix,
                displacements=disp,
                b_ecsi=b_ecsi,
                int_weights=int_weights,
                young_modulus=young_modulus,
                poisson=poisson,
                area=area,
                det_j=det_j,
            )
            residue = residue_init - internal_load_vector
            # Check convergence:
            # 1 - Criterion based on forces:
            #     a) Combined criterion (displacement increment and residue);
            #     b) Norm of residual.
            crit_comb = np.sqrt(
                np.abs(
                    np.dot(
                        residue[free_nodes],
                        disp_increment[free_nodes],
                    )
                )
            )
            crit_residue = np.linalg.norm(residue[free_nodes])

            # Calculate relative residues
            if init_comb_norm:
                crit_comb = crit_comb / init_comb_norm
            if init_res_norm:
                crit_residue = crit_residue / init_res_norm

            # 2- Norm of the primary variables.
            crit_disp = np.linalg.norm(disp_increment[free_nodes])
            disp_norm = np.linalg.norm(disp[free_nodes])
            if disp_norm:
                crit_disp = crit_disp / disp_norm

            table = {
                ConvergenceCriteria.WORK: crit_comb,
                ConvergenceCriteria.DISPLACEMENT: crit_disp,
                ConvergenceCriteria.FORCE: crit_residue,
            }
            conv_measure = table[convergence_criteria]

            crit_disp_list = np.append(crit_disp_list, crit_disp)
            crit_residue_list.append(crit_residue)
            crit_comb_list.append(crit_comb)
            assert len(crit_comb_list) == total_iter_count
        iter_per_load_step[step - 1] = load_step_counter

        crit_disp_per_step.append(crit_disp)
        crit_residue_per_step.append(crit_residue)
        crit_comb_per_step.append(crit_comb)
        if load_step_counter > max_iterations:
            warning(f"No conversion in load step {step}")
        else:
            conv_measure = 1

    return NewtonRaphsonResults(
        displacements=disp,
        crit_disp_list=crit_disp_list,
        crit_residue_list=np.array(crit_residue_list),
        crit_comb_list=np.array(crit_comb_list),
        crit_disp_per_step=np.array(crit_comb_per_step),
        crit_residue_per_step=np.array(crit_residue_per_step),
        crit_comb_per_step=np.array(crit_comb_per_step),
    )


def assemble_global_non_linear_stiff_matrix(
    collocation_pts: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    displacements: npt.NDArray[np.float64],
    b_ecsi: npt.NDArray[np.float64],
    int_weights: npt.NDArray[np.float64],
    young_modulus: float,
    poisson: float,
    area: float,
    det_j: float,
):
    n_elements = incidence_matrix.shape[0]
    n_knots = len(collocation_pts)
    n_int_pts = len(int_weights)
    element_degrees_freedom = incidence_matrix.shape[1]
    # Global tangent stiffness matrices
    global_tan_stiff_m = np.zeros((n_knots, n_knots))

    # Internal force vector
    internal_force_v = np.zeros(n_knots)

    # Lamé coefficients
    lambda_val = poisson * young_modulus / ((1 + poisson) * (1 - 2 * poisson))
    mu = young_modulus / (2 * (1 + poisson))

    for e in range(n_elements):
        # Initializes tangent stiffness and the internal force
        element_tangent_siff_m = np.zeros(
            (element_degrees_freedom, element_degrees_freedom)
        )
        element_internal_force = np.zeros(element_degrees_freedom)

        # Incidence, nodal coordinates and updated nodal coordinates
        elem_incidence = incidence_matrix[e, :]
        element_p_coords = collocation_pts[elem_incidence]
        element_x_coords = element_p_coords + displacements[elem_incidence]

        # Loop over integration points
        for i in range(n_int_pts):
            # Jacobian matrix of the initial and current configurations on
            # integration point i
            jacobian_p = np.dot(b_ecsi[:, i], element_p_coords)
            jacobian_x = np.dot(b_ecsi[:, i], element_x_coords)

            # Computes the tensor of deformation gradient on integration point i
            deformation_gradient = jacobian_x / jacobian_p
            cauchy_green_p_1 = deformation_gradient**2

            # Returns the constitutive matrix on integration point i
            constitutive_matrix_p = (
                2 * mu + lambda_val - 2 * lambda_val * np.log(np.sqrt(cauchy_green_p_1))
            ) / (cauchy_green_p_1**2)

            # Linear part of the displacement gradient tensor on
            # integration point i
            linear_part_disp_grad = b_ecsi[:, i] / jacobian_p

            # Non-linear part of the displacement gradient tensor
            # on integration point i
            non_linear_part_disp_grad = linear_part_disp_grad * deformation_gradient

            # Returns the Second Piola-Kirchhoff stress on integration point i
            piolla_kirc_stress_2 = (
                mu * (1 - 1 / cauchy_green_p_1)
                + lambda_val * np.log(np.sqrt(cauchy_green_p_1)) / cauchy_green_p_1
            )

            # Computes the tangent stiffness and the internal force vector on
            # integration point i
            # Tangent Matrix.
            element_tangent_siff_m += (
                (
                    np.outer(
                        linear_part_disp_grad,
                        piolla_kirc_stress_2 * linear_part_disp_grad,
                    )
                    + np.outer(
                        non_linear_part_disp_grad,
                        constitutive_matrix_p * non_linear_part_disp_grad,
                    )
                )
                * int_weights[i]
                * area
                * jacobian_p
            )

            # Internal force vector
            element_internal_force += (
                non_linear_part_disp_grad
                * piolla_kirc_stress_2
                * int_weights[i]
                * area
                * jacobian_p
            )

        # Assembles stiffness matrix, the internal force vector, and reaction force vector.

        global_tan_stiff_m[
            elem_incidence[:, np.newaxis], elem_incidence[np.newaxis, :]
        ] += element_tangent_siff_m
        internal_force_v[elem_incidence] += element_internal_force

    return global_tan_stiff_m, internal_force_v


@dataclass
class NewtonRaphsonConvergenceParam:
    n_load_steps: int = 10
    max_iterations: int = 100
    convergence_criteria: ConvergenceCriteria = ConvergenceCriteria.FORCE
    precision: float = 1e-7


@dataclass
class BarNewRaphsonPreProcessing:
    det_j: float
    n_degrees_freedom: int
    incidence_matrix: npt.NDArray[np.float64]
    collocation_pts: npt.NDArray[np.float64]
    x_knots_global: npt.NDArray[np.float64]
    int_weights: npt.NDArray[np.float64]
    ecsi_int_pts_coords: npt.NDArray[np.float64]
    load_vector: npt.NDArray[np.float64]
    n_ecsi: npt.NDArray[np.float64]
    b_ecsi: npt.NDArray[np.float64]
    n_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    b_ecsi_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def c0_pre_processing(
    length: float,
    degree: int,
    n_elements: int,
    load_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ecsi_placement_pts_function: Callable[[int], npt.NDArray[np.float64]],
    load_at_end: float,
):
    det_j = calc_element_1D_jacobian(length / n_elements)
    ecsi_placement_pts = ecsi_placement_pts_function(degree=degree)
    x_knots_global = calc_p_knots_global(
        length=length, n_elements=n_elements, esci_placement_coords=ecsi_placement_pts
    )
    n_degrees_freedom = x_knots_global.shape[0]
    integration_points, integration_weights = get_points_weights(
        intorder=2 * degree,
    )
    b_esci_matrix_function = partial(
        d_lagrange_poli,
        degree=degree,
        placement_pts_coords=ecsi_placement_pts,
    )
    b_esci_matrix_at_int_pts = b_esci_matrix_function(
        calc_pts_coords=integration_points,
    )
    n_ecsi_funtion = partial(
        lagrange_poli,
        degree=degree,
        placement_pts_coords=ecsi_placement_pts,
    )
    incidence_matrix = calc_incidence_matrix_c0(n_elements=n_elements, degree=degree)
    load_vector = calc_load_vector(
        x_knots=x_knots_global,
        incidence_matrix=incidence_matrix,
        test_function_local=n_ecsi_funtion,
        load_function=load_function,
        intorder=2 * degree + 2,
        det_j=det_j,
    )
    load_vector[n_elements] += load_at_end
    return BarNewRaphsonPreProcessing(
        det_j=det_j,
        n_degrees_freedom=n_degrees_freedom,
        incidence_matrix=incidence_matrix,
        collocation_pts=x_knots_global,
        x_knots_global=x_knots_global,
        int_weights=integration_weights,
        ecsi_int_pts_coords=integration_points,
        load_vector=load_vector,
        n_ecsi=n_ecsi_funtion(calc_pts_coords=integration_points),
        b_ecsi=b_esci_matrix_at_int_pts,
        n_ecsi_function=n_ecsi_funtion,
        b_ecsi_function=b_esci_matrix_function,
    )


def c1_pre_processing(
    length: float,
    degree: int,
    n_elements: int,
    load_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    load_at_end: float,
):
    element_size = length / n_elements
    det_j = calc_element_1D_jacobian(element_size)
    x_knots_global = calc_knots_global_c1(length=length, n_elements=n_elements)
    p_knots = calc_collocation_pts_c1(
        length=length, n_elements=n_elements, degree=degree
    )
    incidence_matrix = calc_incidence_matrix_c1(n_elements=n_elements, degree=degree)
    n_degrees_freedom = incidence_matrix[-1, -1] + 1
    integration_points, integration_weights = get_points_weights(
        intorder=2 * degree,
    )
    b_ecsi_function = partial(
        c1_basis, degree=degree, element_size=element_size, return_derivative_order=1
    )
    n_ecsi_function = partial(
        c1_basis, degree=degree, element_size=element_size, return_derivative_order=0
    )
    load_vector = calc_load_vector(
        x_knots=x_knots_global,
        incidence_matrix=incidence_matrix,
        test_function_local=n_ecsi_function,
        load_function=load_function,
        intorder=2 * degree + 2,
        det_j=det_j,
    )
    load_vector[2 * n_elements] += load_at_end
    b_esci_matrix_at_int_pts: npt.NDArray[np.float64] = b_ecsi_function(
        calc_pts_coords=integration_points,
    )
    return BarNewRaphsonPreProcessing(
        det_j=det_j,
        n_degrees_freedom=n_degrees_freedom,
        incidence_matrix=incidence_matrix,
        collocation_pts=p_knots,
        x_knots_global=x_knots_global,
        int_weights=integration_weights,
        ecsi_int_pts_coords=integration_points,
        load_vector=load_vector,
        n_ecsi=n_ecsi_function(calc_pts_coords=integration_points),
        b_ecsi=b_esci_matrix_at_int_pts,
        b_ecsi_function=b_ecsi_function,
        n_ecsi_function=n_ecsi_function,
    )


class BarAnalysisInput(Protocol):
    bar_input: BarInputNonLiner

    @property
    def pre_process(self) -> BarNewRaphsonPreProcessing:
        """Mesh pre processing"""
        ...


@dataclass
class BarAnalysis:
    inputs: BarAnalysisInput
    analytical_solution_function: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]
    analytical_derivative_solution_function: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]
    convergence_crit: NewtonRaphsonConvergenceParam = field(
        default_factory=NewtonRaphsonConvergenceParam
    )

    @cached_property
    def bar_input(self):
        return self.inputs.bar_input

    @cached_property
    def pre_process(self):
        return self.inputs.pre_process

    @cached_property
    def bar_result(self):
        return newton_raphson(
            n_load_steps=self.convergence_crit.n_load_steps,
            max_iterations=self.convergence_crit.max_iterations,
            convergence_criteria=self.convergence_crit.convergence_criteria,
            precision=self.convergence_crit.precision,
            young_modulus=self.bar_input.young_modulus,
            poisson=self.bar_input.poisson,
            area=self.bar_input.section_area,
            n_elements=self.bar_input.n_elements,
            n_degrees_freedom=self.pre_process.n_degrees_freedom,
            incidence_matrix=self.pre_process.incidence_matrix,
            collocation_pts=self.pre_process.collocation_pts,
            int_weights=self.pre_process.int_weights,
            load_vector=self.pre_process.load_vector,
            b_ecsi=self.pre_process.b_ecsi,
            det_j=self.pre_process.det_j,
        )

    def n_ecsi(self, ecsi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.pre_process.n_ecsi_function(calc_pts_coords=ecsi)

    def b_ecsi(self, ecsi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.pre_process.b_ecsi_function(calc_pts_coords=ecsi)

    def results(self, esci_calc_pts: npt.NDArray[np.float64] | None = None):
        if esci_calc_pts is None:
            esci_calc_pts = np.linspace(-1, 1, 21)
        return calc_approx_value(
            x_knots_global=self.pre_process.x_knots_global,
            element_incidence_matrix=self.pre_process.incidence_matrix,
            knot_displacements=self.bar_result.displacements,
            ecsi_matrix=self.n_ecsi(esci_calc_pts),
            ecsi_calc_pts=esci_calc_pts,
            factor=1,
            result_name=NUM_DISPLACEMENT,
        )

    @cached_property
    def error_norms(self):
        return calc_l2_h1_error_norms(
            analytical_solution_function=self.analytical_solution_function,
            analytical_derivative_solution_function=self.analytical_derivative_solution_function,
            p_init_global_coords=self.pre_process.collocation_pts,
            knot_displacements=self.bar_result.displacements,
            integration_weights=self.pre_process.int_weights,
            n_ecsi=self.pre_process.n_ecsi,
            b_ecsi=self.pre_process.b_ecsi,
            incidence_matrix=self.pre_process.incidence_matrix,
            det_j=self.pre_process.det_j,
        )

    @cached_property
    def h1_error(self):
        return self.error_norms.h1_error_norm

    @cached_property
    def l2_error(self):
        return self.error_norms.l2_error_norm


@dataclass
class C0BarAnalysisInput:
    bar_input: BarInputNonLiner
    ecsi_placement_pts_function: Callable[
        [int], npt.NDArray[np.float64]
    ] = calc_ecsi_placement_coords_equal_dist

    @cached_property
    def pre_process(self):
        return c0_pre_processing(
            length=self.bar_input.length,
            degree=self.bar_input.degree,
            n_elements=self.bar_input.n_elements,
            load_function=self.bar_input.load_function,
            ecsi_placement_pts_function=self.ecsi_placement_pts_function,
            load_at_end=self.bar_input.load_at_end,
        )


@dataclass
class C1BarAnalysisInput:
    bar_input: BarInputNonLiner

    @cached_property
    def pre_process(self):
        return c1_pre_processing(
            length=self.bar_input.length,
            degree=self.bar_input.degree,
            n_elements=self.bar_input.n_elements,
            load_function=self.bar_input.load_function,
            load_at_end=self.bar_input.load_at_end,
        )


def calc_l2_h1_error_norms(
    analytical_solution_function: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    analytical_derivative_solution_function: Callable[
        [npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    p_init_global_coords: npt.NDArray[np.float64],
    knot_displacements: npt.NDArray[np.float64],
    integration_weights: npt.NDArray[np.float64],
    n_ecsi: npt.NDArray[np.float64],
    b_ecsi: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    det_j: float,
):
    l2_error_norm = 0
    l2_sol_norm = 0
    h1_error_norm = 0
    h1_sol_norm = 0

    for e in incidence_matrix:
        element_displacements = knot_displacements[e]
        p_element = n_ecsi.T @ p_init_global_coords[e]
        analitycal_displacement = analytical_solution_function(p_element)
        analytical_derivative = analytical_derivative_solution_function(p_element)
        num_displacement = n_ecsi.T @ element_displacements
        num_derivative = b_ecsi.T @ element_displacements / det_j
        l2_error_norm += np.sum(
            (analitycal_displacement - num_displacement) ** 2
            * integration_weights
            * det_j
        )
        l2_sol_norm += np.sum(num_displacement**2 * det_j * integration_weights)
        h1_error_norm += np.sum(
            (analytical_derivative - num_derivative) ** 2 * det_j * integration_weights
        )
        h1_sol_norm += np.sum(num_derivative**2 * det_j * integration_weights)

    return EnergyNormsAndErrors(
        l2_error_norm=l2_error_norm**0.5,
        l2_sol_norm=l2_sol_norm**0.5,
        h1_error_norm=h1_error_norm**0.5,
        h1_sol_norm=h1_sol_norm**0.5,
    )
