from dataclasses import dataclass
from time import time
from functools import cached_property, partial
from typing import Callable, Protocol
import numpy as np
import numpy.typing as npt
from bar_1d import BarInputDynamics

from c0_basis import calc_p_knots_global
from c0_basis import calc_incidence_matrix
from lame import calc_lambda, calc_mu
from newton_raphson import calc_l2_h1_error_norms
from polynomials import d_lagrange_poli, get_points_weights, lagrange_poli
from pre_process import (
    calc_element_1D_jacobian,
    calc_element_matrix,
    calc_external_load_vector,
    compose_global_matrix,
)


def pe(x: float):
    print(f"{x:e}")


@dataclass
class DynamicsResults:
    displacements: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]
    accelerations: npt.NDArray[np.float64]
    elapsed_time: float


def run_analysis(
    young_modulus: float,
    poisson: float,
    area: float,
    p_coords: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    integration_weights: npt.NDArray[np.float64],
    det_j: float,
    n_ecsi: npt.NDArray[np.float64],
    b_ecsi: npt.NDArray[np.float64],
    global_mass_matrix: npt.NDArray[np.float64],
    dist_load_function_p_t: Callable[[float, float], float],
    normal_force_pk_1_at_end_function_t: Callable[[float], float],
    disp_function_p_t: Callable[[float, float], float],
    velocity_function_p_t: Callable[[float, float], float],
    t_initial: float,
    t_final: float,
    n_time_steps: int,
):
    n_elements = incidence_matrix.shape[0]
    degrees_of_freedom = incidence_matrix[-1, -1] + 1

    times = np.linspace(t_initial, t_final, n_time_steps + 1)
    delta_t = (t_final - t_initial) / n_time_steps
    a0 = 1 / delta_t**2
    a1 = 1 / (2 * delta_t)
    a2 = 2 * a0
    a3 = 1 / a2

    initial_displacements = disp_function_p_t(p_coords, t_initial)
    displacements = np.zeros((n_time_steps + 2, degrees_of_freedom))
    velocities = np.zeros((n_time_steps + 2, degrees_of_freedom))
    accelerations = np.zeros((n_time_steps + 2, degrees_of_freedom))

    lumped_mass_vector = np.array(
        [np.sum(mass) for mass in global_mass_matrix.T]
    )  # check with mathlab
    internal_load_vector = calc_internal_load_vector(
        p_coords=p_coords,
        incidence_matrix=incidence_matrix,
        displacements=initial_displacements,
        b_ecsi=b_ecsi,
        det_j=det_j,
        integration_weights=integration_weights,
        young_modulus=young_modulus,
        poisson=poisson,
        area=area,
    )
    external_load_vector = calc_external_load_vector(
        p_coords=p_coords,
        incidence_matrix=incidence_matrix,
        n_ecsi=n_ecsi,
        integration_weights=integration_weights,
        det_j=det_j,
        load_function=partial(dist_load_function_p_t, t=t_initial),
    )
    external_load_vector[n_elements] += normal_force_pk_1_at_end_function_t(t_initial)
    free_nodes = np.arange(1, degrees_of_freedom)
    velocities[1, :] = velocity_function_p_t(p_coords, t_initial)
    displacements[1, :] = initial_displacements
    accelerations[1, :] = (
        external_load_vector - internal_load_vector
    ) / lumped_mass_vector
    displacements[0, :] = (
        displacements[1, :] - delta_t * velocities[1, :] + a3 * accelerations[1, :]
    )
    time_start = time()
    hat_mass_vector = lumped_mass_vector * a0
    for step in range(1, n_time_steps):
        t = times[step]
        external_load_vector = calc_external_load_vector(
            p_coords=p_coords,
            incidence_matrix=incidence_matrix,
            n_ecsi=n_ecsi,
            integration_weights=integration_weights,
            det_j=det_j,
            load_function=partial(dist_load_function_p_t, t=t),
        )
        external_load_vector[n_elements] += normal_force_pk_1_at_end_function_t(t)
        residual_load_vector = external_load_vector - internal_load_vector
        displacements[step + 1, free_nodes] = (
            residual_load_vector[free_nodes] / hat_mass_vector[free_nodes]
            + 2 * displacements[step, free_nodes]
            - displacements[step - 1, free_nodes]
        )
        accelerations[step + 1, :] = a0 * (
            displacements[step - 1, :]
            - 2 * displacements[step, :]
            + displacements[step + 1, :]
        )
        velocities[step + 1, :] = a1 * (
            displacements[step + 1, :] - displacements[step - 1, :]
        )
        internal_load_vector = calc_internal_load_vector(
            p_coords=p_coords,
            incidence_matrix=incidence_matrix,
            displacements=displacements[step + 1, :],
            b_ecsi=b_ecsi,
            det_j=det_j,
            integration_weights=integration_weights,
            young_modulus=young_modulus,
            poisson=poisson,
            area=area,
        )

    displacements[-1, :] = displacements[-2, :]
    external_load_vector = calc_external_load_vector(
        p_coords=p_coords,
        incidence_matrix=incidence_matrix,
        n_ecsi=n_ecsi,
        integration_weights=integration_weights,
        det_j=det_j,
        load_function=partial(dist_load_function_p_t, t=t_final),
    )
    accelerations[-1, :] = (
        external_load_vector - internal_load_vector
    ) / lumped_mass_vector
    velocities[-1, :] = velocities[-2, :] + 1 / a1 * accelerations[-1, :]
    elapsed_time = time() - time_start
    return DynamicsResults(
        displacements=displacements,
        velocities=velocities,
        accelerations=accelerations,
        elapsed_time=elapsed_time,
    )


def calc_internal_load_vector(
    p_coords: npt.NDArray[np.float64],
    incidence_matrix: npt.NDArray[np.float64],
    displacements: npt.NDArray[np.float64],
    b_ecsi: npt.NDArray[np.float64],
    det_j: float,
    integration_weights: npt.NDArray[np.float64],
    young_modulus: float,
    poisson: float,
    area: float,
):
    lambda_ = calc_lambda(poisson=poisson, young_modulus=young_modulus)
    mu = calc_mu(poisson=poisson, young_modulus=young_modulus)
    internal_loads = np.zeros(incidence_matrix[-1, -1] + 1)
    for e in incidence_matrix:
        p_coord = p_coords[e]
        x_coord = p_coord + displacements[e]
        internal_load = np.zeros(incidence_matrix.shape[1])
        for col, weight in zip(b_ecsi.T, integration_weights):
            jacob_x = col @ x_coord
            grad_p = jacob_x / det_j
            cauchy_green = grad_p**2
            non_linear_grad_u = col * grad_p / det_j
            stress_piolla_kirchoff_2 = (
                mu * (1 - 1 / cauchy_green)
                + lambda_ * np.log(np.sqrt(cauchy_green)) / cauchy_green
            )
            internal_load += (
                non_linear_grad_u * stress_piolla_kirchoff_2 * weight * area * det_j
            )
        internal_loads[e] += internal_load
    return internal_loads


@dataclass
class DynamicsPreProcess:
    det_j: float
    p_coords: npt.NDArray[np.float64]
    incidence_matrix: npt.NDArray[np.float64]
    integration_weigths: npt.NDArray[np.float64]
    b_ecsi: npt.NDArray[np.float64]
    n_ecsi: npt.NDArray[np.float64]
    global_mass_matrix: npt.NDArray[np.float64]


def pre_process_c0(
    length: float,
    section_area: float,
    young_modulus: float,
    poisson: float,
    density: float,
    n_elements: float,
    degree: int,
    ecsi_placement_coords_function: Callable[[int], npt.NDArray[np.float64]],
):
    det_j = calc_element_1D_jacobian(element_size=length / n_elements)
    ecsi_pts = ecsi_placement_coords_function(degree)
    p_knots_global = calc_p_knots_global(
        length=length, n_elements=n_elements, esci_placement_coords=ecsi_pts
    )
    incidence_matrix = calc_incidence_matrix(n_elements=n_elements, degree=degree)
    integration_pts, integration_weights = get_points_weights(intorder=2 * degree)
    n_ecsi = lagrange_poli(
        calc_pts_coords=integration_pts, degree=degree, placement_pts_coords=ecsi_pts
    )
    b_ecsi = d_lagrange_poli(
        calc_pts_coords=integration_pts, degree=degree, placement_pts_coords=ecsi_pts
    )
    element_mass_matrix = calc_element_matrix(
        factor=det_j * density * section_area,
        ecsi_matrix=n_ecsi,
        integration_weights=integration_weights,
    )
    global_mass_matrix = compose_global_matrix(
        element_matrix=element_mass_matrix, incidence_matrix=incidence_matrix
    )
    return DynamicsPreProcess(
        det_j=det_j,
        p_coords=p_knots_global,
        incidence_matrix=incidence_matrix,
        integration_weigths=integration_weights,
        b_ecsi=b_ecsi,
        n_ecsi=n_ecsi,
        global_mass_matrix=global_mass_matrix,
    )


class BarDynamicsDefinition(Protocol):
    bar: BarInputDynamics

    @property
    def pre_process(self) -> DynamicsPreProcess:
        ...


@dataclass
class C0BarDynamics:
    bar: BarInputDynamics
    ecsi_placement_coords_function: Callable[[int], npt.NDArray[np.float64]]

    @cached_property
    def pre_process(self):
        return pre_process_c0(
            length=self.bar.length,
            section_area=self.bar.section_area,
            young_modulus=self.bar.young_modulus,
            poisson=self.bar.poisson,
            density=self.bar.density,
            n_elements=self.bar.n_elements,
            degree=self.bar.degree,
            ecsi_placement_coords_function=self.ecsi_placement_coords_function,
        )


@dataclass
class BarDynamics:
    data: BarDynamicsDefinition
    dist_load_function: Callable[[np.float64, np.float64], np.float64]
    normal_force_pk_1_at_end_function_t: Callable[[np.float64], np.float64]
    disp_function_p_t: Callable[[float, float], float]
    velocity_function_p_t: Callable[[float, float], float]
    t_initial: float
    t_final: float
    n_time_steps: int

    @cached_property
    def bar(self) -> BarInputDynamics:
        return self.data.bar

    @cached_property
    def pre_process(self):
        return self.data.pre_process

    @cached_property
    def results(self):
        return run_analysis(
            young_modulus=self.bar.young_modulus,
            poisson=self.bar.poisson,
            area=self.bar.section_area,
            p_coords=self.pre_process.p_coords,
            incidence_matrix=self.pre_process.incidence_matrix,
            integration_weights=self.pre_process.integration_weigths,
            det_j=self.pre_process.det_j,
            n_ecsi=self.pre_process.n_ecsi,
            b_ecsi=self.pre_process.b_ecsi,
            global_mass_matrix=self.pre_process.global_mass_matrix,
            dist_load_function_p_t=self.dist_load_function,
            normal_force_pk_1_at_end_function_t=self.normal_force_pk_1_at_end_function_t,
            disp_function_p_t=self.disp_function_p_t,
            velocity_function_p_t=self.velocity_function_p_t,
            t_initial=self.t_initial,
            t_final=self.t_final,
            n_time_steps=self.n_time_steps,
        )

    @cached_property
    def error_norms(self):
        return calc_l2_h1_error_norms(
            analytical_solution_function=partial(
                self.disp_function_p_t, t=self.t_final
            ),
            analytical_derivative_solution_function=partial(
                self.velocity_function_p_t, t=self.t_final
            ),
            p_init_global_coords=self.pre_process.p_coords,
            knot_displacements=self.results.displacements[-1, :],
            intergration_weights=self.pre_process.integration_weigths,
            n_ecsi=self.pre_process.n_ecsi,
            b_ecsi=self.pre_process.b_ecsi,
            incidence_matrix=self.pre_process.incidence_matrix,
            det_j=self.pre_process.det_j,
        )
