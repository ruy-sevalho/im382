import numpy as np
from scipy.integrate import quad


# Define the EnergyNormS2PKEGL function
def energy_norm_S2PKEGL(p, mu, lambda_, area):
    up = 0.2 * p + 0.1 * np.sin(p)
    dupdp = 0.2 + 0.1 * np.cos(p)
    Fp = 1 + dupdp
    dFp = dupdp
    Cp = Fp**2
    dCp = 2 * Fp * dFp
    S2PK = mu * (1 - 1 / Cp) + lambda_ * np.log(np.sqrt(Cp)) / Cp
    S1PK = S2PK * Fp

    EpsilonGL = 0.5 * (Cp - 1)
    dEpsilonGL = 0.5 * dCp

    integrand = S2PK * dEpsilonGL * area
    return integrand


# Perform numerical integration
def calc_energy_norm(mu: float, lambda_: float, area: float, length: float):
    result, _ = quad(lambda p: energy_norm_S2PKEGL(p, mu, lambda_, area), 0, length)

    energy_norm = np.sqrt(result)
    return energy_norm
