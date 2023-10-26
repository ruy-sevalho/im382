def calc_lambda(poisson: float, young_modulus: float):
    return (poisson * young_modulus) / ((1 + poisson) * (1 - 2 * poisson))


def calc_mu(poisson: float, young_modulus: float):
    return young_modulus / (2 * (1 + poisson))
