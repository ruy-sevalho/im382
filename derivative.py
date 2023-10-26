import sympy as sp

x, E, A = sp.symbols("x E A")
q = 1000 * sp.sin(sp.pi / 2 * x)
u_1 = sp.integrate(q, x)
u_2 = sp.integrate(u_1, x)
u_ = -u_2 / (E * A)
u = 4000 / (sp.pi**2 * E * A) * sp.sin(sp.pi / 2 * x)
sigma = sp.diff(u, x)
sigma2 = sp.diff(u_, x)
