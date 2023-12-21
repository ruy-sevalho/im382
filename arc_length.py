# %%
import numpy as np
import matplotlib.pyplot as plt


c0 = 0
c1 = 1
c2 = -1
c3 = 0.3


# %%
def F_internal(x):
    # Internal nodal force vector
    return c3 * (x**3) + c2 * (x**2) + c1 * x + c0


def k_stiff(x):
    # Derivation of function y=-x.^2+x
    # Normally, the global stiffness matrix comes from the FEM model
    return 3 * c3 * (x**2) + 2 * c2 * x + c1


def lambda_solv(a1, a2, a3):
    # Solve the quadratic equation: a1 x**2 + a2 x + a3
    if a1 == 0 and a2 == 0:
        print("No roots found")
        return None
    elif a1 == 0 and a2 != 0:
        delta_lambda = -a3 / a2
    else:
        fac = a2**2 - 4 * a1 * a3
        if fac < 0:
            print("No root found, returning")
            return None
        lambda1 = (-a2 + np.sqrt(fac)) / (2 * a1)
        lambda2 = (-a2 - np.sqrt(fac)) / (2 * a1)
        delta_lambda = max(lambda1, lambda2)

    return delta_lambda


# %%
# Clear console, command window, and close all plots


# 1D Arch-length method with Modified Newton Raphson iteration

# y = -x.^2 + x
# y' = 1 - 2x

# Plot the analytical solution
x = np.arange(0, 2.001, 0.001)
y = F_internal(x)

NR = 1  # NR=1, Newton Raphson; NR=2, Modified newton Raphson

psi = 1.0
deltalL = 0.05  # arch-length

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
(h1,) = ax.plot(x, y, "b")
ax.grid(True)
# ax.set_ylim([0, 0.28])
# %%


# Arch length
# psi = 1.0
# deltalL = 0.65316

# Newton Raphson or Modified Newton Raphson
NR = 1

max_iter = 100
tol = 1e-6


max_step = 19
intendted_iterations = 5

q_ref = 0.4

# %%
lambda_val = np.array([0.0])
p = np.array([0.0])
step = 0

lambda_val_step = np.array([0.0])
p_step = np.array([0.0])
deltalL_all = np.array([0.0])


while lambda_val[-1] < 0.99 and step < max_step:
    iter = 0
    lambda_val_step = np.array(
        [
            lambda_val[-1],
        ]
    )
    p_step = np.array(
        [
            p[-1],
        ]
    )
    # y_equation = (
    #     lambda x: np.sqrt(-psi * (deltalL**2 - p[-1] ** 2 + 2 * p[-1] * x - x**2))
    #     / psi
    #     + lambda_val[-1]
    # )
    # # x += x + p[-1]
    # (h2,) = ax.plot(
    #     x,
    #     y_equation(x),
    #     "r--",
    #     label=f"$(x-{p[-1]})^2 + {psi} * (y-{lambda_val[-1]})^2 = {deltalL**2}$",
    # )
    Residual = 1
    delta_lambda_ini = deltalL / np.sqrt(1 + psi**2)
    while iter <= max_iter + 1 and Residual >= tol:
        if iter == 0:
            delta_lambda = np.array([delta_lambda_ini])
            # Update load factor and displacement
            lambda_val_step = np.append(
                lambda_val_step, lambda_val_step[iter] + delta_lambda[iter]
            )
            F_ini = F_internal(p_step[iter])
            F_ext = lambda_val_step[iter + 1] * q_ref
            Residual = F_ext - F_ini
            k = k_stiff(p_step[iter])
            delta_p = np.array([0.0, Residual / k])
            p_step = np.append(p_step, p_step[iter] + delta_p[iter + 1])
            delta_lambda = np.append(delta_lambda, delta_lambda[iter])
        else:
            if NR == 1:
                k = k_stiff(p_step[iter])
            elif NR == 2:
                k = k_stiff(p_step[iter])

            F_ini = F_internal(p_step[iter])
            F_ext = lambda_val_step[iter] * q_ref
            Residual = F_ext - F_ini

            delta_p_bar = Residual / k
            delta_p_t = q_ref / k
            a1 = delta_p_t**2 + psi * q_ref**2
            # a2 = (
            #     2 * np.dot(delta_p_t, delta_p + delta_p_bar)
            #     + 2 * delta_lambda[iter] * psi**2 * q_ref**2
            # )
            a2 = (
                2 * delta_p_t * (delta_p[iter] + delta_p_bar)
                + 2 * delta_lambda[iter] * psi**2 * q_ref**2
            )
            a3 = (
                (delta_p[iter] + delta_p_bar) ** 2
                - deltalL**2
                + (delta_lambda[iter] ** 2) * psi**2 * q_ref**2
            )

            ddelta_lambda = lambda_solv(a1, a2, a3)

            delta_lambda = np.append(delta_lambda, delta_lambda[iter] + ddelta_lambda)
            delta_p = np.append(
                delta_p, delta_p[iter] + delta_p_bar + ddelta_lambda * delta_p_t
            )

            lambda_val_step = np.append(
                lambda_val_step, lambda_val_step[0] + delta_lambda[iter + 1]
            )
            p_step = np.append(p_step, p_step[0] + delta_p[iter + 1])

        print("iter =", iter)

        # if iter >= 2:
        #     ax.plot([p_step[iter], p_step[iter]], [F_ini, F_ext], color="r")
        #     ax.plot(p_step[iter], F_ini, "ro", markersize=5, markerfacecolor="r")
        #     ax.plot(
        #         [p_step[iter], p_step[iter + 1]],
        #         [F_ini, lambda_val_step[iter + 1] * q_ref],
        #         linestyle="-",
        #         color="r",
        #     )

        # plt.pause(0.2)

        iter += 1

    # Plot results
    (h3,) = ax.plot(
        p_step, lambda_val_step * q_ref, "r-o", markersize=5, markerfacecolor="r"
    )
    lambda_val = np.append(lambda_val, lambda_val_step[-1])
    p = np.append(p, p_step[-1])
    deltalL = deltalL * (intendted_iterations / iter) ** 0.5
    deltalL_all = np.append(deltalL_all, deltalL)
    step += 1
    print(f"step {step}")

ax.set_title(f"Iteration = {iter} Residual = {Residual * 9999}")
ax.legend([h1, h3], ["Analytical solution", "Iteration"])
ax.set_xlabel("Displacement")
ax.set_ylabel("Force λ*R")

# %%
