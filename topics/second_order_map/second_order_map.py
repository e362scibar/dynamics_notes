# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dynamics_notes
#     language: python
#     name: python3
# ---

# %% [markdown]
# - File: `topics/second_order_map/second_order_map.py`
# - Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# - SPDX-License-Identifier: GPL-3.0-or-later
# - Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

# %% [markdown]
# # Second-order transfer maps of a drift space, a sector dipole magnet, and a quadrupole magnet
#
# Memo 2026-004
#
# This memo derives the second-order transfer map ($T_{ijk}$) of the three basic elements of
# [2026-002] from the equations of motion (EOM) expanded to second order in [2026-003]. The linear part is the
# transfer matrix of [2026-002].
#
# Scope:
# - Static magnetic fields only, $k_s = 0$, constant $h$ and $k$ in each element, hard-edge model, and a sector dipole
#   magnet with perpendicular end faces (no edge focusing).
# - Phase-space vector $\mathbf{u} = (x, p_x, y, p_y, z, \delta)^\mathsf{T}$ with the conventions of [2026-001]. The
#   momenta are the canonical ones, normalized by $P_0$. They are not the slopes $x'$ and $y'$.
# - The quadrupole curvature corrections neglected in [2026-001] are not included.
#
# Every result is verified with sympy and numerically against the exact EOM.

# %%
import numpy as np
import sympy as sp
from IPython.display import display
from scipy.integrate import solve_ivp

L, s = sp.symbols("L s", positive=True)  # element length and the running variable of the integrals
h = sp.Symbol("h", positive=True)
g = sp.Symbol("g", positive=True)  # g = sqrt(|k|) of a quadrupole
gamma0 = sp.Symbol("gamma_0", positive=True)
k_sym = sp.Symbol("k", real=True)
x_, px, y_, py, z_s, d = sp.symbols("x p_x y p_y z delta", real=True)
state = [x_, px, y_, py, z_s, d]
state_names = ["x", "p_x", "y", "p_y", "z", r"\delta"]
x0, px0, y0, py0, z0, d0 = sp.symbols("x_0 p_{x0} y_0 p_{y0} z_0 delta_0", real=True)
u0 = sp.Matrix([x0, px0, y0, py0, z0, d0])  # entrance vector

# Numerical constants of the checks.
BETA0_NUM = 0.9
GAMMA0_NUM = float(1.0 / np.sqrt(1.0 - BETA0_NUM**2))
AMPLITUDE = 2.0e-3
ENTRANCE_DIRECTION = np.array([1.0, -0.7, 0.5, 0.3, 0.2, 0.8])  # generic direction of the entrance vector
ODE_RTOL = 1.0e-12
ODE_ATOL = 1.0e-16
RATIO_TOLERANCE = 0.15  # relative tolerance on the expected error ratios 4 and 8
RESIDUAL_TOLERANCE = 1.0e-10
N_RANDOM_POINTS = 5
RNG_SEED = 20261121
RANDOM_LOW, RANDOM_HIGH = 0.2, 1.5  # range of the random positive parameters of the algebraic checks
RANDOM_U_SCALE = 1.0  # scale of the random entrance vector in the algebraic checks

# %% [markdown]
# ## 2. Second-order solution of the equations of motion
#
# [2026-003] gives the EOM through second order, $\mathbf{u}' = \mathbf{A}\mathbf{u} + \mathbf{f}_2(\mathbf{u})$, where
# $\mathbf{A}$ is the matrix of [2026-002, Sec. 2] and $\mathbf{f}_2$ is the quadratic part obtained from $H_3$:
#
# $$
# H_3 = (hx - \delta)\,\frac{p_x^2 + p_y^2}{2} + (hx - \delta)\,\frac{\delta^2}{2\gamma_0^2},
# \qquad \mathbf{f}_2 = \Big(\frac{\partial H_3}{\partial p_x},\ -\frac{\partial H_3}{\partial x},\
# \frac{\partial H_3}{\partial p_y},\ -\frac{\partial H_3}{\partial y},\ \frac{\partial H_3}{\partial \delta},\
# -\frac{\partial H_3}{\partial z}\Big).
# $$
#
# Write the solution as $\mathbf{u}(L) = \mathbf{M}(L)\,\mathbf{u}_0 + \mathbf{w}(L)$, where $\mathbf{M}$ is the linear
# transfer matrix and $\mathbf{w}$ is of second order in $\mathbf{u}_0$. Inserting this into the EOM and keeping the
# second-order terms (the argument of $\mathbf{f}_2$ is the first-order solution $\mathbf{M}\mathbf{u}_0$) gives
#
# $$
# \mathbf{w}' = \mathbf{A}\,\mathbf{w} + \mathbf{f}_2\big(\mathbf{M}(L)\,\mathbf{u}_0\big), \qquad \mathbf{w}(0) = 0 .
# $$
#
# The solution follows from the variation of constants (Duhamel's formula), using $\mathbf{M}' = \mathbf{A}\mathbf{M}$:
#
# $$
# \mathbf{w}(L) = \int_0^L \mathbf{M}(L - s)\;\mathbf{f}_2\big(\mathbf{M}(s)\,\mathbf{u}_0\big)\,ds .
# $$
#
# Each component of $\mathbf{w}$ is a quadratic form in $\mathbf{u}_0$. The second-order transfer map is defined by
#
# $$
# u_i(L) = \sum_{j} R_{ij}\,u_{0j} + \sum_{j \le k} T_{ijk}\,u_{0j}\,u_{0k}, \qquad R_{ij} = M_{ij},
# $$
#
# so that $T_{ijk}$ is the coefficient of the monomial $u_{0j}u_{0k}$ in $w_i$ (the convention of TRANSPORT, with
# each pair $j \le k$ counted once).

# %%
H3 = (h * x_ - d) * (px**2 + py**2) / 2 + (h * x_ - d) * d**2 / (2 * gamma0**2)
f2_state = sp.Matrix([sp.diff(H3, px), -sp.diff(H3, x_), sp.diff(H3, py), -sp.diff(H3, y_),
                      sp.diff(H3, d), -sp.diff(H3, z_s)])
display(sp.Eq(sp.Symbol("f_2"), f2_state))

M_sym, f2_sym = sp.Symbol("M"), sp.Symbol("f_2")
w_sym = sp.Symbol("w")
display(sp.Eq(sp.Symbol("w'"), sp.Symbol("A") * w_sym + sp.Symbol(r"f_2(M u_0)")))
display(sp.Eq(sp.Symbol("w(L)"), sp.Integral(sp.Symbol("M(L-s)") * sp.Symbol(r"f_2(M(s) u_0)"), (s, 0, L))))


def second_order_part(M_of, h_value, length=L) -> sp.Matrix:
    """Second-order part w(L) from Duhamel's formula for the linear matrix function M_of(l) and the value of h."""
    f2_h = f2_state.subs(h, h_value)
    u_s = M_of(s) * u0
    f2_s = f2_h.subs(dict(zip(state, u_s)), simultaneous=True)
    integrand = M_of(length - s) * f2_s
    w = sp.zeros(6, 1)
    for i in range(6):
        expanded = sp.expand(integrand[i].rewrite(sp.exp))
        poly = sp.Poly(expanded, *u0)
        for monomial, coeff in poly.terms():
            integral = sp.integrate(sp.expand(coeff), (s, 0, length))
            integral = sp.simplify(sp.expand(integral).rewrite(sp.cos))
            w[i] += integral * sp.prod([u**p for u, p in zip(u0, monomial)])
    return w


def tensor_elements(w: sp.Matrix) -> dict:
    """Coefficients T_ijk (1-based indices, j <= k) of the quadratic forms w_i."""
    elements = {}
    for i in range(6):
        poly = sp.Poly(sp.expand(w[i]), *u0)
        for monomial, coeff in poly.terms():
            if coeff == 0:
                continue
            indices = [j for j, power in enumerate(monomial) for _ in range(power)]
            j, k = indices
            elements[(i + 1, j + 1, k + 1)] = sp.simplify(coeff)
    return elements


# %% [markdown]
# Linear transfer matrices of [2026-002] (closed forms, rebuilt here as functions of the running length).

# %%
def M_drift(l):
    """Drift space, [2026-002, Sec. 4]."""
    return sp.Matrix([[1, l, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, l, 0, 0], [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, l / gamma0**2], [0, 0, 0, 0, 0, 1]])


def M_bend(l):
    """Sector dipole magnet (k = 0), [2026-002, Sec. 5]."""
    th, rho = h * l, 1 / h
    return sp.Matrix([
        [sp.cos(th), rho * sp.sin(th), 0, 0, 0, rho * (1 - sp.cos(th))],
        [-sp.sin(th) / rho, sp.cos(th), 0, 0, 0, sp.sin(th)],
        [0, 0, 1, l, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [-sp.sin(th), -rho * (1 - sp.cos(th)), 0, 0, 1, l / gamma0**2 - rho * (th - sp.sin(th))],
        [0, 0, 0, 0, 0, 1],
    ])


def M_quad_focusing(l):
    """Quadrupole with k = g^2 > 0, [2026-002, Sec. 6]."""
    x_block = sp.Matrix([[sp.cos(g * l), sp.sin(g * l) / g], [-g * sp.sin(g * l), sp.cos(g * l)]])
    y_block = sp.Matrix([[sp.cosh(g * l), sp.sinh(g * l) / g], [g * sp.sinh(g * l), sp.cosh(g * l)]])
    return sp.diag(x_block, y_block, sp.Matrix([[1, l / gamma0**2], [0, 1]]))


def M_quad_defocusing(l):
    """Quadrupole with k = -g^2 < 0: the planes are exchanged."""
    x_block = sp.Matrix([[sp.cosh(g * l), sp.sinh(g * l) / g], [g * sp.sinh(g * l), sp.cosh(g * l)]])
    y_block = sp.Matrix([[sp.cos(g * l), sp.sin(g * l) / g], [-g * sp.sin(g * l), sp.cos(g * l)]])
    return sp.diag(x_block, y_block, sp.Matrix([[1, l / gamma0**2], [0, 1]]))


def A_matrix(h_value, k_value) -> sp.Matrix:
    """Matrix A of [2026-002, Sec. 2]."""
    return sp.Matrix([
        [0, 1, 0, 0, 0, 0],
        [-(h_value**2 + k_value), 0, 0, 0, 0, h_value],
        [0, 0, 0, 1, 0, 0],
        [0, 0, k_value, 0, 0, 0],
        [-h_value, 0, 0, 0, 0, 1 / gamma0**2],
        [0, 0, 0, 0, 0, 0],
    ])


ELEMENTS = {
    # name: (M(l), h, k)
    "drift": (M_drift, 0, 0),
    "bend": (M_bend, h, 0),
    "quad k>0": (M_quad_focusing, 0, g**2),
    "quad k<0": (M_quad_defocusing, 0, -(g**2)),
}

# %% [markdown]
# ## 3. Drift space
#
# With $h = 0$ the only second-order terms of $\mathbf{f}_2$ are $x' \ni -\delta p_x$, $y' \ni -\delta p_y$ and
# $z' \ni -(p_x^2 + p_y^2)/2 - 3\delta^2/(2\gamma_0^2)$, and the first-order solution is $\mathbf{M}(s)\mathbf{u}_0$
# with constant $p_x$, $p_y$, $\delta$. Integrating over $L$:
#
# $$
# \begin{aligned}
# x(L) &= x_0 + L\,p_{x0} - L\,\delta_0\,p_{x0}, \qquad y(L) = y_0 + L\,p_{y0} - L\,\delta_0\,p_{y0}, \\
# z(L) &= z_0 + \frac{L}{\gamma_0^2}\,\delta_0 - \frac{L}{2}\,(p_{x0}^2 + p_{y0}^2) - \frac{3L}{2\gamma_0^2}\,\delta_0^2 .
# \end{aligned}
# $$
#
# The second-order terms $-L\delta_0 p_{x0}$ and $-L\delta_0 p_{y0}$ are the chromatic change of the slope
# ($x' = p_x/(1 + \delta)$ to first order in $\delta$), and $-L(p_{x0}^2 + p_{y0}^2)/2$ is the path-length increase of
# a slanted trajectory.

# %%
w_drift = second_order_part(M_drift, 0)
w_drift_expected = sp.Matrix([
    -L * d0 * px0, 0, -L * d0 * py0, 0,
    -L * (px0**2 + py0**2) / 2 - 3 * L * d0**2 / (2 * gamma0**2), 0,
])
display(sp.Eq(sp.Symbol("w_{drift}"), w_drift_expected))
assert sp.simplify(w_drift - w_drift_expected) == sp.zeros(6, 1)

# %% [markdown]
# ## 4. Sector dipole magnet
#
# With $h \neq 0$ and $k = 0$, let $\theta = hL$. The integrals of Duhamel's formula contain only products of
# $\sin$ and $\cos$ of $hs$, and they give
#
# $$
# \begin{aligned}
# w_1 &= -\frac{h\sin^2\theta}{2}\,x_0^2 + \sin\theta\cos\theta\,x_0 p_{x0} + \sin^2\theta\,x_0\delta_0
#   - \frac{\sin\theta\cos\theta}{h}\,p_{x0}\delta_0 + \frac{\cos\theta\,(1-\cos\theta)}{2h}\,p_{x0}^2 \\
#   &\quad - \frac{1-\cos\theta}{2h}\,p_{y0}^2
#   - \Big[\frac{\sin^2\theta}{2h} + \frac{1-\cos\theta}{2\gamma_0^2 h}\Big]\delta_0^2 , \\
# w_2 &= -\frac{\sin\theta}{2}\Big(p_{x0}^2 + p_{y0}^2 + \frac{\delta_0^2}{\gamma_0^2}\Big), \\
# w_3 &= \sin\theta\,x_0 p_{y0} + \frac{1-\cos\theta}{h}\,p_{x0}p_{y0} - \frac{\sin\theta}{h}\,p_{y0}\delta_0 , \\
# w_4 &= 0, \\
# w_5 &= -\frac{\sin\theta}{2h}\,(p_{x0}^2 + p_{y0}^2) + \frac{\sin\theta}{\gamma_0^2}\,x_0\delta_0
#   + \frac{1-\cos\theta}{\gamma_0^2 h}\,p_{x0}\delta_0 - \frac{3\sin\theta}{2\gamma_0^2 h}\,\delta_0^2 , \\
# w_6 &= 0 .
# \end{aligned}
# $$

# %%
w_bend = second_order_part(M_bend, h)
th = h * L
w_bend_expected = sp.Matrix([
    (-h * sp.sin(th)**2 / 2 * x0**2 + sp.sin(th) * sp.cos(th) * x0 * px0 + sp.sin(th)**2 * x0 * d0
     - sp.sin(th) * sp.cos(th) / h * px0 * d0 + sp.cos(th) * (1 - sp.cos(th)) / (2 * h) * px0**2
     - (1 - sp.cos(th)) / (2 * h) * py0**2
     - (sp.sin(th)**2 / (2 * h) + (1 - sp.cos(th)) / (2 * gamma0**2 * h)) * d0**2),
    -sp.sin(th) / 2 * (px0**2 + py0**2 + d0**2 / gamma0**2),
    sp.sin(th) * x0 * py0 + (1 - sp.cos(th)) / h * px0 * py0 - sp.sin(th) / h * py0 * d0,
    0,
    (-sp.sin(th) / (2 * h) * (px0**2 + py0**2) + sp.sin(th) / gamma0**2 * x0 * d0
     + (1 - sp.cos(th)) / (gamma0**2 * h) * px0 * d0 - 3 * sp.sin(th) / (2 * gamma0**2 * h) * d0**2),
    0,
])
display(sp.Eq(sp.Symbol("w_{bend}"), w_bend_expected))

# %% [markdown]
# ## 5. Quadrupole magnet
#
# With $h = 0$ and $k = g^2 > 0$, let $\varphi = gL$. Only the terms $x' \ni -\delta p_x$, $y' \ni -\delta p_y$ and the
# $z'$ terms of the drift space remain in $\mathbf{f}_2$, so the second-order terms are the effect of these terms on
# the first-order oscillation:
#
# $$
# \begin{aligned}
# w_1 &= \frac{\delta_0}{2g}\Big[g^2 L\sin\varphi\;x_0 - (\varphi\cos\varphi + \sin\varphi)\,p_{x0}\Big], \\
# w_2 &= \frac{g\,\delta_0}{2}\Big[(\varphi\cos\varphi - \sin\varphi)\,x_0 + L\sin\varphi\;p_{x0}\Big], \\
# w_3 &= -\frac{\delta_0}{2g}\Big[g^2 L\sinh\varphi\;y_0 + (\varphi\cosh\varphi + \sinh\varphi)\,p_{y0}\Big], \\
# w_4 &= -\frac{g\,\delta_0}{2}\Big[(\varphi\cosh\varphi - \sinh\varphi)\,y_0 + L\sinh\varphi\;p_{y0}\Big], \\
# w_5 &= -\frac{3L}{2\gamma_0^2}\,\delta_0^2
#   - \frac{1}{8g}\Big[g^2(2\varphi - \sin 2\varphi)\,x_0^2 - 4g\sin^2\varphi\;x_0 p_{x0}
#   + (2\varphi + \sin 2\varphi)\,p_{x0}^2 \\
#   &\qquad - g^2(2\varphi - \sinh 2\varphi)\,y_0^2 + 4g\sinh^2\varphi\;y_0 p_{y0}
#   + (2\varphi + \sinh 2\varphi)\,p_{y0}^2\Big], \\
# w_6 &= 0 .
# \end{aligned}
# $$
#
# For $k = -g^2 < 0$ the equations of the two planes are exchanged, so
# $\mathbf{w}_{k<0}(\mathbf{u}_0) = \mathbf{P}\,\mathbf{w}_{k>0}(\mathbf{P}\,\mathbf{u}_0)$ with the permutation
# $\mathbf{P}$ that exchanges $(x, p_x)$ and $(y, p_y)$.

# %%
w_quad_f = second_order_part(M_quad_focusing, 0)
phi = g * L
w_quad_expected = sp.Matrix([
    d0 / (2 * g) * (g**2 * L * sp.sin(phi) * x0 - (phi * sp.cos(phi) + sp.sin(phi)) * px0),
    g * d0 / 2 * ((phi * sp.cos(phi) - sp.sin(phi)) * x0 + L * sp.sin(phi) * px0),
    -d0 / (2 * g) * (g**2 * L * sp.sinh(phi) * y0 + (phi * sp.cosh(phi) + sp.sinh(phi)) * py0),
    -g * d0 / 2 * ((phi * sp.cosh(phi) - sp.sinh(phi)) * y0 + L * sp.sinh(phi) * py0),
    (-3 * L / (2 * gamma0**2) * d0**2
     - 1 / (8 * g) * (g**2 * (2 * phi - sp.sin(2 * phi)) * x0**2 - 4 * g * sp.sin(phi)**2 * x0 * px0
                      + (2 * phi + sp.sin(2 * phi)) * px0**2
                      - g**2 * (2 * phi - sp.sinh(2 * phi)) * y0**2 + 4 * g * sp.sinh(phi)**2 * y0 * py0
                      + (2 * phi + sp.sinh(2 * phi)) * py0**2)),
    0,
])
display(sp.Eq(sp.Symbol(r"w_{quad}(k=g^2)"), w_quad_expected))

# Plane exchange for k < 0.
P_swap = sp.Matrix([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
w_quad_d_expected = P_swap * w_quad_expected.subs(dict(zip(u0, P_swap * u0)), simultaneous=True)
w_quad_d = second_order_part(M_quad_defocusing, 0)
display(sp.Eq(sp.Symbol(r"w_{quad}(k=-g^2)"), w_quad_d_expected))

# %% [markdown]
# The nonzero elements $T_{ijk}$ follow from the quadratic forms $w_i$. The second-order terms of the bend and of
# the quadrupole are listed by index below, and the closed forms are those of $\mathbf{w}$ above.

# %%
w_results = {"drift": w_drift, "bend": w_bend, "quad k>0": w_quad_f, "quad k<0": w_quad_d}
w_expected = {"drift": w_drift_expected, "bend": w_bend_expected, "quad k>0": w_quad_expected,
              "quad k<0": w_quad_d_expected}
for name, w in w_results.items():
    elements = tensor_elements(w)
    print(f"{name:9s}: {len(elements)} nonzero T_ijk (i, j, k) =", sorted(elements))


# %% [markdown]
# ## 6. Checks
#
# 1. Duhamel's integrals agree with the closed forms typed above (random numerical points, and symbolically for the
#    drift space).
# 2. $\mathbf{w}' = \mathbf{A}\mathbf{w} + \mathbf{f}_2(\mathbf{M}\mathbf{u}_0)$ and $\mathbf{w}(0) = 0$.
# 3. Second-order symplectic condition. The map $\mathbf{P}(\mathbf{u}_0) = \mathbf{M}\mathbf{u}_0 + \mathbf{w}(\mathbf{u}_0)$
#    is symplectic through first order in $\mathbf{u}_0$ if $\mathbf{M}^\mathsf{T}\mathbf{J}\mathbf{W} +
#    \mathbf{W}^\mathsf{T}\mathbf{J}\mathbf{M} = 0$ with $\mathbf{W} = \partial\mathbf{w}/\partial\mathbf{u}_0$.
# 3b. Defect of the truncated map. Since $\mathbf{M}$ is symplectic, the first non-vanishing term of
#    $\mathbf{D}^\mathsf{T}\mathbf{J}\mathbf{D} - \mathbf{J}$ is $\mathbf{W}^\mathsf{T}\mathbf{J}\mathbf{W}$, which is
#    quadratic in $\mathbf{u}_0$. It vanishes identically for the drift space and is nonzero for the bend and the
#    quadrupole.
# 3c. Determinant. $\det\mathbf{D} - 1 = 0$ for the drift space and $p_{x0}\,p_x(L)\sin^2\theta$ for the bend.
# 3d. Quadrupole closed form. The defect is $\delta_0^2$ times a combination of $A = (Lg)^2 - \sin^2(Lg)$ and
#    $B = (Lg)^2 - \sinh^2(Lg)$.
# 3e. Thin-lens limit of the quadrupole defect.
# 3f. Generating function. The type-2 integrability condition holds for the drift space and fails for the bend and the
#    quadrupole.
# 3g. Third-order compensation: the defect equals the contribution of the third-order terms of the exact map.
# 4. Composition: the map of a length $L_1 + L_2$ equals the map of $L_1$ followed by the map of $L_2$, through second
#    order: $\mathbf{w}(L_1 + L_2) = \mathbf{M}(L_2)\,\mathbf{w}(L_1) + \mathbf{w}_{L_2}(\mathbf{M}(L_1)\,\mathbf{u}_0)$.
# 5. Limits: the bend tends to the drift space for $h \to 0$ and the quadrupole for $g \to 0$.
# 6. Numerical comparison with the exact EOM: the error of the linear map scales as $a^2$, and that of the second-order
#    map as $a^3$.

# %%
rng = np.random.default_rng(RNG_SEED)


U0_SET = set(u0)


def random_point(symbols: list) -> dict:
    """Random positive values for the parameters and random real values for the entrance vector."""
    values = {}
    for sym in symbols:
        if sym in U0_SET:
            values[sym] = RANDOM_U_SCALE * rng.uniform(-1.0, 1.0)
        else:
            values[sym] = rng.uniform(RANDOM_LOW, RANDOM_HIGH)
    return values


def max_residual(expr: sp.Matrix) -> float:
    """Largest absolute value of a matrix expression over random points."""
    symbols = sorted(expr.free_symbols, key=str)
    func = sp.lambdify(symbols, expr, "numpy")
    worst = 0.0
    for _ in range(N_RANDOM_POINTS):
        point = random_point(symbols)
        value = np.array(func(*[point[sym] for sym in symbols]), dtype=float)
        worst = max(worst, float(np.max(np.abs(value))))
    return worst


# Check 1.
for name in w_results:
    residual = max_residual(w_results[name] - w_expected[name])
    print(f"{name:9s} Duhamel - closed form: {residual:.1e}")
    assert residual < RESIDUAL_TOLERANCE

# Check 2.
for name, (M_of, h_value, k_value) in ELEMENTS.items():
    w = w_expected[name]
    f2_el = f2_state.subs(h, h_value).subs(dict(zip(state, M_of(L) * u0)), simultaneous=True)
    ode_residual = max_residual(sp.diff(w, L) - A_matrix(h_value, k_value) * w - f2_el)
    initial_residual = max_residual(w.subs(L, 0))
    print(f"{name:9s} ODE residual: {ode_residual:.1e}, w(0): {initial_residual:.1e}")
    assert ode_residual < RESIDUAL_TOLERANCE and initial_residual < RESIDUAL_TOLERANCE

# %%
# Check 3: second-order symplectic condition.
J2 = sp.Matrix([[0, 1], [-1, 0]])
J_sym = sp.diag(J2, J2, J2)
J_sym_np = np.array(J_sym.tolist(), dtype=float)
for name, (M_of, _h, _k) in ELEMENTS.items():
    W = w_expected[name].jacobian(u0)
    condition = M_of(L).T * J_sym * W + W.T * J_sym * M_of(L)
    residual = max_residual(condition)
    print(f"{name:9s} M^T J W + W^T J M: {residual:.1e}")
    assert residual < RESIDUAL_TOLERANCE

# %%
# Check 3b: defect of the truncated map. The O(u0) group vanishes, so the first non-vanishing term is W^T J W.
defect = {}
for name, (M_of, _h, _k) in ELEMENTS.items():
    W = w_expected[name].jacobian(u0)
    defect[name] = sp.expand(W.T * J_sym * W)
    residual = max_residual(defect[name])
    print(f"{name:9s} W^T J W: {residual:.1e}")
assert sp.expand(defect["drift"]) == sp.zeros(6, 6)
for name in ("bend", "quad k>0", "quad k<0"):
    assert max_residual(defect[name]) > 1.0e-6

# The defect is quadratic in u0: scaling u0 by epsilon scales the defect by epsilon^2.
eps_scale = sp.Symbol("epsilon", positive=True)
for name in defect:
    scaled = defect[name].subs({v: eps_scale * v for v in u0}, simultaneous=True)
    assert sp.simplify(scaled - eps_scale**2 * defect[name]) == sp.zeros(6, 6)
print("the defect is quadratic in u0 for every element")

# %%
# Check 3c: determinant of the truncated map.
for name, (M_of, _h, _k) in ELEMENTS.items():
    D_map = (M_of(L) * u0 + w_expected[name]).jacobian(u0)
    det_defect = sp.simplify(sp.expand(D_map.det() - 1))
    print(f"{name:9s} det(D) - 1: {max_residual(sp.Matrix([det_defect])):.1e}")
    if name == "drift":
        assert sp.simplify(det_defect) == 0
    elif name == "bend":
        theta = h * L
        p_x_exit = d0 * sp.sin(theta) - h * x0 * sp.sin(theta) + px0 * sp.cos(theta)
        assert sp.simplify(det_defect + px0 * p_x_exit * sp.sin(theta) ** 2) == 0
        display(sp.Eq(sp.Symbol(r"\det\mathbf{D} - 1"), -px0 * p_x_exit * sp.sin(theta) ** 2))

# %%
# Check 3d: closed form of the quadrupole defect with A = (Lg)^2 - sin^2(Lg) and B = (Lg)^2 - sinh^2(Lg).
varphi = g * L
A_def = varphi**2 - sp.sin(varphi) ** 2
B_def = varphi**2 - sp.sinh(varphi) ** 2
quad_entries = {
    (0, 1): d0**2 * A_def / 4, (0, 5): d0 * px0 * A_def / 4, (1, 5): -d0 * x0 * A_def / 4,
    (2, 3): -d0**2 * B_def / 4, (2, 5): -d0 * py0 * B_def / 4, (3, 5): d0 * y0 * B_def / 4,
}
quad_closed = sp.zeros(6, 6)
for (i, j), value in quad_entries.items():
    quad_closed[i, j] = value
    quad_closed[j, i] = -value
assert sp.simplify(defect["quad k>0"] - quad_closed) == sp.zeros(6, 6)
display(sp.Eq(sp.Symbol("A"), A_def))
display(sp.Eq(sp.Symbol("B"), B_def))
print("quadrupole defect matches the A, B closed form")

# %%
# Check 3e: the quadrupole defect vanishes in the thin-lens limit L -> 0 with kL = q fixed.
q_thin = sp.Symbol("q", positive=True)
thin_defect = defect["quad k>0"].subs(g, sp.sqrt(q_thin / L))
for i in range(6):
    for j in range(6):
        entry = sp.simplify(thin_defect[i, j])
        if entry != 0:
            assert sp.simplify(sp.limit(entry, L, 0, "+")) == 0
print("the quadrupole defect vanishes in the thin-lens limit")

# %%
# Check 3f: type-2 generating function. w_q = dG/dp0 and w_p = -dG/dq0 require
# d w_q[i]/d q0[j] + d w_p[j]/d p0[i] = 0 for all i, j.
q0 = [x0, y0, z0]
p0 = [px0, py0, d0]
integrability = {}
for name in ELEMENTS:
    w = w_expected[name]
    w_q = [w[0], w[2], w[4]]
    w_p = [w[1], w[3], w[5]]
    violations = 0
    for i in range(3):
        for j in range(3):
            entry = sp.simplify(sp.diff(w_q[i], q0[j]) + sp.diff(w_p[j], p0[i]))
            if entry != 0:
                violations += 1
    integrability[name] = violations
    print(f"{name:9s} integrability violations: {violations}")
assert integrability["drift"] == 0
assert integrability["bend"] > 0
assert integrability["quad k>0"] > 0
print("only the drift space has a type-2 generating function")

# %%
# Check 3g: the third-order terms of the exact map compensate the defect. Numerically, with the exact map of
# [2026-001, Sec. 10] and finite differences.
R_exact_check = sp.sqrt(1 + 2 * d + (1 - 1 / gamma0**2) * d**2 - px**2 - py**2)
H_exact_check = -(1 + h * x_) * R_exact_check + d + h * (x_ + h * x_**2 / 2) + k_sym * (x_**2 - y_**2) / 2
rhs_check = [sp.diff(H_exact_check, px), -sp.diff(H_exact_check, x_), sp.diff(H_exact_check, py),
             -sp.diff(H_exact_check, y_), sp.diff(H_exact_check, d), -sp.diff(H_exact_check, z_s)]
rhs_check_func = sp.lambdify([*state, h, k_sym, gamma0], rhs_check, "numpy")


def exact_jacobian(u_start: np.ndarray, h_num: float, k_num: float, length: float, step: float) -> np.ndarray:
    """Jacobian of the exact map by central differences."""
    jac = np.zeros((6, 6))
    for i in range(6):
        delta_vec = np.zeros(6)
        delta_vec[i] = step
        plus = solve_ivp(lambda _s, u: rhs_check_func(*u, h_num, k_num, GAMMA0_NUM), (0.0, length),
                         u_start + delta_vec, rtol=ODE_RTOL, atol=ODE_ATOL, method="DOP853").y[:, -1]
        minus = solve_ivp(lambda _s, u: rhs_check_func(*u, h_num, k_num, GAMMA0_NUM), (0.0, length),
                          u_start - delta_vec, rtol=ODE_RTOL, atol=ODE_ATOL, method="DOP853").y[:, -1]
        jac[:, i] = (plus - minus) / (2.0 * step)
    return jac


def quadratic_jacobian(w: sp.Matrix, u_start: np.ndarray, params: dict, length: float, step: float) -> np.ndarray:
    """Jacobian of the quadratic part w by central differences."""
    w_func = sp.lambdify([*u0], w.subs({**params, gamma0: GAMMA0_NUM, L: length}), "numpy")
    jac = np.zeros((6, 6))
    for i in range(6):
        delta_vec = np.zeros(6)
        delta_vec[i] = step
        plus = np.array(w_func(*(u_start + delta_vec)), dtype=float).reshape(6)
        minus = np.array(w_func(*(u_start - delta_vec)), dtype=float).reshape(6)
        jac[:, i] = (plus - minus) / (2.0 * step)
    return jac


COMPENSATION_CASES = {"bend": (0.2, 0.0, 1.5), "quad k>0": (0.0, 2.0, 0.4)}
COMPENSATION_AMPLITUDE = 1.0e-2
COMPENSATION_STEP = 1.0e-7
for case, (h_num, k_num, length) in COMPENSATION_CASES.items():
    params = {h: h_num, g: float(np.sqrt(abs(k_num)))}
    u_start = COMPENSATION_AMPLITUDE * ENTRANCE_DIRECTION
    M_num = np.array(ELEMENTS[case][0](L).subs({**params, gamma0: GAMMA0_NUM, L: length}).evalf(), dtype=float)
    W_num = quadratic_jacobian(w_expected[case], u_start, params, length, COMPENSATION_STEP)
    D_exact = exact_jacobian(u_start, h_num, k_num, length, COMPENSATION_STEP)
    T_num = D_exact - M_num - W_num
    lhs = W_num.T @ J_sym_np @ W_num
    rhs = -(M_num.T @ J_sym_np @ T_num + T_num.T @ J_sym_np @ M_num)
    residual = np.max(np.abs(lhs - rhs))
    print(f"{case:9s} |W^T J W| = {np.max(np.abs(lhs)):.2e}, "
          f"|defect + third-order contribution| = {residual:.2e}")
    assert residual < 0.1 * np.max(np.abs(lhs))

# Check 4: composition of two sections of the same element.
L1, L2 = sp.symbols("L_1 L_2", positive=True)
for name, (M_of, _h, _k) in ELEMENTS.items():
    w = w_expected[name]
    lhs = w.subs(L, L1 + L2)
    w_first = w.subs(L, L1)
    u_mid = M_of(L1) * u0
    w_second = w.subs(L, L2).subs(dict(zip(u0, u_mid)), simultaneous=True)
    residual = max_residual(lhs - (M_of(L2) * w_first + w_second))
    print(f"{name:9s} composition: {residual:.1e}")
    assert residual < RESIDUAL_TOLERANCE

# %%
# Check 5: limits.
w_bend_h0 = w_bend_expected.applyfunc(lambda e: sp.limit(e, h, 0))
assert sp.simplify(w_bend_h0 - w_drift_expected) == sp.zeros(6, 1)
display(sp.Eq(sp.Symbol(r"\lim_{h\to 0} w_{bend}"), w_bend_h0))

# The quadrupole tends to the drift space for g -> 0, except for the terms that vanish with k: all T_ijk of the
# quadrupole with a factor g or phi tend to the drift values.
w_quad_g0 = w_quad_expected.applyfunc(lambda e: sp.limit(e, g, 0))
assert sp.simplify(w_quad_g0 - w_drift_expected) == sp.zeros(6, 1)

# Thin-lens limit of the quadrupole: L -> 0 with kL = q fixed. All second-order terms vanish.
q = sp.Symbol("q", positive=True)
thin = w_quad_expected.subs(g, sp.sqrt(q / L)).applyfunc(lambda e: sp.limit(sp.simplify(e), L, 0, "+"))
display(sp.Eq(sp.Symbol(r"w_{thin}"), thin))
assert thin == sp.zeros(6, 1)

# %%
# Check 6: comparison with the exact EOM. The exact Hamiltonian of [2026-001, Sec. 10] with k_s = 0.
R_exact = sp.sqrt(1 + 2 * d + (1 - 1 / gamma0**2) * d**2 - px**2 - py**2)
H_exact = -(1 + h * x_) * R_exact + d + h * (x_ + h * x_**2 / 2) + k_sym * (x_**2 - y_**2) / 2
rhs_exact = [sp.diff(H_exact, px), -sp.diff(H_exact, x_), sp.diff(H_exact, py), -sp.diff(H_exact, y_),
             sp.diff(H_exact, d), -sp.diff(H_exact, z_s)]
rhs_exact_func = sp.lambdify([*state, h, k_sym, gamma0], rhs_exact, "numpy")

NUMERIC_CASES = {
    # name: (h, k, g, length)
    "drift": (0.0, 0.0, 1.0, 1.3),
    "bend": (0.2, 0.0, 1.0, 1.5),
    "quad k>0": (0.0, 2.0, float(np.sqrt(2.0)), 0.4),
    "quad k<0": (0.0, -2.0, float(np.sqrt(2.0)), 0.4),
}


def map_value(M_of, w: sp.Matrix, params: dict, length: float, u_start: np.ndarray, second_order: bool) -> np.ndarray:
    """Evaluate M u0 (+ w) at numeric parameters."""
    values = {**params, gamma0: GAMMA0_NUM, L: length}
    M_num = np.array(M_of(L).subs(values).evalf(), dtype=float)
    result = M_num @ u_start
    if second_order:
        w_func = sp.lambdify([*u0], w.subs(values), "numpy")
        result = result + np.array(w_func(*u_start), dtype=float).reshape(6)
    return result


def errors(case: str, amplitude: float) -> tuple:
    """Max errors of the linear map and of the second-order map with respect to the exact EOM."""
    h_num, k_num, g_num, length = NUMERIC_CASES[case]
    params = {h: h_num, g: g_num}
    u_start = amplitude * ENTRANCE_DIRECTION

    def rhs(_s: float, u: np.ndarray) -> list:
        return rhs_exact_func(*u, h_num, k_num, GAMMA0_NUM)

    sol = solve_ivp(rhs, (0.0, length), u_start, rtol=ODE_RTOL, atol=ODE_ATOL, method="DOP853")
    u_exact = sol.y[:, -1]
    M_of = ELEMENTS[case][0]
    err_1 = np.max(np.abs(map_value(M_of, w_expected[case], params, length, u_start, False) - u_exact))
    err_2 = np.max(np.abs(map_value(M_of, w_expected[case], params, length, u_start, True) - u_exact))
    return err_1, err_2


for case in NUMERIC_CASES:
    lin_a, second_a = errors(case, AMPLITUDE)
    lin_half, second_half = errors(case, AMPLITUDE / 2)
    ratio_lin, ratio_second = lin_a / lin_half, second_a / second_half
    print(f"{case:9s} linear error {lin_a:.2e} (ratio {ratio_lin:.2f}), "
          f"second-order error {second_a:.2e} (ratio {ratio_second:.2f})")
    assert abs(ratio_lin - 4.0) < 4.0 * RATIO_TOLERANCE
    assert abs(ratio_second - 8.0) < 8.0 * RATIO_TOLERANCE
    assert second_a < lin_a

# %% [markdown]
# ## 7. Summary and open points
#
# 1. The second-order part of the map is $\mathbf{w}(L) = \int_0^L \mathbf{M}(L - s)\,\mathbf{f}_2(\mathbf{M}(s)\mathbf{u}_0)\,ds$,
#    and $T_{ijk}$ is the coefficient of $u_{0j}u_{0k}$ ($j \le k$) in $w_i$.
# 2. Drift space: $-L\delta_0 p_{x0}$ and $-L\delta_0 p_{y0}$ in $x$ and $y$ (chromatic slope), and the path-length terms
#    $-L(p_{x0}^2 + p_{y0}^2)/2 - 3L\delta_0^2/(2\gamma_0^2)$ in $z$.
# 3. Sector dipole magnet: closed forms in Section 4, which tend to the drift space for $h \to 0$.
# 4. Quadrupole magnet: all second-order terms of $x$, $p_x$, $y$, $p_y$ are proportional to $\delta_0$ (chromatic
#    effect of $x' = p_x(1 - \delta)$), those of $z$ come from the slope and $\delta_0^2$. A thin quadrupole has no
#    second-order terms in the canonical variables.
# 5. Symplecticity: the map truncated at second order is symplectic through first order in $\mathbf{u}_0$ only. The
#    first defect is $\mathbf{W}^\mathsf{T}\mathbf{J}\mathbf{W}$, quadratic in $\mathbf{u}_0$. It vanishes
#    identically for the drift space, scales as $\sin^2\theta$ for the bend, and as $\delta_0^2$ for the quadrupole.
#    The third-order terms of the exact map compensate it.
# 6. Checks: ODE residual, second-order symplectic condition, composition, limits, and the error scaling
#    $a^2 \to a^3$ against the exact EOM.
#
# Open points (not covered here):
#
# - Chromaticity from the second-order terms of a ring (needs a periodic ring and the dispersion), and the conversion
#   of $T_{ijk}$ to the slope variables $x'$, $y'$.
# - The third-order map and its comparison with a symplectic integrator.
# - Fringe fields, edge focusing, and the combined-function magnet.
# - Sextupole and octupole terms, quadrupole curvature corrections, skew quadrupole, and RF.

# %% [markdown]
# ## License
#
# The narrative text in this notebook is licensed under
# [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). Code is licensed under GPL-3.0-or-later.
# See [LICENSING.md](../../LICENSING.md).
