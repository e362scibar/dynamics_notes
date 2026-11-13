# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: dynamics_notes
#     language: python
#     name: python3
# ---

# %% [markdown]
# - File: `topics/curvilinear_coordinates/curvilinear_coordinates.py`
# - Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# - SPDX-License-Identifier: GPL-3.0-or-later
# - Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

# %% [markdown]
# # Curvilinear (Frenet-Serret) coordinates and the Hamiltonian of a bending magnet
#
# Memo on the transverse and longitudinal Hamiltonian in a curvilinear frame that follows the ideal (reference) orbit
# through a planar bending magnet. The goal is the equations of motion (EOM) for a particle around the reference orbit.
#
# Scope:
# - Planar reference orbit (torsion = 0), so the frame is the Frenet-Serret frame of a planar curve.
# - Static magnetic fields only (no RF, no electric field), so the Hamiltonian does not depend on the time-like variable.
# - Linear EOM are the main result, including the dispersion equation. Higher-order terms are noted but not expanded.
#
# References: [lee2019, Chap. 2]; [wiedemann2015, around p. 93].
# The page-level content of those references was not re-checked while writing this memo.
#
# Conventions used here:
# - $s$ is the arc length along the reference orbit and the independent variable.
# - $x$ is the horizontal coordinate, positive outward (away from the center of curvature).
# - $y$ is the vertical coordinate.
# - $h(s) = 1/\rho(s)$ is the curvature of the reference orbit; $h > 0$ for a bend whose center lies inside.
# - $q$ and $m$ are the charge and mass; $P_0$ and $\beta_0$ are the reference momentum and velocity factor.
# - Some literature uses other signs for $x$ and $h$. Check the sign convention before comparing formulas.

# %% [markdown]
# ## 1. Frenet-Serret frame and metric
#
# For a planar curve $\mathbf{R}(s)$ in the horizontal plane with curvature $h(s)$ and zero torsion,
# the Frenet-Serret equations in this convention are
#
# $$
# \frac{d\mathbf{T}}{ds} = -h\,\mathbf{e}_x, \qquad
# \frac{d\mathbf{e}_x}{ds} = h\,\mathbf{T}, \qquad
# \mathbf{T} = \frac{d\mathbf{R}}{ds},
# $$
#
# where $\mathbf{T} = d\mathbf{R}/ds$ is the unit tangent vector of $\mathbf{R}(s)$, $\mathbf{e}_x$ is the outward
# horizontal unit normal, and $\mathbf{e}_y$ is the vertical unit vector.
# The position of a particle near the reference orbit is
#
# $$
# \mathbf{r}(s, x, y) = \mathbf{R}(s) + x\,\mathbf{e}_x(s) + y\,\mathbf{e}_y .
# $$
#
# Differentiating gives $\partial_s \mathbf{r} = (1 + hx)\,\mathbf{T} + \dots$, while $\partial_x \mathbf{r} = \mathbf{e}_x$
# and $\partial_y \mathbf{r} = \mathbf{e}_y$. The line element is therefore
#
# $$
# d\ell^2 = (1 + hx)^2\,ds^2 + dx^2 + dy^2 .
# $$
#
# The factor $1 + hx$ is the only non-trivial metric coefficient. It is the reason the Hamiltonian is not
# a simple Cartesian one.

# %%
import sympy as sp
import numpy as np
from IPython.display import display

# %%
# Check the metric with an explicit circular orbit of radius rho.
# Horizontal plane: (X, Z); vertical: Y.
s, x, y, rho = sp.symbols('s x y rho', positive=True)
theta = s / rho
r_vec = sp.Matrix([(rho + x) * sp.cos(theta), (rho + x) * sp.sin(theta), y])
jac = r_vec.jacobian([s, x, y])
metric = sp.simplify(jac.T * jac)
display(metric)
# Expected: diag((1 + x/rho)**2, 1, 1), i.e. h = 1/rho.
assert sp.simplify(metric[0, 0] - (1 + x / rho) ** 2) == 0
assert sp.simplify(metric[1, 1] - 1) == 0 and sp.simplify(metric[2, 2] - 1) == 0
assert sp.simplify(metric[0, 1]) == 0 and sp.simplify(metric[0, 2]) == 0 and sp.simplify(metric[1, 2]) == 0

# %% [markdown]
# ## 2. Lagrangian with $s$ as the independent variable
#
# For a particle with charge $q$ in a static magnetic field with potentials $(A_x, A_y, A_s)$ (physical components along
# $\mathbf{e}_x$, $\mathbf{e}_y$, $\mathbf{T}$), the Lagrangian with time $t$ as the independent variable is
#
# $$
# L = -mc^2\sqrt{1 - \frac{v^2}{c^2}} + q\,\mathbf{v}\cdot\mathbf{A}, \qquad
# v_T = (1+hx)\frac{ds}{dt}, \quad v_x = \frac{dx}{dt}, \quad v_y = \frac{dy}{dt}.
# $$
#
# **Euler-Lagrange equation and the Lorentz force.** The Euler-Lagrange equation for the Cartesian position
# $\mathbf{r}$ is
#
# $$
# \frac{d}{dt}\frac{\partial L}{\partial \mathbf{v}} - \frac{\partial L}{\partial \mathbf{r}} = 0,
# \qquad \mathbf{v} = \dot{\mathbf{r}} .
# $$
#
# The velocity derivative is $\partial L/\partial\mathbf{v} = \gamma m\mathbf{v} + q\mathbf{A}$ with
# $\gamma = 1/\sqrt{1 - v^2/c^2}$. The position derivative, with $\mathbf{v}$ held fixed, is
#
# $$
# \frac{\partial L}{\partial \mathbf{r}} = q\,\nabla(\mathbf{v}\cdot\mathbf{A})
# = q\big[(\mathbf{v}\cdot\nabla)\mathbf{A} + \mathbf{v}\times(\nabla\times\mathbf{A})\big].
# $$
#
# The vector potential is evaluated along the trajectory, so
# $d\mathbf{A}/dt = \partial\mathbf{A}/\partial t + (\mathbf{v}\cdot\nabla)\mathbf{A}$. Substituting these
# into the Euler-Lagrange equation, the directional derivative $(\mathbf{v}\cdot\nabla)\mathbf{A}$ of the vector
# potential along the trajectory cancels and
#
# $$
# \frac{d}{dt}(\gamma m\,\mathbf{v}) = q\,\mathbf{v}\times\mathbf{B} - q\,\frac{\partial \mathbf{A}}{\partial t},
# \qquad \mathbf{B} = \nabla\times\mathbf{A} .
# $$
#
# For a static magnetic field $\partial\mathbf{A}/\partial t = 0$, so this is the Lorentz force equation
# $d\mathbf{p}/dt = q\,\mathbf{v}\times\mathbf{B}$ with $\mathbf{p} = \gamma m\mathbf{v}$. The next cell checks
# the identity with sympy.

# %%
# Euler-Lagrange equation in Cartesian coordinates: d/dt(dL/dv) - dL/dr = 0 must give the Lorentz force.
t_ = sp.symbols('t', real=True)
m_, c_, q_ = sp.symbols('m c q', positive=True)
X = sp.Function('X')(t_)
Y = sp.Function('Y')(t_)
Z = sp.Function('Z')(t_)
Ax_f = sp.Function('Ax')(X, Y, Z)
Ay_f = sp.Function('Ay')(X, Y, Z)
Az_f = sp.Function('Az')(X, Y, Z)
vx, vy, vz = sp.diff(X, t_), sp.diff(Y, t_), sp.diff(Z, t_)
v2 = vx**2 + vy**2 + vz**2
gamma = 1 / sp.sqrt(1 - v2 / c_**2)
L_cart = -m_ * c_**2 * sp.sqrt(1 - v2 / c_**2) + q_ * (vx * Ax_f + vy * Ay_f + vz * Az_f)

Bx = sp.diff(Az_f, Y) - sp.diff(Ay_f, Z)
By = sp.diff(Ax_f, Z) - sp.diff(Az_f, X)
Bz = sp.diff(Ay_f, X) - sp.diff(Ax_f, Y)
v_cross_B = (vy * Bz - vz * By, vz * Bx - vx * Bz, vx * By - vy * Bx)

for name, coord, vel, cross in zip(("x", "y", "z"), (X, Y, Z), (vx, vy, vz), v_cross_B):
    euler_lagrange = sp.diff(sp.diff(L_cart, vel), t_) - sp.diff(L_cart, coord)
    lorentz = sp.diff(gamma * m_ * vel, t_) - q_ * cross
    residual = sp.simplify(euler_lagrange - lorentz)
    print(f"{name}: residual = {residual}")
    assert residual == 0

# %% [markdown]
# Use $s$ as the independent variable by writing $t' = dt/ds$, $x' = dx/ds$, $y' = dy/ds$. Then
# $v_T t' = 1 + hx$, $v_x t' = x'$, $v_y t' = y'$, and $L\,dt = L_s\,ds$ with
#
# $$
# L_s = -mc^2\,\underbrace{\sqrt{t'^2 - \frac{(1+hx)^2 + x'^2 + y'^2}{c^2}}}_{Q}
#       + q\big[(1+hx)A_s + x'A_x + y'A_y\big].
# $$
#
# The variable $t'$ is not independent. The tangential velocity is $v_T = (1+hx)\,ds/dt = (1+hx)/t'$, and the
# tangential mechanical momentum is $P_T = \gamma m v_T = E v_T/c^2$, so
#
# $$
# t' = \frac{(1+hx)\,E}{P_T c^2}, \qquad
# P_T = \sqrt{\frac{E^2}{c^2} - m^2c^2 - P_x^2 - P_y^2} .
# $$
#
# The generalized coordinates are $(t, x, y)$ and their conjugate momenta are
#
# $$
# p_t = \frac{\partial L_s}{\partial t'} = -\gamma m c^2 = -E, \qquad
# p_x = \frac{\partial L_s}{\partial x'} = \gamma m v_x + qA_x, \qquad
# p_y = \frac{\partial L_s}{\partial y'} = \gamma m v_y + qA_y .
# $$
#
# The Hamiltonian with $s$ as "time" is $K = p_x x' + p_y y' + p_t t' - L_s$. Let $P_x = p_x - qA_x$ and
# $P_y = p_y - qA_y$ be the mechanical transverse momenta. Substituting the definitions and using
# $E^2/c^2 - m^2c^2 = p^2$, with $p^2 = P_T^2 + P_x^2 + P_y^2$ and $P_T = \gamma m v_T$, gives
#
# $$
# K = -(1 + hx)\Big[\sqrt{\frac{E^2}{c^2} - m^2c^2 - P_x^2 - P_y^2} + qA_s\Big].
# $$
#
# Here $E = -p_t$ and the square root is the tangential mechanical momentum $P_T$.
# The $-(1+hx)$ prefactor comes from $t'$ and the metric factor, and it multiplies the whole tangential part.
#
# The next cell checks this Legendre transform numerically at random values.

# %%
m_, c_, q_, h_, xv, X1, Y1, T1, As, Ax, Ay = sp.symbols('m c q h x X1 Y1 T1 As Ax Ay', real=True)
pt, px_, py_ = sp.symbols('pt px py', real=True)
Q_ = sp.sqrt(T1**2 - ((1 + h_ * xv)**2 + X1**2 + Y1**2) / c_**2)
L_s = -m_ * c_**2 * Q_ + q_ * ((1 + h_ * xv) * As + X1 * Ax + Y1 * Ay)
p_x_expr = sp.diff(L_s, X1)
p_y_expr = sp.diff(L_s, Y1)
p_t_expr = sp.diff(L_s, T1)
K_expr = p_x_expr * X1 + p_y_expr * Y1 + p_t_expr * T1 - L_s
# Closed form from the text, with E = -p_t and P = p - qA.
E_expr = -p_t_expr
P_T_expr = sp.sqrt(E_expr**2 / c_**2 - m_**2 * c_**2 - (p_x_expr - q_ * Ax)**2 - (p_y_expr - q_ * Ay)**2)
K_closed = -(1 + h_ * xv) * (P_T_expr + q_ * As)

numeric_values = {
    m_: 0.5, c_: 2.0, q_: -1.0, h_: 0.3, xv: 0.2, X1: 0.1, Y1: -0.05, T1: 1.5, As: 0.7, Ax: 0.2, Ay: -0.1,
}
K_num = complex(sp.N(K_expr.subs(numeric_values)))
K_closed_num = complex(sp.N(K_closed.subs(numeric_values)))
print(f"K (Legendre transform) = {K_num.real:.12f}")
print(f"K (closed form)        = {K_closed_num.real:.12f}")
assert abs(K_num - K_closed_num) < 1e-10

# t' is not independent: t' = (1+hx) E / (P_T c^2).
t_prime_closed = (1 + h_ * xv) * E_expr / (P_T_expr * c_**2)
t_prime_num = complex(sp.N(T1.subs(numeric_values)))
t_prime_closed_num = complex(sp.N(t_prime_closed.subs(numeric_values)))
print(f"t' (independent)       = {t_prime_num.real:.12f}")
print(f"t' (closed form)       = {t_prime_closed_num.real:.12f}")
assert abs(t_prime_num - t_prime_closed_num) < 1e-10

# %% [markdown]
# ## 3. Normalization: $\delta$, $p_x$, $p_y$ and the normalized Hamiltonian
#
# Normalize with the reference momentum $P_0$ (constant along the ring):
#
# $$
# 1 + \delta = \frac{p}{P_0}, \qquad p_x \to \frac{P_x}{P_0}, \qquad p_y \to \frac{P_y}{P_0}, \qquad
# a_x = \frac{qA_x}{P_0}, \quad a_y = \frac{qA_y}{P_0}, \quad a_s = \frac{qA_s}{P_0} .
# $$
#
# Dividing $K$ by $P_0$ and writing the transverse momenta in normalized form:
#
# $$
# H = -(1+hx)\sqrt{(1+\delta)^2 - (p_x - a_x)^2 - (p_y - a_y)^2} \; - \; (1+hx)\,a_s .
# $$
#
# The time-like variable $\tau$ is conjugate to $\delta$. It is obtained from $(t, E)$ by a canonical transformation
# (the standard choice in [lee2019] and [wiedemann2015]). For a static magnetic field, $H$ does not depend on $\tau$, so
# $\delta$ is a constant of motion. The transverse EOM do not depend on $\tau$ either, so $\tau$ is not needed here.
#
# **Energy deviation.** Some texts use the energy deviation $\delta_L = \Delta E/(P_0 c)$ instead of $\delta$.
# With $E = E_0 + \Delta E$ and $E_0 = \gamma_0 mc^2$,
#
# $$
# \Big(\frac{p}{P_0}\Big)^2 = 1 + \frac{2\,\delta_L}{\beta_0} + \delta_L^2 ,
# $$
#
# which reduces to $(1+\delta)^2$ at first order in $\delta_L$. The next cell checks this identity symbolically.

# %%
dL, beta0, E0, p0c = sp.symbols('delta_L beta0 E0 p0c', positive=True)
# With E0 = p0*c^2 / (beta0 * c) = p0c / beta0, and p0c = beta0 * E0 is the standard relation.
E_ref = p0c / beta0
dE = dL * p0c
lhs = ((E_ref + dE) ** 2 - (E_ref**2 - p0c**2)) / p0c**2
rhs = 1 + 2 * dL / beta0 + dL**2
display(sp.simplify(lhs - rhs))
assert sp.simplify(lhs - rhs) == 0

# %% [markdown]
# ## 4. Vector potentials of the ideal dipole and the quadrupole
#
# ### Dipole
#
# An ideal sector dipole has a uniform vertical field $B_y = B_0$ in the Cartesian sense. In the curvilinear
# coordinates, with $A_x = 0$ and metric factor $1+hx$, the vertical component of the curl is
#
# $$
# B_y \propto \frac{1}{1+hx}\,\frac{\partial}{\partial x}\big[(1+hx)\,A_s\big] .
# $$
#
# Requiring this to be uniform gives $(1+hx)\,a_s = -h\,(x + \tfrac12 h x^2)$, that is
#
# $$
# a_s = -\frac{h\,(x + \tfrac12 h x^2)}{1+hx}, \qquad h = \frac{qB_0}{P_0} = \frac{1}{\rho}.
# $$
#
# The overall sign is a convention. It is fixed below by requiring the reference orbit ($x = 0$, $p = 0$,
# $\delta = 0$) to be an equilibrium of $H$. With this $a_s$, the dipole term in $H$ is exact:
#
# $$
# -(1+hx)\,a_s = h\,\big(x + \tfrac12 h x^2\big) .
# $$
#
# So the dipole part of $H$ is
#
# $$
# H_{\text{dip}} = -(1+hx)\sqrt{(1+\delta)^2 - p_x^2 - p_y^2} + h\,\big(x + \tfrac12 h x^2\big).
# $$
#
# ### Quadrupole
#
# For a straight quadrupole with normalized gradient $k(s) = qg/P_0$ (focusing in $x$ for $k > 0$), the
# potential term at the order needed for linear optics is $H_{\text{quad}} = \tfrac12 k\,(x^2 - y^2)$.
# Curvature corrections to the quadrupole term are of higher order in $x$ and are not kept here.

# %%
x_, y_, px, py, d, h, k = sp.symbols('x y p_x p_y delta h k', real=True)

# Exact dipole potential: check the curl condition and the expansion of -(1+hx) a_s.
a_s_dip = -h * (x_ + h * x_**2 / 2) / (1 + h * x_)
curl_check = sp.simplify(sp.diff((1 + h * x_) * a_s_dip, x_) / (1 + h * x_))
display(curl_check)  # expected: -h (uniform, the constant that encodes B0 = P0 h / q)
assert sp.simplify(curl_check + h) == 0
assert sp.simplify(-(1 + h * x_) * a_s_dip - h * (x_ + h * x_**2 / 2)) == 0

# %% [markdown]
# ## 5. Full Hamiltonian and exact equations of motion
#
# Collecting the terms, with the straight-quadrupole approximation for the quadrupole:
#
# $$
# H(x, p_x, y, p_y, \delta; s) = -(1+hx)\sqrt{(1+\delta)^2 - p_x^2 - p_y^2}
#   + h\big(x + \tfrac12 h x^2\big) + \tfrac12 k\,(x^2 - y^2).
# $$
#
# Hamilton's equations with $s$ as the independent variable are
#
# $$
# x' = \frac{\partial H}{\partial p_x}, \qquad p_x' = -\frac{\partial H}{\partial x}, \qquad
# y' = \frac{\partial H}{\partial p_y}, \qquad p_y' = -\frac{\partial H}{\partial y}, \qquad
# \delta' = -\frac{\partial H}{\partial \tau} = 0 .
# $$
#
# The reference orbit is an equilibrium only if $\partial H/\partial x = 0$ at $x = p = \delta = 0$.
# The next cell checks this and prints the exact EOM.

# %%
H_full = -(1 + h * x_) * sp.sqrt((1 + d) ** 2 - px**2 - py**2) + h * (x_ + h * x_**2 / 2) + k * (x_**2 - y_**2) / 2
state = [x_, px, y_, py, d]

# Reference orbit is an equilibrium: dH/dx vanishes at the origin.
dHdx_ref = sp.simplify(sp.diff(H_full, x_).subs({x_: 0, px: 0, py: 0, y_: 0, d: 0}))
display(dHdx_ref)
assert dHdx_ref == 0

# Exact EOM.
rhs_exact = [sp.diff(H_full, px), -sp.diff(H_full, x_), sp.diff(H_full, py), -sp.diff(H_full, y_), 0]
for name, eq in zip(["x'", "p_x'", "y'", "p_y'", "delta'"], rhs_exact):
    print(name, "=")
    display(sp.simplify(eq))

# %% [markdown]
# ## 6. Linearized equations of motion
#
# Linearize around the reference orbit. The Jacobian of the exact EOM at the origin gives the linear matrix
# $\mathbf{A}$, with $\mathbf{z}' = \mathbf{A}\,\mathbf{z}$ and $\mathbf{z} = (x, p_x, y, p_y, \delta)^\mathsf{T}$.
# Expected result:
#
# $$
# \mathbf{A} = \begin{pmatrix}
# 0 & 1 & 0 & 0 & 0 \\
# -(h^2 + k) & 0 & 0 & 0 & h \\
# 0 & 0 & 0 & 1 & 0 \\
# 0 & 0 & k & 0 & 0 \\
# 0 & 0 & 0 & 0 & 0
# \end{pmatrix}.
# $$
#
# This gives the Hill equations and the inhomogeneous term $h\delta$ from the bend:
#
# $$
# x'' + (h^2 + k)\,x = h\,\delta, \qquad y'' - k\,y = 0 .
# $$

# %%
state_zero = {x_: 0, px: 0, y_: 0, py: 0, d: 0}
A_mat = sp.Matrix(rhs_exact).jacobian(state)
A_mat = sp.simplify(A_mat.subs(state_zero))
display(A_mat)
A_expected = sp.Matrix([
    [0, 1, 0, 0, 0],
    [-(h**2 + k), 0, 0, 0, h],
    [0, 0, 0, 1, 0],
    [0, 0, k, 0, 0],
    [0, 0, 0, 0, 0],
])
assert sp.simplify(A_mat - A_expected) == sp.zeros(5, 5)

# %% [markdown]
# ### Checks against known results
#
# - Dispersion: with constant $h$ and $k$, a constant solution $D = h/(h^2 + k)$ satisfies
#   $D'' + (h^2 + k)D = h$. This is the closed-orbit (periodic) dispersion of a uniform bending section.
# - Vertical plane: no $h^2$ term, as expected for a bend in the horizontal plane.
# - Linear coupling: none in this Hamiltonian (no skew terms).

# %%
D_const = h / (h**2 + k)
# For constant h and k the second derivative in s of D_const is zero, so the check is algebraic.
assert sp.simplify((h**2 + k) * D_const - h) == 0

# %% [markdown]
# ## 7. Cross-check: Euler-Lagrange equations of $L_s$
#
# Take $m = c = 1$ and the gauge $A_x = A_y = 0$, with $qA_s = P_0 a_s$. The $t$ equation gives the constant
# $\partial L_s/\partial t' = -E$, which fixes $t'$. The $x$ and $y$ equations are then second-order equations for
# $x(s)$ and $y(s)$. Linearizing them with $\sqrt{E^2-1} = P_0(1+\delta)$ must reproduce
# $x'' + (h^2+k)x = h\delta$ and $y'' - ky = 0$.

# %%
tp, xp, yp, xv, yv = sp.symbols('tp xp yp xv yv', real=True)
E_sym, delta_sym = sp.symbols('E delta', real=True)
P0_sym = sp.Symbol('P0', positive=True)
S_expr = (1 + h * xv) ** 2 + xp**2 + yp**2
F_expr = P0_sym * (-h * (xv + h * xv**2 / 2) / (1 + h * xv) - k * (xv**2 - yv**2) / 2)
L_fn = -sp.sqrt(tp**2 - S_expr) + (1 + h * xv) * F_expr

# t equation: dL/dt' = -E is a constant of motion. Its positive root is t' = E sqrt(S/(E^2-1)).
tp_expected = E_sym * sp.sqrt(S_expr / (E_sym**2 - 1))
t_residual = sp.diff(L_fn, tp).subs(tp, tp_expected) + E_sym
for values in [{E_sym: 2.5, h: 0.3, xv: 0.1, xp: 0.2, yp: -0.1}, {E_sym: 4.0, h: -0.2, xv: -0.3, xp: 0.1, yp: 0.4}]:
    assert abs(complex(sp.N(t_residual.subs(values)))) < 1e-12
print("t' = E sqrt(S/(E^2-1)) satisfies dL/dt' = -E")

# Momenta and the x-force with t' eliminated, checked against the closed forms in the memo (m = c = 1).
p_x_closed = sp.sqrt(E_sym**2 - 1) * xp / sp.sqrt(S_expr)
p_x_expr = sp.diff(L_fn, xp).subs(tp, tp_expected)
dL_dx_expr = sp.diff(L_fn, xv).subs(tp, tp_expected)
gamma_v_T = sp.sqrt(E_sym**2 - 1) * (1 + h * xv) / sp.sqrt(S_expr)
for values in [{E_sym: 2.5, h: 0.3, xv: 0.1, xp: 0.2, yp: -0.1, P0_sym: 1.0, k: 0.4, yv: 0.05},
               {E_sym: 4.0, h: -0.2, xv: -0.3, xp: 0.1, yp: 0.4, P0_sym: 2.0, k: -0.7, yv: -0.2}]:
    assert abs(complex(sp.N((p_x_expr - p_x_closed).subs(values)))) < 1e-12
    # The curvature part of dL/dx equals h * gamma m v_T; the rest is the potential term.
    curvature_part = dL_dx_expr - (h * F_expr + (1 + h * xv) * sp.diff(F_expr, xv))
    assert abs(complex(sp.N((curvature_part - h * gamma_v_T).subs(values)))) < 1e-12
print("p_x = sqrt(E^2-1) x' / sqrt(S) and the curvature term of dL/dx checked numerically")

# Trajectory: x(s), y(s) are O(eps), and delta is O(eps). Substitute and expand the Euler-Lagrange equations.
eps = sp.Symbol('epsilon')
X, Y = sp.Function('X')(s), sp.Function('Y')(s)
dL_dyp = sp.diff(L_fn, yp).subs(tp, tp_expected)
dL_dy = sp.diff(L_fn, yv).subs(tp, tp_expected)
trajectory = {xv: eps * X, yv: eps * Y, xp: eps * X.diff(s), yp: eps * Y.diff(s), E_sym: sp.sqrt(1 + P0_sym**2 * (1 + eps * delta_sym)**2)}
EL_x = sp.diff(p_x_expr.subs(trajectory), s) - dL_dx_expr.subs(trajectory)
EL_y = sp.diff(dL_dyp.subs(trajectory), s) - dL_dy.subs(trajectory)
order0_x = sp.simplify(sp.series(EL_x, eps, 0, 2).removeO().coeff(eps, 0))
order1_x = sp.simplify(sp.series(EL_x, eps, 0, 2).removeO().coeff(eps, 1))
order1_y = sp.simplify(sp.series(EL_y, eps, 0, 2).removeO().coeff(eps, 1))
print("zeroth order of x equation:", order0_x)
print("first order of x equation:", order1_x)
print("first order of y equation:", order1_y)
assert order0_x == 0
assert sp.simplify(order1_x - P0_sym * (X.diff(s, 2) + (h**2 + k) * X - h * delta_sym)) == 0
assert sp.simplify(order1_y - P0_sym * (Y.diff(s, 2) - k * Y)) == 0

# %% [markdown]
# ## 8. Summary
#
# 1. Metric in Frenet-Serret coordinates: $d\ell^2 = (1+hx)^2 ds^2 + dx^2 + dy^2$.
# 2. Hamiltonian with $s$ as the independent variable:
#    $H = -(1+hx)\sqrt{(1+\delta)^2 - (p_x-a_x)^2 - (p_y-a_y)^2} - (1+hx)\,a_s$.
# 3. Ideal dipole: $-(1+hx)\,a_s = h(x + \tfrac12 h x^2)$, so the bend adds the term $h x$ (cancelled by the
#    $-(1+hx)$ term at equilibrium) and the $\tfrac12 h^2 x^2$ term, which produces the $h^2$ focusing.
# 4. Linear EOM: $x'' + (h^2+k)x = h\delta$ and $y'' - ky = 0$ (no RF, so $\delta' = 0$).
#
# ## Open points
#
# - The RF cavity adds an electric potential, so $\delta$ and $\tau$ are no longer constant. Not covered here.
# - Quadrupole curvature corrections and sextupole/octupole terms are not included. They are needed for
#   nonlinear dynamics (see the notebook on dynamic aperture).
# - Nonplanar reference orbits (non-zero torsion) add a term to the frame. Not needed for SPring-8-II dipoles.
# - Whether the $\tau$-canonical transformation gives exactly the form above, with no additional
#   $\delta$-dependent metric factor, should be checked against [lee2019, Chap. 2] before using $H$ beyond
#   the linear level.

# %% [markdown]
# ## License
#
# The narrative text in this notebook is licensed under
# [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). Code is licensed under GPL-3.0-or-later.
# See [LICENSING.md](../../LICENSING.md).
