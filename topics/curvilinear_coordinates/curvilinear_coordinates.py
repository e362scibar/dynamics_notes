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
# - File: `topics/curvilinear_coordinates/curvilinear_coordinates.py`
# - Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# - SPDX-License-Identifier: GPL-3.0-or-later
# - Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

# %% [markdown]
# # Canonical equations of motion in curvilinear (Frenet-Serret) coordinates for beamlines with bending magnets
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
# ## 2. Frenet-Serret frame and metric
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
# The velocity of a particle is $\mathbf{v} = \dot{\mathbf{r}}$ with $\dot s = ds/dt$. From $\partial_s\mathbf{r}$,
# $\partial_x\mathbf{r}$, and $\partial_y\mathbf{r}$, it is
#
# $$
# \mathbf{v} = (1+hx)\,\dot s\,\mathbf{T} + \dot x\,\mathbf{e}_x + \dot y\,\mathbf{e}_y .
# $$
#
# Its tangential component is therefore $v_T = (1+hx)\,\dot s$. This is the origin of the metric factor.
#
# The factor $1 + hx$ is the only non-trivial metric coefficient. It is the reason the Hamiltonian is not
# a simple Cartesian one.

# %%
import sympy as sp
import numpy as np
from IPython.display import display


def show_vec_eq(lhs_vec, rhs_vec):
    """Display a vector equation as a column of component equations."""
    for lhs_c, rhs_c in zip(lhs_vec, rhs_vec):
        display(sp.Eq(lhs_c, rhs_c))

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

# %%
# Frenet-Serret relations (blocks above) for the circular orbit R(s) = rho (cos(s/rho), sin(s/rho), 0).
rho_f, s_f, x_f, y_f = sp.symbols(r'\rho s x y', positive=True)
h_f = 1 / rho_f
th_f = s_f / rho_f
R_v = sp.Matrix([rho_f * sp.cos(th_f), rho_f * sp.sin(th_f), 0])
T_v = R_v.diff(s_f)
ex_v = sp.Matrix([sp.cos(th_f), sp.sin(th_f), 0])
ey_v = sp.Matrix([0, 0, 1])
# Vector form of the Frenet-Serret equations, then the components.
h_sym = sp.Symbol('h', real=True)
T_vec_sym = sp.Symbol(r"\mathbf{T}")
ex_vec_sym = sp.Symbol(r"\mathbf{e}_x")
R_vec_sym_f = sp.Symbol(r"\mathbf{R}(s)")
display(sp.Eq(sp.Symbol(r"\frac{d\mathbf{T}}{ds}"), -h_sym * ex_vec_sym))
show_vec_eq(T_v.diff(s_f), -h_f * ex_v)
display(sp.Eq(sp.Symbol(r"\frac{d\mathbf{e}_x}{ds}"), h_sym * T_vec_sym))
show_vec_eq(ex_v.diff(s_f), h_f * T_v)
display(sp.Eq(T_vec_sym, sp.Symbol(r"\frac{d\mathbf{R}}{ds}")))
show_vec_eq(T_v, R_v.diff(s_f))

# Test results: the Frenet-Serret relations hold for the explicit circular orbit.
check_T = (T_v.diff(s_f) + h_f * ex_v).applyfunc(sp.simplify) == sp.zeros(3, 1)
check_ex = (ex_v.diff(s_f) - h_f * T_v).applyfunc(sp.simplify) == sp.zeros(3, 1)
print("dT/ds = -h e_x holds for the circular orbit:")
display(check_T)
print("de_x/ds = h T holds for the circular orbit:")
display(check_ex)
assert check_T and check_ex

r_v = R_v + x_f * ex_v + y_f * ey_v
# Vector form first, then the components.
r_vec_sym = sp.Symbol(r"\mathbf{r}(s, x, y)")
R_vec_sym = sp.Symbol(r"\mathbf{R}(s)")
ex_vec_sym = sp.Symbol(r"\mathbf{e}_x(s)")
ey_vec_sym = sp.Symbol(r"\mathbf{e}_y")
display(sp.Eq(r_vec_sym, R_vec_sym + x_f * ex_vec_sym + y_f * ey_vec_sym))
r_sym = sp.Matrix(sp.symbols('r_1 r_2 r_3'))
show_vec_eq(r_sym, R_v + x_f * ex_v + y_f * ey_v)

# Line element from the Jacobian of the metric check above.
ds_, dx_, dy_ = sp.symbols('ds dx dy', real=True)
dl_vec = jac * sp.Matrix([ds_, dx_, dy_])
dl2 = sp.expand((dl_vec.T * dl_vec)[0])
dl2_expected = (1 + x / rho) ** 2 * ds_**2 + dx_**2 + dy_**2
display(sp.Eq(sp.Symbol("d\\ell^2"), dl2_expected))
assert sp.simplify(dl2 - dl2_expected) == 0

# Velocity on the circular orbit: v = dr/dt must equal (1+hx) s-dot T + x-dot e_x + y-dot e_y.
tt_v = sp.Symbol('t', real=True)
s_v, x_v, y_v = (sp.Function(name)(tt_v) for name in ('s', 'x', 'y'))
r_v_t = (R_v + x_f * ex_v + y_f * ey_v).subs({s_f: s_v, x_f: x_v, y_f: y_v})
v_v_t = r_v_t.diff(tt_v)
v_model = (1 + x_v / rho_f) * s_v.diff(tt_v) * T_v.subs(s_f, s_v) + x_v.diff(tt_v) * ex_v.subs(s_f, s_v) \
    + y_v.diff(tt_v) * ey_v
check_v = (v_v_t - v_model).applyfunc(sp.simplify) == sp.zeros(3, 1)
print("v = (1+hx) s-dot T + x-dot e_x + y-dot e_y on the circular orbit:")
display(check_v)
assert check_v

# %% [markdown]
# ## 3. Lagrangian and equations of motion
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
Ax_f = sp.Function('A_x')(X, Y, Z)
Ay_f = sp.Function('A_y')(X, Y, Z)
Az_f = sp.Function('A_z')(X, Y, Z)
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
    print(f"{name}: residual =")
    display(residual)
    assert residual == 0

# %%
# Time-based equations (blocks above), in the Cartesian sympy model of the cell above.
display(sp.Eq(sp.Symbol('L'), L_cart))
vT_sym, sdot, h_t, x_t = sp.symbols('v_T sdot h x', real=True)
display(sp.Eq(vT_sym, (1 + h_t * x_t) * sdot))
display(sp.Eq(sp.Symbol('v_x'), sp.diff(X, t_)))
display(sp.Eq(sp.Symbol('v_y'), sp.diff(Y, t_)))

# Euler-Lagrange equation (x component) and its right-hand side.
EL_x = sp.diff(sp.diff(L_cart, vx), t_) - sp.diff(L_cart, X)
display(sp.Eq(EL_x, 0))

# Position derivative: dL/dr = q[(v.grad)A + v x (curl A)] (x component) with B = curl A.
dL_dX = sp.diff(L_cart, X)
grad_form = q_ * (vx * sp.diff(Ax_f, X) + vy * sp.diff(Ay_f, X) + vz * sp.diff(Az_f, X))
curl_form = q_ * (vx * sp.diff(Ax_f, X) + vy * sp.diff(Ax_f, Y) + vz * sp.diff(Ax_f, Z) + v_cross_B[0])
display(sp.Eq(sp.Symbol(r'\frac{\partial L}{\partial x}'), dL_dX))
assert sp.simplify(dL_dX - grad_form) == 0
assert sp.simplify(grad_form - curl_form) == 0

# Lorentz force equation (x component): d/dt(gamma m v_x) = q (v x B)_x for static fields.
display(sp.Eq(sp.diff(gamma * m_ * vx, t_), q_ * v_cross_B[0]))

# %% [markdown]
# ## 4. Lagrangian with $s$ as the independent variable
#
# Before writing $L_s$, the vector potential is decomposed in the local orthonormal frame. Its physical components
# along $\mathbf{T}$, $\mathbf{e}_x$, and $\mathbf{e}_y$ are $A_s$, $A_x$, and $A_y$:
#
# $$
# \mathbf{A} = A_s\,\mathbf{T} + A_x\,\mathbf{e}_x + A_y\,\mathbf{e}_y .
# $$
#
# With the velocity of Section 2, the scalar product is
#
# $$
# \mathbf{v}\cdot\mathbf{A} = (1+hx)\,\dot s\,A_s + \dot x\,A_x + \dot y\,A_y .
# $$
#
# The next cell checks both relations on the explicit circular orbit.
#
# Use $s$ as the independent variable by writing $t' = dt/ds$, $x' = dx/ds$, $y' = dy/ds$. Then
# $v_T t' = 1 + hx$, $v_x t' = x'$, $v_y t' = y'$, and $L\,dt = L_s\,ds$ with
#
# $$
# L_s = -mc^2\,\underbrace{\sqrt{t'^2 - \frac{(1+hx)^2 + x'^2 + y'^2}{c^2}}}_{Q}
#       + q\big[(1+hx)A_s + x'A_x + y'A_y\big].
# $$
#
# %%
# Vector potential in the local frame and v . A (blocks above), on the explicit circular orbit of section 2.
tt = sp.Symbol('t', real=True)
s_t, x_t, y_t = (sp.Function(name)(tt) for name in ('s', 'x', 'y'))
AS_, AX_, AY_ = sp.symbols('A_s A_x A_y', real=True)
T_t = T_v.subs(s_f, s_t)
ex_t = ex_v.subs(s_f, s_t)
ey_t = ey_v
A_t = AS_ * T_t + AX_ * ex_t + AY_ * ey_t
display(sp.Eq(sp.Symbol(r"\mathbf{A}"), AS_ * sp.Symbol(r"\mathbf{T}") + AX_ * sp.Symbol(r"\mathbf{e}_x")
              + AY_ * sp.Symbol(r"\mathbf{e}_y")))
r_t = (R_v + x_f * ex_v + y_f * ey_v).subs({s_f: s_t, x_f: x_t, y_f: y_t})
v_t = r_t.diff(tt)
v_dot_A = (v_t.T * A_t)[0]
v_dot_A_model = (1 + x_t / rho_f) * s_t.diff(tt) * AS_ + x_t.diff(tt) * AX_ + y_t.diff(tt) * AY_
display(sp.Eq(sp.Symbol(r"\mathbf{v}\cdot\mathbf{A}"), v_dot_A_model))
check_vA = sp.simplify(v_dot_A - v_dot_A_model) == 0
print("v . A = (1+hx) s-dot A_s + x-dot A_x + y-dot A_y on the circular orbit:")
display(check_vA)
assert check_vA

# %%
# L_s with s as the independent variable and the t' relation (blocks above).
tp_s, xp_s, yp_s = sp.Symbol("t'", real=True), sp.Symbol("x'", real=True), sp.Symbol("y'", real=True)
xv_s, yv_s = sp.symbols('x y', real=True)
As_s, Ax_s, Ay_s = sp.symbols('A_s A_x A_y', real=True)
h_s = sp.Symbol('h', real=True)
E_s = sp.Symbol('E', positive=True)
Px_s, Py_s = sp.symbols('P_x P_y', real=True)
Q_s = sp.sqrt(tp_s**2 - ((1 + h_s * xv_s) ** 2 + xp_s**2 + yp_s**2) / c_**2)
L_s_model = -m_ * c_**2 * Q_s + q_ * ((1 + h_s * xv_s) * As_s + xp_s * Ax_s + yp_s * Ay_s)
display(sp.Eq(sp.Symbol('L_s'), L_s_model))
display(sp.Eq(sp.Symbol('Q'), Q_s))
# %% [markdown]
# ## 5. Euler-Lagrange equations of $L_s$
#
# The variable $t'$ is not independent. The tangential velocity is $v_T = (1+hx)\,ds/dt = (1+hx)/t'$, and the
# tangential mechanical momentum is $P_T = \gamma m v_T = E v_T/c^2$, so
#
# The transverse mechanical momenta are $P_x = \gamma m v_x$ and $P_y = \gamma m v_y$, with $v_x = x'/t'$ and
# $v_y = y'/t'$. Using $\gamma = t'/Q$, these momenta and $P_T = \gamma m v_T$ are written in $Q$ as
#
# $$
# P_x = \frac{m\,x'}{Q}, \qquad P_y = \frac{m\,y'}{Q}, \qquad P_T = \frac{m\,(1+hx)}{Q} .
# $$
#
# Then
#
# $$
# t' = \frac{(1+hx)\,E}{P_T c^2}, \qquad
# P_T = \sqrt{\frac{E^2}{c^2} - m^2c^2 - P_x^2 - P_y^2} .
# $$
#
# %%
# t' relation and the transverse momenta written in terms of Q (section 5).
tp_s, xp_s, yp_s = sp.Symbol("t'", real=True), sp.Symbol("x'", real=True), sp.Symbol("y'", real=True)
xv_s, yv_s = sp.symbols('x y', real=True)
As_s, Ax_s, Ay_s = sp.symbols('A_s A_x A_y', real=True)
h_s = sp.Symbol('h', real=True)
E_s = sp.Symbol('E', positive=True)
Px_s, Py_s = sp.symbols('P_x P_y', real=True)
Q_s = sp.sqrt(tp_s**2 - ((1 + h_s * xv_s) ** 2 + xp_s**2 + yp_s**2) / c_**2)
L_s_model = -m_ * c_**2 * Q_s + q_ * ((1 + h_s * xv_s) * As_s + xp_s * Ax_s + yp_s * Ay_s)
E_s = sp.Symbol('E', positive=True)
Px_s, Py_s = sp.symbols('P_x P_y', real=True)
P_T_s = sp.sqrt(E_s**2 / c_**2 - m_**2 * c_**2 - Px_s**2 - Py_s**2)
display(sp.Eq(sp.Symbol("t'"), (1 + h_s * xv_s) * E_s / (P_T_s * c_**2)))
display(sp.Eq(sp.Symbol('P_T'), P_T_s))
display(sp.Eq(sp.Symbol('P_x'), m_ * xp_s / Q_s))
display(sp.Eq(sp.Symbol('P_y'), m_ * yp_s / Q_s))
display(sp.Eq(P_T_s, m_ * (1 + h_s * xv_s) / Q_s))

# %% [markdown]
# For the $x$ equation, the derivatives with $t'$ treated as independent are
#
# $$
# \frac{\partial L_s}{\partial x'} = \gamma m v_x + qA_x = p_x , \qquad
# \frac{\partial L_s}{\partial x} = \gamma m h\,v_T
#   + q\big[hA_s + (1+hx)\,\partial_x A_s + x'\,\partial_x A_x + y'\,\partial_x A_y\big] .
# $$
#
# The $y$ equation is treated in the same way. With $t'$ independent,
#
# $$
# \frac{\partial L_s}{\partial y'} = \gamma m v_y + qA_y = p_y , \qquad
# \frac{\partial L_s}{\partial y} = q\big[(1+hx)\,\partial_y A_s + x'\,\partial_y A_x + y'\,\partial_y A_y\big] .
# $$
#
# There is no curvature term in the $y$ equation, because the metric factor $1+hx$ does not depend on $y$.

# %%
# Partial derivatives of L_s for general potentials A_x, A_y, A_s (m, c, q kept symbolic; t not explicit).
# The symbols are named so that sympy renders them as in the memo: x', y', t' and A_x, A_y, A_s.
m_p, c_p, q_p = sp.symbols('m c q', positive=True)
tp_p = sp.Symbol("t'", real=True)
xp_p, yp_p = sp.Symbol("x'", real=True), sp.Symbol("y'", real=True)
xv_p, yv_p = sp.Symbol('x', real=True), sp.Symbol('y', real=True)
h_p = sp.Symbol('h', real=True)
Ax_p, Ay_p, As_p = (sp.Function(name)(xv_p, yv_p) for name in ('A_x', 'A_y', 'A_s'))
Q_p = sp.sqrt(tp_p**2 - ((1 + h_p * xv_p) ** 2 + xp_p**2 + yp_p**2) / c_p**2)
L_p = -m_p * c_p**2 * Q_p + q_p * ((1 + h_p * xv_p) * As_p + xp_p * Ax_p + yp_p * Ay_p)

dL_dxp = sp.diff(L_p, xp_p)
dL_dyp_p = sp.diff(L_p, yp_p)
dL_dx_p = sp.diff(L_p, xv_p)
dL_dy_p = sp.diff(L_p, yv_p)
display(sp.Eq(sp.Symbol(r"\frac{\partial L_s}{\partial x'}"), dL_dxp))
display(sp.Eq(sp.Symbol(r"\frac{\partial L_s}{\partial x}"), dL_dx_p))
display(sp.Eq(sp.Symbol(r"\frac{\partial L_s}{\partial y'}"), dL_dyp_p))
display(sp.Eq(sp.Symbol(r"\frac{\partial L_s}{\partial y}"), dL_dy_p))

expected_xp = m_p * xp_p / Q_p + q_p * Ax_p
expected_yp = m_p * yp_p / Q_p + q_p * Ay_p
expected_x = m_p * h_p * (1 + h_p * xv_p) / Q_p + q_p * (
    h_p * As_p + (1 + h_p * xv_p) * sp.diff(As_p, xv_p)
    + xp_p * sp.diff(Ax_p, xv_p) + yp_p * sp.diff(Ay_p, xv_p))
expected_y = q_p * ((1 + h_p * xv_p) * sp.diff(As_p, yv_p)
                    + xp_p * sp.diff(Ax_p, yv_p) + yp_p * sp.diff(Ay_p, yv_p))
assert sp.simplify(dL_dxp - expected_xp) == 0
assert sp.simplify(dL_dyp_p - expected_yp) == 0
assert sp.simplify(dL_dx_p - expected_x) == 0
assert sp.simplify(dL_dy_p - expected_y) == 0
print("the x and y partial derivatives of L_s agree with the memo")

# %% [markdown]
# ## 6. Reduction to the Lorentz force equation
#
# Substituting the partial derivatives into the Euler-Lagrange equations gives
#
# $$
# \frac{d}{ds}\big(\gamma m v_x + qA_x\big)
#   = \gamma m h\,v_T + q\big[hA_s + (1+hx)\,\partial_x A_s + x'\,\partial_x A_x + y'\,\partial_x A_y\big] ,
# $$
#
# $$
# \frac{d}{ds}\big(\gamma m v_y + qA_y\big)
#   = q\big[(1+hx)\,\partial_y A_s + x'\,\partial_y A_x + y'\,\partial_y A_y\big] .
# $$
#
# These are the same equations as the Lorentz force equation, written with $s$ as the independent variable. Convert
# $d/ds$ to $d/dt$ with $d/ds = t'\,d/dt$ and use $x' = v_x t'$ and $y' = v_y t'$. The total time derivatives of the
# potentials along the trajectory are
#
# $$
# \begin{aligned}
# \frac{dA_x}{dt} &= \frac{\partial_s A_x}{t'} + v_x\,\partial_x A_x + v_y\,\partial_y A_x , \\
# \frac{dA_y}{dt} &= \frac{\partial_s A_y}{t'} + v_x\,\partial_x A_y + v_y\,\partial_y A_y .
# \end{aligned}
# $$
#
# The terms with $\partial_x A_x$ and $\partial_y A_y$ cancel between the two sides. The mixed terms do not cancel.
# In the $x$ equation, $dA_x/dt$ contributes $-q\,v_y\,\partial_y A_x$ and $\partial L_s/\partial x$ contributes
# $+q\,v_y\,\partial_x A_y$, so the pair enters as $q\,v_y(\partial_x A_y - \partial_y A_x) = -q\,v_y B_s$. In the
# $y$ equation the same pair enters with the opposite sign. Hence
#
# $$
# \begin{aligned}
# \frac{d}{dt}\big(\gamma m v_x\big) &= \frac{\gamma m h\,v_T}{t'}
#   + \frac{q}{t'}\big[hA_s + (1+hx)\,\partial_x A_s - \partial_s A_x\big] - q\,v_y B_s , \\
# \frac{d}{dt}\big(\gamma m v_y\big) &= \frac{q}{t'}\big[(1+hx)\,\partial_y A_s - \partial_s A_y\big]
#   + q\,v_x B_s .
# \end{aligned}
# $$
#
# The field components in the local frame are
#
# $$
# \begin{aligned}
# B_s &= -\big(\partial_x A_y - \partial_y A_x\big), \\
# B_x &= -\frac{(1+hx)\,\partial_y A_s - \partial_s A_y}{1+hx}, \\
# B_y &= -\frac{\partial_s A_x - \partial_x\big[(1+hx)A_s\big]}{1+hx} ,
# \end{aligned}
# $$
#
# so the brackets are $(1+hx)B_y$ and $-(1+hx)B_x$. With $(1+hx)/t' = v_T$,
#
# $$
# \begin{aligned}
# \frac{d}{dt}\big(\gamma m v_x\big) &= \frac{\gamma m h\,v_T}{t'} + q\,v_T B_y - q\,v_y B_s , \\
# \frac{d}{dt}\big(\gamma m v_y\big) &= -q\,v_T B_x + q\,v_x B_s .
# \end{aligned}
# $$
#
# The first term on the right of the $x$ equation is the rotation of the frame. The triad
# $(\mathbf{T},\mathbf{e}_x,\mathbf{e}_y)$ is left-handed, $\mathbf{T}\times\mathbf{e}_x = -\mathbf{e}_y$, and it
# rotates with $d\mathbf{T}/dt = -h\,(ds/dt)\,\mathbf{e}_x$ and $d\mathbf{e}_x/dt = h\,(ds/dt)\,\mathbf{T}$. For
# $\mathbf{p} = p_T\,\mathbf{T} + p_x\,\mathbf{e}_x + p_y\,\mathbf{e}_y$ with $p_T = \gamma m v_T$,
#
# $$
# \frac{d\mathbf{p}}{dt} = \Big[\dot{p}_T + \frac{h}{t'}\,p_x\Big]\mathbf{T}
#   + \Big[\dot{p}_x - \frac{h}{t'}\,p_T\Big]\mathbf{e}_x + \dot{p}_y\,\mathbf{e}_y ,
# $$
#
# so the $x$ component of $d\mathbf{p}/dt$ is $d(\gamma m v_x)/dt - h\,\gamma m v_T/t'$. Hence
#
# $$
# \Big(\frac{d\mathbf{p}}{dt}\Big)_x = q\big(v_T B_y - v_y B_s\big) , \qquad
# \Big(\frac{d\mathbf{p}}{dt}\Big)_y = q\big(v_x B_s - v_T B_x\big) .
# $$
#
# In the left-handed triad the cross product has the components
#
# $$
# \begin{aligned}
# (\mathbf{v}\times\mathbf{B})_s &= v_y B_x - v_x B_y , \\
# (\mathbf{v}\times\mathbf{B})_x &= v_T B_y - v_y B_s , \\
# (\mathbf{v}\times\mathbf{B})_y &= v_x B_s - v_T B_x ,
# \end{aligned}
# $$
#
# so these are exactly the transverse components of $d\mathbf{p}/dt = q\,\mathbf{v}\times\mathbf{B}$ for any field.
# The tangential component is not independent: for a static magnetic field $|\mathbf{p}|$ is constant, so
# $\mathbf{p}\cdot d\mathbf{p}/dt = 0$. Projecting onto $\mathbf{T}$ and inserting the transverse components above
# gives
#
# $$
# \begin{aligned}
# \Big(\frac{d\mathbf{p}}{dt}\Big)_T
#   &= -\frac{p_x}{p_T}\Big(\frac{d\mathbf{p}}{dt}\Big)_x
#      -\frac{p_y}{p_T}\Big(\frac{d\mathbf{p}}{dt}\Big)_y \\
#   &= -q\,\frac{v_x\big(v_T B_y - v_y B_s\big) + v_y\big(v_x B_s - v_T B_x\big)}{v_T} \\
#   &= q\big(v_y B_x - v_x B_y\big) = q\,(\mathbf{v}\times\mathbf{B})_s ,
# \end{aligned}
# $$
#
# where the terms in $B_s$ cancel. All three components of the Lorentz force equation are therefore reproduced.

# %%
# Chain rule check with an explicit potential: dA_x/dt must contain d_y A_x, not d_x A_y.
s_c, x_c, y_c = sp.symbols('s x y', real=True)
Ax_expr = sp.Function('A_x')(s_c, x_c, y_c)
Ay_expr = sp.Function('A_y')(s_c, x_c, y_c)
s_of_t, x_of_t, y_of_t = (sp.Function(name)(sp.Symbol('t', real=True)) for name in ('s', 'x', 'y'))
Ax_on_traj = Ax_expr.subs({s_c: s_of_t, x_c: x_of_t, y_c: y_of_t})
dAx_dt_exact = sp.diff(Ax_on_traj, sp.Symbol('t', real=True))
dAx_dt_chain = (sp.diff(Ax_expr, s_c) * sp.diff(s_of_t, sp.Symbol('t', real=True))
                + sp.diff(Ax_expr, x_c) * sp.diff(x_of_t, sp.Symbol('t', real=True))
                + sp.diff(Ax_expr, y_c) * sp.diff(y_of_t, sp.Symbol('t', real=True)))
dAx_dt_chain = dAx_dt_chain.subs({s_c: s_of_t, x_c: x_of_t, y_c: y_of_t})
display(sp.Eq(sp.Symbol(r'\frac{dA_x}{dt}'), dAx_dt_chain))
assert sp.simplify(dAx_dt_exact - dAx_dt_chain) == 0
print("dA_x/dt = d_s A_x s-dot + d_x A_x x-dot + d_y A_x y-dot: the y term is d_y A_x")

# %%
# Reduction to the Lorentz force equation. The potentials are kept general; their partial derivatives on the
# trajectory are independent symbols, and the kinematic relations v_T t' = 1+hx, v_x t' = x', v_y t' = y' are
# substituted at the end.
As_v, As_s_v, As_x_v, As_y_v = sp.symbols(
    r'A_s \partial_{s}A_{s} \partial_{x}A_{s} \partial_{y}A_{s}', real=True)
Ax_v, Ax_s_v, Ax_x_v, Ax_y_v = sp.symbols(
    r'A_x \partial_{s}A_{x} \partial_{x}A_{x} \partial_{y}A_{x}', real=True)
Ay_v, Ay_s_v, Ay_x_v, Ay_y_v = sp.symbols(
    r'A_y \partial_{s}A_{y} \partial_{x}A_{y} \partial_{y}A_{y}', real=True)
h_v, q_v, m_v, gamma_v = sp.symbols(r'h q m \gamma', real=True)
hx_v = sp.Symbol(r'h\,x', real=True)
tp_v, xp_v, yp_v = sp.symbols(r"t' x' y'", real=True)
vT_v, vx_v, vy_v = sp.symbols(r'v_T v_x v_y', real=True)

# Field components in the left-handed triad (T, e_x, e_y).
Bs_v = -(Ay_x_v - Ax_y_v)
Bx_v = -((1 + hx_v) * As_y_v - Ay_s_v) / (1 + hx_v)
By_v = -(Ax_s_v - (h_v * As_v + (1 + hx_v) * As_x_v)) / (1 + hx_v)
display(sp.Eq(sp.Symbol('B_s'), Bs_v))
display(sp.Eq(sp.Symbol('B_x'), Bx_v))
display(sp.Eq(sp.Symbol('B_y'), By_v))

# Euler-Lagrange equations, with x' = v_x t' and y' = v_y t'.
dL_dx_v = gamma_v * m_v * h_v * vT_v + q_v * (h_v * As_v + (1 + hx_v) * As_x_v
                                              + vx_v * tp_v * Ax_x_v + vy_v * tp_v * Ay_x_v)
dL_dy_v = q_v * ((1 + hx_v) * As_y_v + vx_v * tp_v * Ax_y_v + vy_v * tp_v * Ay_y_v)

# Total time derivatives of the potentials along the trajectory (chain rule).
# Ax_y_v is d_y A_x and Ay_x_v is d_x A_y: the mixed partials are what survive as B_s.
dAx_dt_v = Ax_s_v / tp_v + vx_v * Ax_x_v + vy_v * Ax_y_v
dAy_dt_v = Ay_s_v / tp_v + vx_v * Ay_x_v + vy_v * Ay_y_v

pdot_x_v = dL_dx_v / tp_v - q_v * dAx_dt_v
pdot_y_v = dL_dy_v / tp_v - q_v * dAy_dt_v

# Frame rotation: (dp/dt)_x = pdot_x - h (ds/dt) p_T, with p_T = gamma m v_T and ds/dt = 1/t'.
pdot_cart_x_v = pdot_x_v - h_v * (1 / tp_v) * gamma_v * m_v * vT_v

kin_v = {vT_v: (1 + hx_v) / tp_v, vx_v: xp_v / tp_v, vy_v: yp_v / tp_v}
res_x_v = sp.simplify((pdot_cart_x_v - q_v * (vT_v * By_v - vy_v * Bs_v)).subs(kin_v))
res_y_v = sp.simplify((pdot_y_v - q_v * (vx_v * Bs_v - vT_v * Bx_v)).subs(kin_v))
display(sp.Eq(sp.Symbol(r'\Big(\frac{d\mathbf{p}}{dt}\Big)_x'), q_v * (vT_v * By_v - vy_v * Bs_v)))
display(sp.Eq(sp.Symbol(r'\Big(\frac{d\mathbf{p}}{dt}\Big)_y'), q_v * (vx_v * Bs_v - vT_v * Bx_v)))
print("residual of the x equation:", res_x_v)
print("residual of the y equation:", res_y_v)
assert res_x_v == 0
assert res_y_v == 0
print("the Euler-Lagrange equations reduce to the transverse Lorentz force for any field")

# %%
# The tangential component follows from p . dp/dt = 0 and must reproduce q (v x B)_s.
pT_v = gamma_v * m_v * vT_v
px_v = gamma_v * m_v * vx_v
py_v = gamma_v * m_v * vy_v
dpdt_x_v = q_v * (vT_v * By_v - vy_v * Bs_v)
dpdt_y_v = q_v * (vx_v * Bs_v - vT_v * Bx_v)
# p . dp/dt = 0 gives (dp/dt)_T = -(p_x (dp/dt)_x + p_y (dp/dt)_y) / p_T
dpdt_T_v = sp.simplify(-(px_v * dpdt_x_v + py_v * dpdt_y_v) / pT_v)
vxB_s_v = vy_v * Bx_v - vx_v * By_v
display(sp.Eq(sp.Symbol(r"\Big(\frac{d\mathbf{p}}{dt}\Big)_T"), dpdt_T_v))
print("(dp/dt)_T - q (v x B)_s:", sp.simplify(dpdt_T_v - q_v * vxB_s_v))
assert sp.simplify(dpdt_T_v - q_v * vxB_s_v) == 0
print("the tangential component is q (v x B)_s: the B_s terms cancel")

# %% [markdown]
# ## 7. Gauge choice for dipole and quadrupole fields
#
# The vector potential is defined only up to $\mathbf{A}\to\mathbf{A}+\nabla\chi$. In the local frame the gradient of a
# scalar has the physical components
#
# $$
# \nabla\chi = \frac{\partial_s\chi}{1+hx}\,\mathbf{T} + \partial_x\chi\,\mathbf{e}_x + \partial_y\chi\,\mathbf{e}_y ,
# $$
#
# so the components change as
#
# $$
# A_s \to A_s + \frac{\partial_s\chi}{1+hx}, \qquad
# A_x \to A_x + \partial_x\chi, \qquad
# A_y \to A_y + \partial_y\chi .
# $$
#
# The metric factor $1+hx$ appears only in the tangential component. Substituting this into $L_s$ changes the
# Lagrangian by
#
# $$
# \Delta L_s = q\Big[(1+hx)\,\frac{\partial_s\chi}{1+hx} + x'\,\partial_x\chi + y'\,\partial_y\chi\Big]
#            = q\,\frac{d\chi}{ds} ,
# $$
#
# a total derivative with respect to $s$, so the Euler-Lagrange equations are unchanged.
#
# Setting $A_x = A_y = 0$ requires $\partial_x\chi = -A_x$ and $\partial_y\chi = -A_y$, that is, $\chi$ must be the
# line integral of $-(A_x, A_y)$ in the transverse plane at fixed $s$. Such a $\chi$ exists if and only if the
# transverse part of $\mathbf{A}$ is a gradient,
#
# $$
# \partial_x A_y - \partial_y A_x = 0 .
# $$
#
# In this frame $B_s = -(\partial_x A_y - \partial_y A_x)$, so the gauge $A_x = A_y = 0$ is available exactly when the
# field has no longitudinal component. A dipole has $B_y$ only and a quadrupole has $B_x \propto y$ and
# $B_y \propto x$, so $B_s = 0$ for both, and the gauge can be taken. A solenoid breaks the condition.
#
# After $A_x = A_y = 0$ is imposed, the remaining freedom is $\chi = \chi(s)$, which shifts only the tangential
# component,
#
# $$
# A_s \to A_s + \frac{\chi'(s)}{1+hx} .
# $$
#
# This is not a physical change: $B_y \propto \partial_x[(1+hx)A_s]/(1+hx)$ is invariant, and the shift of $L_s$ is
# again the total derivative $q\chi'(s)$. The function $\chi'(s)$ is therefore fixed by a convention, not by the
# equations of motion.
#
# The convention $A_s = 0$ on the reference orbit fixes $\chi'(s)$ completely and determines $A_s$ for the dipole and
# the quadrupole. With $A_x = A_y = 0$ the vertical field is $B_y = \partial_x[(1+hx)A_s]/(1+hx)$. An ideal sector
# dipole has the uniform field $B_y = B_0 = P_0 h/q$, so
#
# $$
# \partial_x\big[(1+hx)A_s\big] = B_0\,(1+hx)
# \quad\Longrightarrow\quad
# (1+hx)A_s = B_0\Big(x + \tfrac{1}{2} h x^2\Big) + C(s) .
# $$
#
# The convention $A_s = 0$ at $x = 0$ sets $C(s) = 0$. A normal quadrupole with normalized gradient
# $k(s) = qg/P_0$ (focusing in $x$ for $k > 0$) and a skew quadrupole with normalized skew gradient
# $k_s(s) = qg_s/P_0$ add the transverse fields $B_x \propto ky + k_s x$ and $B_y \propto kx - k_s y$, that is, the
# potential terms $\tfrac{1}{2}k\,(x^2 - y^2)$ and $k_s\,xy$. With the sign convention of this memo, the potential of
# a dipole, a normal quadrupole, and a skew quadrupole is
#
# $$
# a_s = \frac{qA_s}{P_0}
#     = -\frac{h\big(x + \tfrac{1}{2} h x^2\big) + \tfrac{1}{2} k\,(x^2 - y^2) + k_s\,xy}{1+hx} ,
# $$
#
# so that the term entering the equations of motion is exactly
#
# $$
# -(1+hx)\,a_s = h\big(x + \tfrac{1}{2} h x^2\big) + \tfrac{1}{2} k\,(x^2 - y^2) + k_s\,xy .
# $$
#
# The overall sign is a convention. It is fixed by requiring the reference orbit $x = y = 0$ to be an equilibrium,
# that is, by the vanishing of the transverse force there. Curvature corrections to the quadrupole terms are higher
# order and are not kept.

# %%
# Gauge transformation in the curvilinear frame: A -> A + grad(chi).
chi_f = sp.Function('chi')(s_f, x_f, y_f)
jac_c = sp.Matrix.hstack(r_v.diff(s_f), r_v.diff(x_f), r_v.diff(y_f))
jac_c_inv = sp.simplify(jac_c.inv())


def d_cart(expr, axis):
    """Derivative with respect to Cartesian axis 0, 1, 2 at fixed (s, x, y)."""
    return sum(jac_c_inv[q, axis] * sp.diff(expr, [s_f, x_f, y_f][q]) for q in range(3))


grad_chi_cart = sp.Matrix([d_cart(chi_f, i) for i in range(3)])
grad_chi_local = sp.Matrix([sp.simplify(grad_chi_cart.dot(v)) for v in (T_v, ex_v, ey_v)])
# Components of grad(chi) along T, e_x, e_y.
display(sp.Eq(sp.Symbol(r"(\nabla\chi)_s"), grad_chi_local[0]))
display(sp.Eq(sp.Symbol(r"(\nabla\chi)_x"), grad_chi_local[1]))
display(sp.Eq(sp.Symbol(r"(\nabla\chi)_y"), grad_chi_local[2]))
assert sp.simplify(grad_chi_local[0] - sp.diff(chi_f, s_f) / (1 + h_f * x_f)) == 0
assert sp.simplify(grad_chi_local[1] - sp.diff(chi_f, x_f)) == 0
assert sp.simplify(grad_chi_local[2] - sp.diff(chi_f, y_f)) == 0
print("grad(chi) = (d_s chi / (1+hx)) T + d_x chi e_x + d_y chi e_y")

# %%
# Component shifts and the change of L_s. The metric factor cancels in the tangential term, leaving a total derivative.
As_g, Ax_g, Ay_g = sp.symbols('A_s A_x A_y', real=True)
chi_s, chi_x, chi_y = sp.symbols(r'\partial_{s}\chi \partial_{x}\chi \partial_{y}\chi', real=True)
display(sp.Eq(sp.Symbol(r"A_s^{\prime}"), As_g + chi_s / (1 + h_f * x_f)))
display(sp.Eq(sp.Symbol(r"A_x^{\prime}"), Ax_g + chi_x))
display(sp.Eq(sp.Symbol(r"A_y^{\prime}"), Ay_g + chi_y))

q_g, xp_g, yp_g = sp.symbols(r"q x' y'", real=True)
delta_L = q_g * ((1 + h_f * x_f) * (chi_s / (1 + h_f * x_f)) + xp_g * chi_x + yp_g * chi_y)
display(sp.Eq(sp.Symbol(r"\Delta L_s"), delta_L))
assert sp.simplify(delta_L - q_g * (chi_s + xp_g * chi_x + yp_g * chi_y)) == 0
print("Delta L_s = q dchi/ds is a total derivative")

# %%
# Longitudinal field in the curvilinear frame, and the integrability condition for A_x = A_y = 0.
As_f, Ax_f, Ay_f = (sp.Function(name)(s_f, x_f, y_f) for name in ('A_s', 'A_x', 'A_y'))
A_cart = As_f * T_v + Ax_f * ex_v + Ay_f * ey_v
B_cart = sp.Matrix([
    d_cart(A_cart[2], 1) - d_cart(A_cart[1], 2),
    d_cart(A_cart[0], 2) - d_cart(A_cart[2], 0),
    d_cart(A_cart[1], 0) - d_cart(A_cart[0], 1),
])
B_local = sp.Matrix([sp.simplify(B_cart.dot(v)) for v in (T_v, ex_v, ey_v)])
display(sp.Eq(sp.Symbol('B_s'), -(sp.diff(Ay_f, x_f) - sp.diff(Ax_f, y_f))))
assert sp.simplify(B_local[0] - (sp.diff(Ax_f, y_f) - sp.diff(Ay_f, x_f))) == 0
print("B_s = -(d_x A_y - d_y A_x): the gauge A_x = A_y = 0 needs B_s = 0")

# A solenoid-like transverse potential has B_s != 0, so A_x and A_y cannot both be removed.
Bs_sym = sp.Symbol('B_s', real=True)
A_sol = (Bs_sym * y_f / 2) * ex_v + (-Bs_sym * x_f / 2) * ey_v
B_sol = sp.Matrix([
    d_cart(A_sol[2], 1) - d_cart(A_sol[1], 2),
    d_cart(A_sol[0], 2) - d_cart(A_sol[2], 0),
    d_cart(A_sol[1], 0) - d_cart(A_sol[0], 1),
])
B_sol_local = sp.Matrix([sp.simplify(B_sol.dot(v)) for v in (T_v, ex_v, ey_v)])
display(sp.Eq(sp.Symbol(r"B_s^{\mathrm{solenoid}}"), B_sol_local[0]))
assert sp.simplify(B_sol_local[0] - Bs_sym) == 0
assert sp.simplify(sp.diff(A_sol.dot(ey_v), x_f) - sp.diff(A_sol.dot(ex_v), y_f) + Bs_sym) == 0
print("a solenoid has B_s != 0, so the integrability condition fails")

# %%
# Residual gauge freedom after A_x = A_y = 0: chi = chi(s) shifts only A_s, and L_s by the total derivative q chi'(s).
chi_of_s = sp.Function('chi')(s_f)
display(sp.Eq(sp.Symbol(r"A_s^{\prime}"), As_g + sp.diff(chi_of_s, s_f) / (1 + h_f * x_f)))
residual_L = sp.simplify((1 + h_f * x_f) * (sp.diff(chi_of_s, s_f) / (1 + h_f * x_f)))
assert sp.simplify(residual_L - sp.diff(chi_of_s, s_f)) == 0
print("residual gauge: A_s -> A_s + chi'(s)/(1+hx), and L_s shifts by q chi'(s)")

# %% [markdown]
# ## 8. Linearized Euler-Lagrange equations
#
# Take $m = c = 1$ and the gauge $A_x = A_y = 0$ of Section 7, with the convention $A_s = 0$ on the reference orbit.
# The Lagrangian $L_s$ then reduces to
#
# $$
# L_s = -\sqrt{t'^2 - (1+hx)^2 - x'^2 - y'^2} + q\,(1+hx)\,A_s .
# $$
#
# With $Q = \sqrt{t'^2 - (1+hx)^2 - x'^2 - y'^2}$, the partial derivatives of $L_s$ are
#
# $$
# \begin{aligned}
# \frac{\partial L_s}{\partial x'} &= \frac{x'}{Q} , &
# \frac{\partial L_s}{\partial y'} &= \frac{y'}{Q} , \\
# \frac{\partial L_s}{\partial x} &= \frac{h(1+hx)}{Q} + q\big[hA_s + (1+hx)\,\partial_x A_s\big] , &
# \frac{\partial L_s}{\partial y} &= q\,(1+hx)\,\partial_y A_s .
# \end{aligned}
# $$
#
# The $t$ equation gives the constant of motion $\partial L_s/\partial t' = -E$, that is, $t'/Q = E$. Together with
# $Q^2 = t'^2 - S$, where $S = (1+hx)^2 + x'^2 + y'^2$, this fixes $t' = E\sqrt{S/(E^2-1)}$ and
# $Q = \sqrt{S}/\sqrt{E^2-1}$. Substituting these into the Euler-Lagrange equations for $L_s$ gives second-order
# equations for $x(s)$ and $y(s)$.
#
# Linearize in $x$, $x'$, $y$, $y'$, and $\delta$, with $\sqrt{E^2-1} = P_0(1+\delta)$ and $qA_s = P_0 a_s$. Since
# $x'$ and $y'$ are first order, $S = 1 + 2hx + \mathcal{O}(2)$ and $\sqrt{S} = 1 + hx + \mathcal{O}(2)$, so the
# left-hand sides become $P_0 x'' + \mathcal{O}(2)$ and $P_0 y'' + \mathcal{O}(2)$. On the right-hand sides the
# dipole and quadrupole potential gives
#
# $$
# a_s = -hx + \tfrac{1}{2} h^2 x^2 - \tfrac{1}{2} k\,(x^2 - y^2) - k_s\,xy + \mathcal{O}(3) ,
# $$
#
# with $\partial_x a_s = -h + h^2 x - kx - k_s y + \mathcal{O}(2)$ and
# $\partial_y a_s = ky - k_s x + \mathcal{O}(2)$. Collecting the first-order terms must reproduce
# $x'' + (h^2+k)x + k_s y = h\delta$ and $y'' - ky + k_s x = 0$.

# %%
h, k, k_s = sp.symbols('h k k_s', real=True)
tp, xp, yp, xv, yv = sp.symbols(r"t' x' y' x y", real=True)
E_sym, delta_sym = sp.symbols('E delta', real=True)
P0_sym = sp.Symbol('P0', positive=True)
S_expr = (1 + h * xv) ** 2 + xp**2 + yp**2
F_expr = P0_sym * (-h * (xv + h * xv**2 / 2) / (1 + h * xv) - k * (xv**2 - yv**2) / 2 - k_s * xv * yv)
L_fn = -sp.sqrt(tp**2 - S_expr) + (1 + h * xv) * F_expr

# Partial derivatives of L_s for general potentials A_x, A_y, A_s (m, c, q kept symbolic; t not explicit).
m_s, c_s, q_s = sp.symbols('m c q', positive=True)
Ax_f, Ay_f, As_f = (sp.Function(name)(xv, yv) for name in ('Ax', 'Ay', 'As'))
Q_gen = sp.sqrt(tp**2 - ((1 + h * xv) ** 2 + xp**2 + yp**2) / c_s**2)
L_gen = -m_s * c_s**2 * Q_gen + q_s * ((1 + h * xv) * As_f + xp * Ax_f + yp * Ay_f)
expected_partials = {
    tp: -m_s * c_s**2 * tp / Q_gen,
    xp: m_s * xp / Q_gen + q_s * Ax_f,
    yp: m_s * yp / Q_gen + q_s * Ay_f,
    xv: m_s * h * (1 + h * xv) / Q_gen
    + q_s * (h * As_f + (1 + h * xv) * sp.diff(As_f, xv) + xp * sp.diff(Ax_f, xv) + yp * sp.diff(Ay_f, xv)),
    yv: q_s * ((1 + h * xv) * sp.diff(As_f, yv) + xp * sp.diff(Ax_f, yv) + yp * sp.diff(Ay_f, yv)),
}
for var, expected in expected_partials.items():
    assert sp.simplify(sp.diff(L_gen, var) - expected) == 0, var
print("partial derivatives of L_s verified for general A_x, A_y, A_s")

# t equation: dL/dt' = -E is a constant of motion. Its positive root is t' = E sqrt(S/(E^2-1)).
tp_expected = E_sym * sp.sqrt(S_expr / (E_sym**2 - 1))
t_residual = sp.diff(L_fn, tp).subs(tp, tp_expected) + E_sym
for values in [{E_sym: 2.5, h: 0.3, xv: 0.1, xp: 0.2, yp: -0.1}, {E_sym: 4.0, h: -0.2, xv: -0.3, xp: 0.1, yp: 0.4}]:
    assert abs(complex(sp.N(t_residual.subs(values)))) < 1e-12
print("t' = E sqrt(S/(E^2-1)) satisfies dL/dt' = -E")
# Lorentz factor with s as the independent variable: gamma = t'/Q, with Q = sqrt(t'^2 - S). It equals E.
Q_expr = sp.sqrt(tp**2 - S_expr)
gamma_s = (tp / Q_expr).subs(tp, tp_expected)
for values in [{E_sym: 2.5, h: 0.3, xv: 0.1, xp: 0.2, yp: -0.1}, {E_sym: 4.0, h: -0.2, xv: -0.3, xp: 0.1, yp: 0.4}]:
    assert abs(complex(sp.N((gamma_s - E_sym).subs(values)))) < 1e-12
print("gamma = t'/Q = E on the t' solution")

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
print("zeroth order of x equation:")
display(order0_x)
print("first order of x equation:")
display(order1_x)
print("first order of y equation:")
display(order1_y)
assert order0_x == 0
assert sp.simplify(order1_x - P0_sym * (X.diff(s, 2) + (h**2 + k) * X + k_s * Y - h * delta_sym)) == 0
assert sp.simplify(order1_y - P0_sym * (Y.diff(s, 2) - k * Y + k_s * X)) == 0

# %%
# Eq. (14) in natural units m = c = 1: t' = (1 + hx) E / P_T with P_T = sqrt(E^2 - 1 - P_x^2 - P_y^2),
# where P_x = x' sqrt(E^2-1) / sqrt(S) and P_y = y' sqrt(E^2-1) / sqrt(S).
P_perp_sq = (E_sym**2 - 1) * (xp**2 + yp**2) / S_expr
P_T_closed = sp.sqrt(E_sym**2 - 1 - P_perp_sq)
P_x_closed = xp * sp.sqrt(E_sym**2 - 1) / sp.sqrt(S_expr)
t_prime_eq14 = (1 + h * xv) * E_sym / P_T_closed
for values in [{E_sym: 2.5, h: 0.3, xv: 0.1, xp: 0.2, yp: -0.1}, {E_sym: 4.0, h: -0.2, xv: -0.3, xp: 0.1, yp: 0.4}]:
    assert abs(complex(sp.N((t_prime_eq14 - tp_expected).subs(values)))) < 1e-12
print("eq. (14) gives t' = E sqrt(S/(E^2-1)) in natural units")

# Q = t'/gamma = sqrt(S/(E^2-1)) in natural units; check P_x = x'/Q, P_T = (1+hx)/Q.
Q_closed = sp.sqrt(S_expr / (E_sym**2 - 1))
for values in [{E_sym: 2.5, h: 0.3, xv: 0.1, xp: 0.2, yp: -0.1}, {E_sym: 4.0, h: -0.2, xv: -0.3, xp: 0.1, yp: 0.4}]:
    assert abs(complex(sp.N((Q_closed - tp_expected / E_sym).subs(values)))) < 1e-12
    assert abs(complex(sp.N((xp / Q_closed - P_x_closed).subs(values)))) < 1e-12
    assert abs(complex(sp.N(((1 + h * xv) / Q_closed - P_T_closed).subs(values)))) < 1e-12
print("P_x = x'/Q, P_y = y'/Q and P_T = (1+hx)/Q in natural units")

# %% [markdown]
# ## 9. Hamiltonian derivation and normalization
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
# The residual gauge freedom $\chi = \chi(s)$ of Section 7 shifts $A_s$ by $\chi'(s)/(1+hx)$ and therefore adds
# $-q\chi'(s)/P_0$ to $H$. This is a function of $s$ only, so it does not change Hamilton's equations.
#
# The next cell checks this Legendre transform numerically at random values.

# %%
m_, c_, q_, h_, xv = sp.symbols('m c q h x', real=True)
X1, Y1, T1 = sp.Symbol("x'", real=True), sp.Symbol("y'", real=True), sp.Symbol("t'", real=True)
As, Ax, Ay = sp.symbols('A_s A_x A_y', real=True)
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

# %%
# Conjugate momenta and the Legendre transform (blocks above), from the L_s model of the cell above.
display(sp.Eq(sp.Symbol('p_t'), p_t_expr))
display(sp.Eq(sp.Symbol('p_x'), p_x_expr))
display(sp.Eq(sp.Symbol('p_y'), p_y_expr))
display(sp.Eq(sp.Symbol('K'), K_closed))
assert sp.simplify(p_t_expr + E_expr) == 0

# %% [markdown]
# ### Normalization: $\delta$, $p_x$, $p_y$ and the normalized Hamiltonian
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
dL, beta0, E0, p0c = sp.symbols(r'\delta_{L} \beta_{0} E_{0} p_{0}c', positive=True)
# With E0 = p0*c^2 / (beta0 * c) = p0c / beta0, and p0c = beta0 * E0 is the standard relation.
E_ref = p0c / beta0
dE = dL * p0c
lhs = ((E_ref + dE) ** 2 - (E_ref**2 - p0c**2)) / p0c**2
rhs = 1 + 2 * dL / beta0 + dL**2
display(sp.simplify(lhs - rhs))
assert sp.simplify(lhs - rhs) == 0

# %%
# Normalization and the normalized Hamiltonian (blocks above), with m = c = 1 and P_0 > 0.
P0_n = sp.Symbol('P0', positive=True)
p_n, Px_n, Py_n, Ax_n, Ay_n, As_n, delta_n = sp.symbols('p P_x P_y A_x A_y A_s delta', real=True)
h_n, x_n, k_n = sp.symbols('h x k', real=True)
px_n, py_n, ax_n, ay_n, a_s_n = sp.symbols('p_x p_y a_x a_y a_s', real=True)
display(sp.Eq(1 + delta_n, p_n / P0_n))
display(sp.Eq(px_n, Px_n / P0_n))
display(sp.Eq(py_n, Py_n / P0_n))
display(sp.Eq(ax_n, Ax_n / P0_n))
display(sp.Eq(ay_n, Ay_n / P0_n))
display(sp.Eq(a_s_n, As_n / P0_n))

H_norm = -(1 + h_n * x_n) * sp.sqrt((1 + delta_n) ** 2 - (px_n - ax_n) ** 2 - (py_n - ay_n) ** 2) - (1 + h_n * x_n) * a_s_n
display(sp.Eq(sp.Symbol('H'), H_norm))

# Check: K / P_0 with E = sqrt(1 + P_0^2 (1+delta)^2) (m = c = 1) reproduces H.
E_of_delta = sp.sqrt(1 + P0_n**2 * (1 + delta_n) ** 2)
K_model = -(1 + h_n * x_n) * (sp.sqrt(E_of_delta**2 - 1 - (P0_n * (px_n - ax_n)) ** 2 - (P0_n * (py_n - ay_n)) ** 2)
                              + P0_n * a_s_n)
values_n = {P0_n: 2.0, delta_n: 0.1, px_n: 0.3, py_n: -0.2, ax_n: 0.05, ay_n: 0.02, a_s_n: 0.4, h_n: 0.3, x_n: 0.2}
assert abs(complex(sp.N((K_model / P0_n - H_norm).subs(values_n)))) < 1e-12
print("K / P_0 reproduces the normalized Hamiltonian")

# Energy deviation (block above).
display(sp.Eq(lhs, rhs))

# %% [markdown]
# ## 10. Full Hamiltonian and exact equations of motion
#
# With the gauge $A_x = A_y = 0$ the potential terms are $-(1+hx)a_s$, and the potential of Section 7 gives them
# exactly. Collecting the terms, with the straight-quadrupole approximation for the quadrupole:
#
# $$
# H(x, p_x, y, p_y, \delta; s) = -(1+hx)\sqrt{(1+\delta)^2 - p_x^2 - p_y^2}
#   + h\big(x + \tfrac12 h x^2\big) + \tfrac12 k\,(x^2 - y^2) + k_s\,xy .
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
x_, y_, px, py, d, h, k, k_s = sp.symbols('x y p_x p_y delta h k k_s', real=True)
H_full = (-(1 + h * x_) * sp.sqrt((1 + d) ** 2 - px**2 - py**2)
          + h * (x_ + h * x_**2 / 2) + k * (x_**2 - y_**2) / 2 + k_s * x_ * y_)
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

# %%
# Full Hamiltonian and Hamilton's equations (blocks above).
display(sp.Eq(sp.Symbol('H'), H_full))
display(sp.Eq(sp.Symbol("x'"), sp.diff(H_full, px)))
display(sp.Eq(sp.Symbol(r"p_{x}'"), -sp.diff(H_full, x_)))
display(sp.Eq(sp.Symbol("y'"), sp.diff(H_full, py)))
display(sp.Eq(sp.Symbol(r"p_{y}'"), -sp.diff(H_full, y_)))
display(sp.Eq(sp.Symbol(r"\delta'"), 0))

# %% [markdown]
# ## 11. Linearized equations of motion
#
# Linearize around the reference orbit. The Jacobian of the exact EOM at the origin gives the linear matrix
# $\mathbf{A}$, with $\mathbf{z}' = \mathbf{A}\,\mathbf{z}$ and $\mathbf{z} = (x, p_x, y, p_y, \delta)^\mathsf{T}$.
# Expected result:
#
# $$
# \mathbf{A} = \begin{pmatrix}
# 0 & 1 & 0 & 0 & 0 \\
# -(h^2 + k) & 0 & -k_s & 0 & h \\
# 0 & 0 & 0 & 1 & 0 \\
# -k_s & 0 & k & 0 & 0 \\
# 0 & 0 & 0 & 0 & 0
# \end{pmatrix}.
# $$
#
# This gives the Hill equations and the inhomogeneous term $h\delta$ from the bend:
#
# $$
# x'' + (h^2 + k)\,x + k_s\,y = h\,\delta, \qquad y'' - k\,y + k_s\,x = 0 .
# $$

# %%
state_zero = {x_: 0, px: 0, y_: 0, py: 0, d: 0}
A_mat = sp.Matrix(rhs_exact).jacobian(state)
A_mat = sp.simplify(A_mat.subs(state_zero))
display(A_mat)
A_expected = sp.Matrix([
    [0, 1, 0, 0, 0],
    [-(h**2 + k), 0, -k_s, 0, h],
    [0, 0, 0, 1, 0],
    [-k_s, 0, k, 0, 0],
    [0, 0, 0, 0, 0],
])
assert sp.simplify(A_mat - A_expected) == sp.zeros(5, 5)

# %%
# Linear system and linear equations (blocks above).
print("A =")
display(A_expected)
Xf, Yf = sp.Function('x')(s), sp.Function('y')(s)
display(sp.Eq(Xf.diff(s, 2) + (h**2 + k) * Xf + k_s * Yf, h * d))
display(sp.Eq(Yf.diff(s, 2) - k * Yf + k_s * Xf, 0))

# %% [markdown]
# ### Checks against known results
#
# - Dispersion: with constant $h$ and $k$, a constant solution $D = h/(h^2 + k)$ satisfies
#   $D'' + (h^2 + k)D = h$. This is the closed-orbit (periodic) dispersion of a uniform bending section.
# - Vertical plane: no $h^2$ term, as expected for a bend in the horizontal plane.
# - Linear coupling: the skew quadrupole term $k_s\,xy$ couples $x$ and $y$ linearly; setting $k_s = 0$ removes it.

# %%
D_const = h / (h**2 + k)
# For constant h and k the second derivative in s of D_const is zero, so the check is algebraic.
assert sp.simplify((h**2 + k) * D_const - h) == 0

# %% [markdown]
# ## 12. Summary
#
# 1. Metric in Frenet-Serret coordinates: $d\ell^2 = (1+hx)^2 ds^2 + dx^2 + dy^2$.
# 2. Hamiltonian with $s$ as the independent variable:
#    $H = -(1+hx)\sqrt{(1+\delta)^2 - (p_x-a_x)^2 - (p_y-a_y)^2} - (1+hx)\,a_s$.
# 3. Ideal dipole: $-(1+hx)\,a_s = h(x + \tfrac12 h x^2)$, so the bend adds the term $h x$ (cancelled by the
#    $-(1+hx)$ term at equilibrium) and the $\tfrac12 h^2 x^2$ term, which produces the $h^2$ focusing.
# 4. The Euler-Lagrange equations for $L_s$ reduce to the Lorentz force equation for $B_s = 0$.
# 5. Gauge: $A_x = A_y = 0$ is available whenever $B_s = 0$, which holds for dipoles and quadrupoles. The residual
#    freedom $\chi = \chi(s)$ only shifts $A_s$ and is fixed by convention.
# 6. Linear EOM: $x'' + (h^2+k)x = h\delta$ and $y'' - ky = 0$ (no RF, so $\delta' = 0$).
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
