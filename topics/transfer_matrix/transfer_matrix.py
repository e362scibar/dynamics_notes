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
# - File: `topics/transfer_matrix/transfer_matrix.py`
# - Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# - SPDX-License-Identifier: GPL-3.0-or-later
# - Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

# %% [markdown]
# # Linear transfer matrices of a drift space, a sector dipole magnet, and a quadrupole magnet
#
# Memo 2026-002
#
# This memo derives the linear transfer matrix of three basic elements (drift space, sector dipole magnet, and
# quadrupole magnet) from the linearized equations of motion (EOM) of the curvilinear-coordinate memo
# [2026-001, Sec. 11]. The reference orbit is planar and there is no x-y coupling ($k_s = 0$).
#
# Scope:
# - Static magnetic fields only (no RF), so $\delta' = 0$.
# - Every element has constant $h$ and $k$ along its length. Fringe fields are ignored (hard-edge model).
# - The sector dipole magnet has end faces perpendicular to the reference orbit, so there is no edge focusing.
# - The phase-space vector is $\mathbf{u} = (x, p_x, y, p_y, z, \delta)^\mathsf{T}$ with the conventions of
#   [2026-001]: $z = s - \beta_0 c\,t$ ($z > 0$ means ahead of the reference particle) and
#   $\delta = (E - E_0)/(\beta_0 c P_0)$. To first order $p_x = x'$ and $p_y = y'$.
#
# References: the results are standard; see [lee2019] and [wiedemann2015]. The page-level content of those references
# was not re-checked while writing this memo. Every result here is verified with sympy and numerically against the
# exact (nonlinear) EOM.

# %%
import numpy as np
import sympy as sp
from IPython.display import display
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Symbols. L is the running length inside an element: M(L) maps the entrance vector to the vector at distance L.
L = sp.Symbol("L", positive=True)
h = sp.Symbol("h", positive=True)
k = sp.Symbol("k", real=True)
g = sp.Symbol("g", positive=True)  # g = sqrt(|k|) for a quadrupole
beta0 = sp.Symbol("beta_0", positive=True)
gamma0 = sp.Symbol("gamma_0", positive=True)
x_, px, y_, py, z_s, d = sp.symbols("x p_x y p_y z delta", real=True)
state = [x_, px, y_, py, z_s, d]
# Entrance values (initial condition of an element).
x0, px0, y0, py0, z0, d0 = sp.symbols("x_0 p_{x0} y_0 p_{y0} z_0 delta_0", real=True)
u0 = sp.Matrix([x0, px0, y0, py0, z0, d0])

# Numerical constants used by the checks below.
BETA0_NUM = 0.9
AMPLITUDE = 1.0e-6  # amplitude of the initial vector for the comparison with the exact EOM
REL_TOL_EXACT = 1.0e-4  # nonlinear corrections are O(AMPLITUDE), so a relative tolerance of 1e-4 is generous
ODE_RTOL = 1.0e-12
ODE_ATOL = 1.0e-15
ENTRANCE_DIRECTION = np.array([1.0, -0.7, 0.5, 0.3, 0.2, 0.8])  # generic direction of the entrance vector

# %% [markdown]
# ## 2. Transfer matrix
#
# The linearized EOM of [2026-001, Sec. 11] with $k_s = 0$ are
#
# $$
# x'' + (h^2 + k)\,x = h\,\delta, \qquad y'' - k\,y = 0, \qquad
# z' = -h\,x + \frac{\delta}{\gamma_0^2}, \qquad \delta' = 0 .
# $$
#
# In matrix form $\mathbf{u}' = \mathbf{A}\,\mathbf{u}$ with
#
# $$
# \mathbf{A} = \begin{pmatrix}
# 0 & 1 & 0 & 0 & 0 & 0 \\
# -(h^2+k) & 0 & 0 & 0 & 0 & h \\
# 0 & 0 & 0 & 1 & 0 & 0 \\
# 0 & 0 & k & 0 & 0 & 0 \\
# -h & 0 & 0 & 0 & 0 & 1/\gamma_0^2 \\
# 0 & 0 & 0 & 0 & 0 & 0
# \end{pmatrix}.
# $$
#
# The matrix is rebuilt here as the Jacobian of the exact EOM at the origin (Hamiltonian of [2026-001, Sec. 10]
# with $k_s = 0$).

# %%
# Exact Hamiltonian with k_s = 0 and the exact EOM [2026-001].
H_full = (-(1 + h * x_) * sp.sqrt(1 + 2 * d + beta0**2 * d**2 - px**2 - py**2) + d
          + h * (x_ + h * x_**2 / 2) + k * (x_**2 - y_**2) / 2)
rhs_exact = [sp.diff(H_full, px), -sp.diff(H_full, x_), sp.diff(H_full, py), -sp.diff(H_full, y_),
             sp.diff(H_full, d), -sp.diff(H_full, z_s)]

state_zero = {x_: 0, px: 0, y_: 0, py: 0, z_s: 0, d: 0}
beta0_of_gamma0 = sp.sqrt(1 - 1 / gamma0**2)  # beta0 expressed with gamma0
A_gen = sp.Matrix(rhs_exact).jacobian(state).subs(state_zero).subs(beta0, beta0_of_gamma0)
A_gen = sp.simplify(A_gen)
A_expected = sp.Matrix([
    [0, 1, 0, 0, 0, 0],
    [-(h**2 + k), 0, 0, 0, 0, h],
    [0, 0, 0, 1, 0, 0],
    [0, 0, k, 0, 0, 0],
    [-h, 0, 0, 0, 0, 1 / gamma0**2],
    [0, 0, 0, 0, 0, 0],
])
display(sp.Eq(sp.Symbol("A"), A_expected))
assert sp.simplify(A_gen - A_expected) == sp.zeros(6, 6)

# %% [markdown]
# The transfer matrix $\mathbf{M}(s)$ of an element maps the entrance vector to the vector at distance $s$:
#
# $$
# \mathbf{u}(s) = \mathbf{M}(s)\,\mathbf{u}(0).
# $$
#
# The equation for $\mathbf{M}$ follows from differentiating this relation with respect to $s$ in two ways. The
# entrance vector $\mathbf{u}(0)$ is constant, so $\mathbf{u}'(s) = \mathbf{M}'(s)\,\mathbf{u}(0)$. On the other hand,
# the EOM give $\mathbf{u}'(s) = \mathbf{A}\,\mathbf{u}(s) = \mathbf{A}\,\mathbf{M}(s)\,\mathbf{u}(0)$. Equating,
#
# $$
# \big(\mathbf{M}'(s) - \mathbf{A}\,\mathbf{M}(s)\big)\,\mathbf{u}(0) = 0 .
# $$
#
# This holds for every $\mathbf{u}(0)$, so the matrix in the parentheses is zero. At $s = 0$ the first relation gives
# $\mathbf{M}(0) = \mathbf{I}$. Hence
#
# $$
# \mathbf{M}' = \mathbf{A}\,\mathbf{M}, \qquad \mathbf{M}(0) = \mathbf{I}.
# $$
#
# This initial-value problem has a unique solution. For constant $\mathbf{A}$ (constant $h$ and $k$) it is the matrix
# exponential
#
# $$
# \mathbf{M}(s) = \exp(\mathbf{A}\,s).
# $$
#
# The matrix of a chain of elements is the product in reverse order, $\mathbf{M}_\mathrm{tot} = \mathbf{M}_n \cdots
# \mathbf{M}_2 \mathbf{M}_1$ with the first element on the right.
#
# Because $\delta' = 0$ and $z$ does not feed back into the other variables, $\mathbf{M}$ is block triangular: the
# vertical plane decouples, and $z$ is obtained by integrating $-h\,x + \delta/\gamma_0^2$ after $x$ is known.

# %%
M_sym, A_sym = sp.Symbol("M"), sp.Symbol("A")
u_s, u_0 = sp.Symbol("u(s)"), sp.Symbol("u(0)")
display(sp.Eq(u_s, M_sym * u_0))
display(sp.Eq(sp.Symbol("u'(s)"), sp.Symbol("M'") * u_0))
display(sp.Eq(sp.Symbol("u'(s)"), A_sym * M_sym * u_0))
display(sp.Eq((sp.Symbol("M'") - A_sym * M_sym) * u_0, 0))
display(sp.Eq(sp.Symbol("M'"), A_sym * M_sym))
display(sp.Eq(M_sym, sp.exp(A_sym * sp.Symbol("s"))))

# %% [markdown]
# ## 3. Symplectic property
#
# Expand the exact Hamiltonian of [2026-001, Sec. 10] with $k_s = 0$ in the deviations $(x, p_x, y, p_y, \delta)$:
# $H = H_0 + H_1 + H_2 + \cdots$, where $H_n$ is homogeneous of degree $n$ (the expansion is carried to third order in
# [2026-003, Sec. 2]). The first terms are
#
# $$
# H_0 = -1, \qquad H_1 = 0, \qquad
# H_2 = \frac{p_x^2 + p_y^2}{2} + \frac{(h^2 + k)\,x^2}{2} - \frac{k\,y^2}{2} - h\,x\,\delta
#       + \frac{\delta^2}{2\gamma_0^2}.
# $$
#
# The constant $H_0$ does not enter the EOM, and $H_1 = 0$ because the reference orbit is an equilibrium. The
# quadratic part $H_2$ gives the linearized EOM, which are therefore Hamiltonian, $\mathbf{u}' = \mathbf{J}\,\nabla H_2$,
# so $\mathbf{A} = \mathbf{J}\,\mathbf{W}$ with the symmetric Hessian $\mathbf{W}$ of $H_2$ and
# $\mathbf{J} = \mathrm{diag}(\mathbf{J}_2, \mathbf{J}_2, \mathbf{J}_2)$, $\mathbf{J}_2 = \begin{pmatrix} 0 & 1 \\ -1 &
# 0 \end{pmatrix}$. With the ordering $\mathbf{u} = (x, p_x, y, p_y, z, \delta)^\mathsf{T}$ the Hessian
# $W_{ij} = \partial^2 H_2/\partial u_i\partial u_j$ is
#
# $$
# \mathbf{W} = \begin{pmatrix}
# h^2 + k & 0 & 0 & 0 & 0 & -h \\
# 0 & 1 & 0 & 0 & 0 & 0 \\
# 0 & 0 & -k & 0 & 0 & 0 \\
# 0 & 0 & 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 0 & 0 & 0 \\
# -h & 0 & 0 & 0 & 0 & 1/\gamma_0^2
# \end{pmatrix}.
# $$
#
# It is symmetric by construction, and $\mathbf{J}\mathbf{W}$ reproduces the matrix $\mathbf{A}$. Using
# $\mathbf{J}^\mathsf{T}\mathbf{J} = \mathbf{I}$ and $\mathbf{J}^2 = -\mathbf{I}$ and the symmetry of $\mathbf{W}$,
#
# $$
# \mathbf{A}^\mathsf{T}\mathbf{J} + \mathbf{J}\mathbf{A} = 0
# \quad\Longrightarrow\quad
# \frac{d}{ds}\big(\mathbf{M}^\mathsf{T}\mathbf{J}\mathbf{M}\big) = 0
# \quad\Longrightarrow\quad
# \mathbf{M}^\mathsf{T}\mathbf{J}\mathbf{M} = \mathbf{J},
# $$
#
# that is, $\mathbf{M}$ is symplectic. A symplectic matrix has $\det\mathbf{M} = \pm 1$, and $\det\mathbf{M}$ is
# continuous in $s$ with $\det\mathbf{M}(0) = 1$, so $\det\mathbf{M} = 1$.

# %%
# Expansion of the exact Hamiltonian in the bookkeeping parameter epsilon (deviations scaled by epsilon).
eps = sp.Symbol("epsilon", positive=True)
H_scaled = H_full.subs(beta0, beta0_of_gamma0).subs({x_: eps * x_, px: eps * px, y_: eps * y_, py: eps * py, d: eps * d})
H_series = sp.expand(sp.series(H_scaled, eps, 0, 3).removeO())
H0, H1, H2 = (H_series.coeff(eps, order) for order in range(3))
H2_text = ((px**2 + py**2) / 2 + (h**2 + k) * x_**2 / 2 - k * y_**2 / 2 - h * x_ * d + d**2 / (2 * gamma0**2))
display(sp.Eq(sp.Symbol("H_0"), H0))
display(sp.Eq(sp.Symbol("H_1"), H1))
display(sp.Eq(sp.Symbol("H_2"), H2_text))
assert H0 == -1
assert H1 == 0
assert sp.simplify(H2 - H2_text) == 0

J2 = sp.Matrix([[0, 1], [-1, 0]])
J_sym = sp.diag(J2, J2, J2)
# A = J W with the symmetric Hessian W of H_2 (u' = J grad H_2).
W_hess = sp.hessian(H2_text, state)
assert W_hess == W_hess.T
W_expected = sp.Matrix([
    [h**2 + k, 0, 0, 0, 0, -h],
    [0, 1, 0, 0, 0, 0],
    [0, 0, -k, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [-h, 0, 0, 0, 0, 1 / gamma0**2],
])
display(sp.Eq(sp.Symbol("W"), W_hess))
assert W_hess == W_expected
assert sp.simplify(J_sym * W_hess - A_expected) == sp.zeros(6, 6)
display(sp.Eq(sp.Symbol("A^T J + J A"), A_expected.T * J_sym + J_sym * A_expected))
assert A_expected.T * J_sym + J_sym * A_expected == sp.zeros(6, 6)
display(sp.Eq(sp.Symbol("M^T J M"), sp.Symbol("J")))


# %%
def check_transfer_matrix(M: sp.Matrix, A_el: sp.Matrix) -> None:
    """Check M' = A M, M(0) = I, symplecticity, and det M = 1 for a candidate transfer matrix M(L)."""
    assert sp.simplify(sp.diff(M, L) - A_el * M) == sp.zeros(6, 6)
    assert M.subs(L, 0) == sp.eye(6)
    assert sp.simplify(M.T * J_sym * M - J_sym) == sp.zeros(6, 6)
    assert sp.simplify(M.det()) == 1


def element_matrix_numeric(A_el: sp.Matrix, length: float, params: dict) -> np.ndarray:
    """Return exp(A L) for numeric parameters as a NumPy array (reference value for the closed forms)."""
    A_num = np.array(A_el.subs(params).evalf(), dtype=float)
    return expm(A_num * length)


# %% [markdown]
# ## 4. Drift space
#
# A drift space has $h = 0$ and $k = 0$. Then $\mathbf{A}$ is nilpotent, $\mathbf{A}^2 = 0$, so the series of the
# matrix exponential stops after the linear term:
#
# $$
# \mathbf{M}_\mathrm{drift}(L) = \mathbf{I} + \mathbf{A}\,L = \begin{pmatrix}
# 1 & L & 0 & 0 & 0 & 0 \\
# 0 & 1 & 0 & 0 & 0 & 0 \\
# 0 & 0 & 1 & L & 0 & 0 \\
# 0 & 0 & 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 0 & 1 & L/\gamma_0^2 \\
# 0 & 0 & 0 & 0 & 0 & 1
# \end{pmatrix}.
# $$
#
# That is, $x(L) = x_0 + L\,p_{x0}$, $y(L) = y_0 + L\,p_{y0}$, and $z(L) = z_0 + L\,\delta_0/\gamma_0^2$. The last
# relation is the slip: a particle with $\delta_0 > 0$ is faster and moves ahead of the reference particle, and the
# slip vanishes for $\gamma_0 \to \infty$.

# %%
A_drift = A_gen.subs({h: 0, k: 0})
display(sp.Eq(sp.Symbol("A_drift"), A_drift))
assert A_drift**2 == sp.zeros(6, 6)

M_drift = sp.eye(6) + A_drift * L  # exp(A L) = I + A L because A^2 = 0
M_drift_expected = sp.Matrix([
    [1, L, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, L, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, L / gamma0**2],
    [0, 0, 0, 0, 0, 1],
])
display(sp.Eq(sp.Symbol(r"M_{drift}"), M_drift_expected))
assert sp.simplify(M_drift - M_drift_expected) == sp.zeros(6, 6)
assert sp.simplify(M_drift - (A_drift * L).exp()) == sp.zeros(6, 6)
check_transfer_matrix(M_drift_expected, A_drift)

# %% [markdown]
# ## 5. Sector dipole magnet
#
# A sector dipole magnet has $h \neq 0$ and $k = 0$ (uniform field, perpendicular end faces). The horizontal equation
# $x'' + h^2 x = h\delta$ has the constant particular solution $x = \delta/h$ (because $x'' = 0$ and $h^2\,\delta/h =
# h\delta$). Adding the homogeneous solution and imposing $x(0) = x_0$, $x'(0) = p_{x0}$ gives
#
# $$
# x(L) = \cos(hL)\,x_0 + \frac{\sin(hL)}{h}\,p_{x0} + \frac{1 - \cos(hL)}{h}\,\delta_0 .
# $$
#
# The coefficient of $\delta_0$ is the dispersion $D(L)$ of [2026-001, Sec. 11] for $K = h$. The vertical motion is
# free, $y'' = 0$, because $k = 0$ and the bend adds no $h^2$ term in the vertical plane. The path length follows
# from $z' = -h\,x + \delta_0/\gamma_0^2$ by integration along the element.

# %%
x_f = sp.Function("x")
ode_x = sp.Eq(x_f(L).diff(L, 2) + h**2 * x_f(L), h * d0)
ics_x = {x_f(0): x0, sp.Derivative(x_f(L), L).subs(L, 0): px0}
x_solution = sp.dsolve(ode_x, x_f(L), ics=ics_x).rhs
x_expected = sp.cos(h * L) * x0 + sp.sin(h * L) / h * px0 + (1 - sp.cos(h * L)) / h * d0
display(sp.Eq(sp.Symbol("x(L)"), x_expected))
assert sp.simplify(x_solution - x_expected) == 0

# Path length: z(L) = z0 + integral of (-h x + delta0 / gamma0^2).
tau = sp.Symbol("tau", positive=True)
z_expected = z0 + sp.integrate(-h * x_expected.subs(L, tau) + d0 / gamma0**2, (tau, 0, L))
display(sp.Eq(sp.Symbol("z(L)"), sp.simplify(z_expected)))

# Assemble u(L) and take the Jacobian with respect to the entrance vector.
u_bend = sp.Matrix([
    x_expected,
    sp.diff(x_expected, L),
    y0 + L * py0,
    py0,
    z_expected,
    d0,
])
M_bend = sp.simplify(u_bend.jacobian(u0))

# %% [markdown]
# With the bend angle $\theta = hL$ and $\rho = 1/h$ the transfer matrix is
#
# $$
# \mathbf{M}_\mathrm{bend}(L) = \begin{pmatrix}
# \cos\theta & \rho\sin\theta & 0 & 0 & 0 & \rho(1-\cos\theta) \\
# -\sin\theta/\rho & \cos\theta & 0 & 0 & 0 & \sin\theta \\
# 0 & 0 & 1 & L & 0 & 0 \\
# 0 & 0 & 0 & 1 & 0 & 0 \\
# -\sin\theta & -\rho(1-\cos\theta) & 0 & 0 & 1 & M_{56} \\
# 0 & 0 & 0 & 0 & 0 & 1
# \end{pmatrix},
# \qquad
# M_{56} = \frac{L}{\gamma_0^2} - \rho\,(\theta - \sin\theta).
# $$
#
# The term $-\rho(\theta - \sin\theta) = -\int_0^L h\,D\,ds$ in $M_{56}$ is the path-length change of the dispersive
# orbit in the bend, and $L/\gamma_0^2$ is the slip, as in the drift space. For $\theta \to 0$ the matrix tends to
# the drift matrix.

# %%
theta = h * L
rho = 1 / h
M56_bend = L / gamma0**2 - rho * (theta - sp.sin(theta))
M_bend_expected = sp.Matrix([
    [sp.cos(theta), rho * sp.sin(theta), 0, 0, 0, rho * (1 - sp.cos(theta))],
    [-sp.sin(theta) / rho, sp.cos(theta), 0, 0, 0, sp.sin(theta)],
    [0, 0, 1, L, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [-sp.sin(theta), -rho * (1 - sp.cos(theta)), 0, 0, 1, M56_bend],
    [0, 0, 0, 0, 0, 1],
])
display(sp.Eq(sp.Symbol(r"M_{56}"), M56_bend))
display(sp.Eq(sp.Symbol(r"M_{bend}"), M_bend_expected))
assert sp.simplify(M_bend - M_bend_expected) == sp.zeros(6, 6)

A_bend = A_gen.subs(k, 0)
check_transfer_matrix(M_bend_expected, A_bend)

# %% [markdown]
# ## 6. Quadrupole magnet
#
# A quadrupole magnet has $h = 0$ and $k \neq 0$; $k > 0$ focuses in $x$ and defocuses in $y$. The equations
# decouple completely, $x'' + k x = 0$, $y'' - k y = 0$, $z' = \delta_0/\gamma_0^2$, and $\delta$ does not enter the
# transverse motion (no dispersion, no bend). For $k = g^2 > 0$ the horizontal plane oscillates and the vertical
# plane grows exponentially:
#
# $$
# \mathbf{M}_\mathrm{quad}(L) = \begin{pmatrix}
# \cos gL & \sin(gL)/g & 0 & 0 & 0 & 0 \\
# -g\sin gL & \cos gL & 0 & 0 & 0 & 0 \\
# 0 & 0 & \cosh gL & \sinh(gL)/g & 0 & 0 \\
# 0 & 0 & g\sinh gL & \cosh gL & 0 & 0 \\
# 0 & 0 & 0 & 0 & 1 & L/\gamma_0^2 \\
# 0 & 0 & 0 & 0 & 0 & 1
# \end{pmatrix}, \qquad g = \sqrt{k}.
# $$
#
# For $k = -g^2 < 0$ the roles of the planes are exchanged: the upper-left block is the hyperbolic one and the
# middle block is the trigonometric one, with $g = \sqrt{-k}$.

# %%
def trig_block(w: sp.Expr, length: sp.Expr) -> sp.Matrix:
    """2x2 block of a focusing plane, x'' + w^2 x = 0."""
    return sp.Matrix([[sp.cos(w * length), sp.sin(w * length) / w], [-w * sp.sin(w * length), sp.cos(w * length)]])


def hyp_block(w: sp.Expr, length: sp.Expr) -> sp.Matrix:
    """2x2 block of a defocusing plane, x'' - w^2 x = 0."""
    return sp.Matrix([[sp.cosh(w * length), sp.sinh(w * length) / w], [w * sp.sinh(w * length), sp.cosh(w * length)]])


def quad_matrix(x_block: sp.Matrix, y_block: sp.Matrix, length: sp.Expr) -> sp.Matrix:
    """Assemble the 6x6 quadrupole matrix from the x block, the y block, and the slip of the z row."""
    return sp.diag(x_block, y_block, sp.Matrix([[1, length / gamma0**2], [0, 1]]))


M_quad_f = quad_matrix(trig_block(g, L), hyp_block(g, L), L)  # k = +g^2: focusing in x
M_quad_d = quad_matrix(hyp_block(g, L), trig_block(g, L), L)  # k = -g^2: defocusing in x
display(sp.Eq(sp.Symbol(r"M_{quad}(k=g^2)"), M_quad_f))
display(sp.Eq(sp.Symbol(r"M_{quad}(k=-g^2)"), M_quad_d))

A_quad_f = A_gen.subs({h: 0, k: g**2})
A_quad_d = A_gen.subs({h: 0, k: -g**2})
check_transfer_matrix(M_quad_f, A_quad_f)
check_transfer_matrix(M_quad_d, A_quad_d)

# %% [markdown]
# ### Thin-lens approximation
#
# Take $L \to 0$ with the integrated strength $q = kL$ fixed. Then $gL = \sqrt{qL} \to 0$ and the transfer matrix tends to
#
# $$
# \mathbf{M}_x = \begin{pmatrix} 1 & 0 \\ -q & 1 \end{pmatrix}, \qquad
# \mathbf{M}_y = \begin{pmatrix} 1 & 0 \\ q & 1 \end{pmatrix}.
# $$
#
# The focal length of the focusing plane is $f = 1/q = 1/(kL)$ for $k > 0$.

# %%
q = sp.Symbol("q", positive=True)
g_thin = sp.sqrt(q / L)  # k = g^2 = q / L, so kL = q


def thin_limit(block: sp.Matrix) -> sp.Matrix:
    """Limit L -> 0+ of a 2x2 block with g = sqrt(q / L)."""
    return block.subs(g, g_thin).applyfunc(lambda e: sp.limit(sp.simplify(e), L, 0, "+"))


M_thin_x = thin_limit(trig_block(g, L))
M_thin_y = thin_limit(hyp_block(g, L))
display(sp.Eq(sp.Symbol(r"M_x"), M_thin_x))
display(sp.Eq(sp.Symbol(r"M_y"), M_thin_y))
assert M_thin_x == sp.Matrix([[1, 0], [-q, 1]])
assert M_thin_y == sp.Matrix([[1, 0], [q, 1]])

# %% [markdown]
# ## 7. Checks
#
# 1. Each closed form satisfies $\mathbf{M}' = \mathbf{A}\mathbf{M}$, $\mathbf{M}(0) = \mathbf{I}$, is symplectic, and has
#    $\det\mathbf{M} = 1$ (the function `check_transfer_matrix` above).
# 2. The closed forms agree with the numerical matrix exponential $\exp(\mathbf{A}L)$ (SciPy).
# 3. The closed forms reproduce the linear part of the exact (nonlinear) EOM, integrated numerically for a small
#    initial vector.
# 4. Composition: $\mathbf{M}(L_1 + L_2) = \mathbf{M}(L_2)\,\mathbf{M}(L_1)$.
# 5. Limits: the sector dipole magnet tends to the drift space for $h \to 0$, and the quadrupole magnet tends to the
#    drift space for $k \to 0$.
# 6. The first row and second row of the $\delta_0$ column of the sector dipole magnet reproduce the dispersion
#    $D = \rho(1 - \cos hs)$ and $D' = \sin hs$ of [2026-001, Sec. 11].

# %%
# Numeric samples: parameters, length, and the matching closed form.
H_NUM, K_NUM, LENGTH_BEND, LENGTH_QUAD = 0.2, 2.0, 1.5, 0.4
G_NUM = float(np.sqrt(K_NUM))
GAMMA0_NUM = float(1.0 / np.sqrt(1.0 - BETA0_NUM**2))
cases = {
    "drift": (A_drift, M_drift_expected, {h: 0.0, k: 0.0}, 1.3),
    "bend": (A_bend, M_bend_expected, {h: H_NUM, k: 0.0}, LENGTH_BEND),
    "quad k>0": (A_quad_f, M_quad_f, {h: 0.0, k: K_NUM, g: G_NUM}, LENGTH_QUAD),
    "quad k<0": (A_quad_d, M_quad_d, {h: 0.0, k: -K_NUM, g: G_NUM}, LENGTH_QUAD),
}


def closed_form_numeric(M_expr: sp.Matrix, params: dict, length: float) -> np.ndarray:
    """Evaluate a closed-form matrix at numeric parameters."""
    values = {**params, gamma0: GAMMA0_NUM, L: length}
    return np.array(M_expr.subs(values).evalf(), dtype=float)


# Check 2: closed form against the matrix exponential.
for name, (A_el, M_expr, params, length) in cases.items():
    M_closed = closed_form_numeric(M_expr, params, length)
    M_expm = element_matrix_numeric(A_el.subs(gamma0, GAMMA0_NUM), length, params)
    err = np.max(np.abs(M_closed - M_expm))
    print(f"{name:9s} max |closed form - expm| = {err:.2e}")
    assert err < 1.0e-12

# %%
# Check 3: exact EOM (nonlinear) for a small initial vector.
exact_func = sp.lambdify([*state, h, k, beta0], rhs_exact, "numpy")


def integrate_exact(u_start: np.ndarray, params: dict, length: float) -> np.ndarray:
    """Integrate the exact EOM of the Hamiltonian over `length` with solve_ivp."""
    h_num = float(params[h])
    k_num = float(params[k])

    def rhs(_s: float, u: np.ndarray) -> list:
        return exact_func(*u, h_num, k_num, BETA0_NUM)

    sol = solve_ivp(rhs, (0.0, length), u_start, rtol=ODE_RTOL, atol=ODE_ATOL, method="DOP853")
    return sol.y[:, -1]


u_start = AMPLITUDE * ENTRANCE_DIRECTION
for name, (_A_el, M_expr, params, length) in cases.items():
    u_exact = integrate_exact(u_start, params, length)
    u_linear = closed_form_numeric(M_expr, params, length) @ u_start
    rel_err = np.max(np.abs(u_exact - u_linear)) / AMPLITUDE
    print(f"{name:9s} max |exact - linear| / amplitude = {rel_err:.2e}")
    assert rel_err < REL_TOL_EXACT

# %%
# Check 4: composition of two sections of the same element.
L1, L2 = sp.symbols("L_1 L_2", positive=True)
for name, M_expr in {"drift": M_drift_expected, "bend": M_bend_expected, "quad k>0": M_quad_f}.items():
    lhs = M_expr.subs(L, L1 + L2)
    rhs_prod = M_expr.subs(L, L2) * M_expr.subs(L, L1)
    diff_expr = (lhs - rhs_prod).applyfunc(lambda e: sp.simplify(sp.expand_trig(sp.expand(e.rewrite(sp.exp)))))
    assert diff_expr == sp.zeros(6, 6), name
display(sp.Eq(sp.Symbol("M(L_1+L_2)"), sp.Symbol("M(L_2)") * sp.Symbol("M(L_1)")))

# %%
# Check 5: limits.
M_bend_h0 = M_bend_expected.applyfunc(lambda e: sp.limit(e, h, 0))
assert sp.simplify(M_bend_h0 - M_drift_expected) == sp.zeros(6, 6)
M_quad_k0 = M_quad_f.applyfunc(lambda e: sp.limit(e, g, 0))
assert sp.simplify(M_quad_k0 - M_drift_expected) == sp.zeros(6, 6)
display(sp.Eq(sp.Symbol(r"\lim_{h \to 0} M_{bend}"), M_bend_h0))

# Check 6: dispersion of [2026-001, Sec. 11] with K = h and D(0) = D'(0) = 0.
s = sp.Symbol("s", positive=True)
D_ent = h / h**2 * (1 - sp.cos(h * s))
assert sp.simplify(M_bend_expected[0, 5].subs(L, s) - D_ent) == 0
assert sp.simplify(M_bend_expected[1, 5].subs(L, s) - sp.diff(D_ent, s)) == 0
# Symplectic partners of the dispersion column: M_51 = -M_26 and M_52 = -M_16, a consequence of M^T J M = J.
assert sp.simplify(M_bend_expected[4, 0] + M_bend_expected[1, 5]) == 0
assert sp.simplify(M_bend_expected[4, 1] + M_bend_expected[0, 5]) == 0

# %% [markdown]
# ## 8. Summary and open points
#
# 1. $\mathbf{M}' = \mathbf{A}\mathbf{M}$, $\mathbf{M}(0) = \mathbf{I}$; for constant $h$ and $k$,
#    $\mathbf{M} = \exp(\mathbf{A}L)$. The matrices are symplectic and $\det\mathbf{M} = 1$.
# 2. Drift space: $\mathbf{M} = \mathbf{I} + \mathbf{A}L$ because $\mathbf{A}^2 = 0$; slip $M_{56} = L/\gamma_0^2$.
# 3. Sector dipole magnet: trigonometric horizontal block with the bend angle $\theta = hL$, dispersion column
#    $(\rho(1-\cos\theta),\ \sin\theta)$, path-length row $(-\sin\theta,\ -\rho(1-\cos\theta))$, free vertical motion,
#    and $M_{56} = L/\gamma_0^2 - \rho(\theta - \sin\theta)$.
# 4. Quadrupole magnet: trigonometric block in the focusing plane, hyperbolic block in the defocusing plane, no
#    dispersion, and $M_{56} = L/\gamma_0^2$. The thin-lens limit has the focal length $f = 1/(kL)$.
# 5. All closed forms were checked against $\exp(\mathbf{A}L)$ and against the exact nonlinear EOM at small amplitude.
#
# Open points (not covered here):
#
# - Edge focusing of dipole magnets with end-face angles, and fringe fields.
# - The combined-function magnet ($h \neq 0$, $k \neq 0$), whose horizontal block uses $K^2 = h^2 + k$.
# - The skew quadrupole ($k_s \neq 0$) and other x-y coupling elements.
# - RF cavities, which make $\delta$ non-constant, and nonlinear (sextupole and octupole) elements.

# %% [markdown]
# ## License
#
# The narrative text in this notebook is licensed under
# [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). Code is licensed under GPL-3.0-or-later.
# See [LICENSING.md](../../LICENSING.md).
