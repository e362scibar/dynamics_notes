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
# - File: `topics/second_order_eom/second_order_eom.py`
# - Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# - SPDX-License-Identifier: GPL-3.0-or-later
# - Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

# %% [markdown]
# # Canonical equations of motion expanded to second order in curvilinear coordinates
#
# Memo 2026-003
#
# This memo expands the exact Hamiltonian of [2026-001, Sec. 10] to third order in the deviations
# $(x, p_x, y, p_y, \delta)$. The canonical equations of motion (EOM) derived from it are then exact through second
# order. The linear part reproduces [2026-001, Sec. 11] and [2026-002, Sec. 2]. The second-order part is the starting
# point for second-order transfer maps, which are not covered here.
#
# Scope:
# - Static magnetic fields only (no RF), so $\delta' = 0$.
# - Dipole and quadrupole fields with $k_s = 0$ and gauge $A_x = A_y = 0$. Their potential terms are exactly
#   quadratic in $x$ and $y$ [2026-001, Sec. 10], so they give no third-order terms.
# - Conventions of [2026-001]: $z = s - \beta_0 c\,t$, $\delta = (E - E_0)/(\beta_0 c P_0)$, $h > 0$ when the
#   center of curvature lies inside, $k > 0$ focuses in $x$.
#
# Every result is verified with sympy and numerically against the exact EOM.

# %%
import numpy as np
import sympy as sp
from IPython.display import display
from scipy.integrate import solve_ivp

h = sp.Symbol("h", real=True)
k = sp.Symbol("k", real=True)
gamma0 = sp.Symbol("gamma_0", positive=True)
eps = sp.Symbol("epsilon", positive=True)  # bookkeeping parameter: the order of a term is its power of epsilon
x_, px, y_, py, z_s, d = sp.symbols("x p_x y p_y z delta", real=True)
state = [x_, px, y_, py, z_s, d]
state_names = ["x", "p_x", "y", "p_y", "z", r"\delta"]

# Numerical constants of the checks at the end.
BETA0_NUM = 0.9
GAMMA0_NUM = float(1.0 / np.sqrt(1.0 - BETA0_NUM**2))
H_NUM, K_NUM, LENGTH_NUM = 0.2, 0.5, 1.0
AMPLITUDE = 2.0e-3
ENTRANCE_DIRECTION = np.array([1.0, -0.7, 0.5, 0.3, 0.2, 0.8])  # generic direction of the entrance vector
ODE_RTOL = 1.0e-12
ODE_ATOL = 1.0e-16
RATIO_TOLERANCE = 0.15  # relative tolerance on the expected error ratios 4 and 8

# %% [markdown]
# ## 2. Expansion of the Hamiltonian
#
# The exact Hamiltonian of [2026-001, Sec. 10] with $k_s = 0$ is
#
# $$
# H = -(1 + hx)\,R + \delta + h\Big(x + \tfrac{1}{2} h x^2\Big) + \tfrac{1}{2} k\,(x^2 - y^2),
# \qquad R = \sqrt{1 + 2\delta + \beta_0^2\delta^2 - p_x^2 - p_y^2}.
# $$
#
# Scale every deviation $(x, p_x, y, p_y, \delta) \to \epsilon\,(x, p_x, y, p_y, \delta)$ and expand in $\epsilon$.
# With $1 - \beta_0^2 = 1/\gamma_0^2$ the square root is, through third order,
#
# $$
# R = 1 + \delta - \frac{p_x^2 + p_y^2}{2} - \frac{\delta^2}{2\gamma_0^2}
#     + \frac{\delta\,(p_x^2 + p_y^2)}{2} + \frac{\delta^3}{2\gamma_0^2} + O(4).
# $$

# %%
R_exact = sp.sqrt(1 + 2 * d + (1 - 1 / gamma0**2) * d**2 - px**2 - py**2)
H_full = (-(1 + h * x_) * R_exact + d + h * (x_ + h * x_**2 / 2) + k * (x_**2 - y_**2) / 2)

scale = {x_: eps * x_, px: eps * px, y_: eps * y_, py: eps * py, d: eps * d}
ORDER = 4  # keep terms up to epsilon^3
R_series = sp.expand(sp.series(R_exact.subs(scale), eps, 0, ORDER).removeO())

p_sq = px**2 + py**2
R_text = (1 + eps * d - eps**2 * (p_sq / 2 + d**2 / (2 * gamma0**2))
          + eps**3 * (d * p_sq / 2 + d**3 / (2 * gamma0**2)))
display(sp.Eq(sp.Symbol("R"), R_text))
assert sp.simplify(R_series - sp.expand(R_text)) == 0

# %% [markdown]
# Substituting the series gives $H = H_0 + H_1 + H_2 + H_3 + \dots$ where $H_n$ is homogeneous of degree $n$ in the
# deviations. The constant $H_0 = -1$ does not enter the EOM. The linear part vanishes, $H_1 = 0$, because the
# reference orbit is an equilibrium. The quadratic and cubic parts are
#
# $$
# H_2 = \frac{p_x^2 + p_y^2}{2} + \frac{(h^2 + k)\,x^2}{2} - \frac{k\,y^2}{2} - h\,x\,\delta
#       + \frac{\delta^2}{2\gamma_0^2},
# $$
#
# $$
# H_3 = (hx - \delta)\,\frac{p_x^2 + p_y^2}{2} + (hx - \delta)\,\frac{\delta^2}{2\gamma_0^2}.
# $$
#
# There is no cubic term in $x$ and $y$ alone, because the dipole and quadrupole potential terms are exactly
# quadratic.

# %%
H_series = sp.expand(sp.series(H_full.subs(scale), eps, 0, ORDER).removeO())
H0, H1, H2, H3 = (H_series.coeff(eps, n) for n in range(4))
assert H0 == -1
assert H1 == 0

H2_text = (p_sq / 2 + (h**2 + k) * x_**2 / 2 - k * y_**2 / 2 - h * x_ * d + d**2 / (2 * gamma0**2))
H3_text = (h * x_ - d) * p_sq / 2 + (h * x_ - d) * d**2 / (2 * gamma0**2)
display(sp.Eq(sp.Symbol("H_2"), H2_text))
display(sp.Eq(sp.Symbol("H_3"), H3_text))
assert sp.simplify(H2 - H2_text) == 0
assert sp.simplify(H3 - H3_text) == 0

# No cubic term in x and y alone.
assert sp.simplify(H3_text.subs({px: 0, py: 0, d: 0})) == 0

# %% [markdown]
# ## 3. Equations of motion through second order
#
# Hamilton's equations with $H_2 + H_3$ give the EOM through second order. The $H_2$ part is the linear EOM of
# [2026-001, Sec. 11] with $k_s = 0$, and the $H_3$ part is the second-order correction:
#
# $$
# \begin{aligned}
# x' &= p_x + (hx - \delta)\,p_x, \\
# p_x' &= -(h^2 + k)\,x + h\,\delta - \frac{h\,(p_x^2 + p_y^2)}{2} - \frac{h\,\delta^2}{2\gamma_0^2}, \\
# y' &= p_y + (hx - \delta)\,p_y, \\
# p_y' &= k\,y, \\
# z' &= -h\,x + \frac{\delta}{\gamma_0^2} - \frac{p_x^2 + p_y^2}{2} + \frac{2hx\,\delta - 3\delta^2}{2\gamma_0^2}, \\
# \delta' &= 0 .
# \end{aligned}
# $$

# %%
def hamilton_rhs(hamiltonian: sp.Expr) -> list:
    """Right-hand sides of Hamilton's equations for the pairs (x, p_x), (y, p_y), (z, delta)."""
    return [sp.diff(hamiltonian, px), -sp.diff(hamiltonian, x_), sp.diff(hamiltonian, py), -sp.diff(hamiltonian, y_),
            sp.diff(hamiltonian, d), -sp.diff(hamiltonian, z_s)]


eom_lin = hamilton_rhs(H2)
eom_quad = hamilton_rhs(H3)
eom_text = [
    px + (h * x_ - d) * px,
    -(h**2 + k) * x_ + h * d - h * p_sq / 2 - h * d**2 / (2 * gamma0**2),
    py + (h * x_ - d) * py,
    k * y_,
    -h * x_ + d / gamma0**2 - p_sq / 2 + (2 * h * x_ * d - 3 * d**2) / (2 * gamma0**2),
    sp.Integer(0),
]
for name, lin, quad, text in zip(state_names, eom_lin, eom_quad, eom_text):
    display(sp.Eq(sp.Symbol(f"{name}'"), text))
    assert sp.simplify(lin + quad - text) == 0

# The linear part reproduces the matrix A of [2026-001, Sec. 11] and [2026-002, Sec. 2] (k_s = 0).
A_lin = sp.Matrix(eom_lin).jacobian(state)
A_expected = sp.Matrix([
    [0, 1, 0, 0, 0, 0],
    [-(h**2 + k), 0, 0, 0, 0, h],
    [0, 0, 0, 1, 0, 0],
    [0, 0, k, 0, 0, 0],
    [-h, 0, 0, 0, 0, 1 / gamma0**2],
    [0, 0, 0, 0, 0, 0],
])
assert sp.simplify(A_lin - A_expected) == sp.zeros(6, 6)

# %% [markdown]
# Reading of the second-order terms:
#
# - $x' = p_x\,(1 + hx - \delta)$: the factor $1 + hx$ is the metric factor of the curved frame and $-\delta$
#   comes from dividing by the longitudinal momentum $p/P_0 = 1 + \delta + \dots$. The same holds for $y'$.
# - $p_x'$ contains $-h\,p_\perp^2/2$ with $p_\perp^2 = p_x^2 + p_y^2$. It is the sagitta of the curved reference
#   orbit: a particle with a transverse slope covers a longer arc in the bend.
# - $z'$ contains $-p_\perp^2/2$, the path-length increase caused by the betatron slope, and the terms with
#   $1/\gamma_0^2$ that describe the speed difference.
# - $p_y' = ky$ has no second-order term, and $\delta' = 0$.

# %% [markdown]
# ## 4. Checks
#
# 1. Exact EOM versus expansion: the series of the exact right-hand sides in $\epsilon$ agrees with
#    $\epsilon\,(\text{linear}) + \epsilon^2\,(\text{second order})$ through $\epsilon^2$.
# 2. Numerical scaling. Integrate the exact EOM and the truncated EOM for the amplitude $a$ and for $a/2$ over a
#    magnet with $h = 0.2$, $k = 0.5$, $L = 1$. The error of the linear EOM must scale as $a^2$ (ratio 4 between
#    $a$ and $a/2$), and the error of the second-order EOM as $a^3$ (ratio 8).

# %%
rhs_exact = hamilton_rhs(H_full)
for name, exact, lin, quad in zip(state_names, rhs_exact, eom_lin, eom_quad):
    series_exact = sp.expand(sp.series(exact.subs(scale), eps, 0, 3).removeO())
    assert sp.simplify(series_exact - (eps * lin + eps**2 * quad)) == 0, name
display(sp.Eq(sp.Symbol(r"u'_{exact}"), sp.Symbol(r"\epsilon\,u'_{(1)} + \epsilon^2 u'_{(2)} + O(\epsilon^3)")))

# %%
args = [*state, h, k, gamma0]
rhs_exact_func = sp.lambdify(args, rhs_exact, "numpy")
rhs_lin_func = sp.lambdify(args, [e for e in eom_lin], "numpy")
rhs_second_func = sp.lambdify(args, [lin + quad for lin, quad in zip(eom_lin, eom_quad)], "numpy")


def integrate(rhs_func, amplitude: float) -> np.ndarray:
    """Integrate an EOM right-hand side over the magnet and return the final state."""
    u_start = amplitude * ENTRANCE_DIRECTION

    def rhs(_s: float, u: np.ndarray) -> list:
        return rhs_func(*u, H_NUM, K_NUM, GAMMA0_NUM)

    sol = solve_ivp(rhs, (0.0, LENGTH_NUM), u_start, rtol=ODE_RTOL, atol=ODE_ATOL, method="DOP853")
    return sol.y[:, -1]


def errors(amplitude: float) -> tuple:
    """Max errors of the linear and the second-order EOM with respect to the exact EOM."""
    u_exact = integrate(rhs_exact_func, amplitude)
    err_lin = np.max(np.abs(integrate(rhs_lin_func, amplitude) - u_exact))
    err_second = np.max(np.abs(integrate(rhs_second_func, amplitude) - u_exact))
    return err_lin, err_second


lin_a, second_a = errors(AMPLITUDE)
lin_half, second_half = errors(AMPLITUDE / 2)
ratio_lin, ratio_second = lin_a / lin_half, second_a / second_half
print(f"amplitude {AMPLITUDE:.1e}: linear error {lin_a:.2e}, second-order error {second_a:.2e}")
print(f"amplitude {AMPLITUDE / 2:.1e}: linear error {lin_half:.2e}, second-order error {second_half:.2e}")
print(f"ratio linear = {ratio_lin:.2f} (expected 4), second order = {ratio_second:.2f} (expected 8)")
assert abs(ratio_lin - 4.0) < 4.0 * RATIO_TOLERANCE
assert abs(ratio_second - 8.0) < 8.0 * RATIO_TOLERANCE
assert second_a < lin_a

# %% [markdown]
# ## 5. Summary and open points
#
# 1. $H = H_2 + H_3 + \dots$ with $H_2$ the quadratic Hamiltonian of the linear EOM and
#    $H_3 = (hx - \delta)\,\big[(p_x^2 + p_y^2)/2 + \delta^2/(2\gamma_0^2)\big]$.
# 2. The EOM through second order are listed in Section 3. The second-order terms come only from the square root
#    $R$ and the curvature factor $1 + hx$. The dipole and quadrupole potentials add none.
# 3. The checks confirm the expansion symbolically and by the error scaling $a^2 \to a^3$.
#
# Open points (not covered here):
#
# - Second-order transfer maps (the tensor $T_{ijk}$) of the drift space, the sector dipole magnet, and the
#   quadrupole magnet from these EOM, and the chromaticity that follows.
# - Quadrupole curvature corrections, which are neglected in [2026-001], and sextupole and octupole terms.
# - Skew quadrupole ($k_s \neq 0$) and RF.

# %% [markdown]
# ## License
#
# The narrative text in this notebook is licensed under
# [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). Code is licensed under GPL-3.0-or-later.
# See [LICENSING.md](../../LICENSING.md).
