"""First-order-condition solver for rational behavioral response.

For each age type a, choose (theta, e) in [0,1] x [0,1] to maximize:
  U(theta, e) = sum_g alpha_g log(1 + m_g(theta))
              - 0.5 * kappa * e^2
              - V_a * (1-e) * sum_g p_g * m_g(theta) * I_g/N_g

Voluntary groups: m_g(theta) = theta * m_bar_g
Institutional groups: m_g = m_bar_g (independent of theta)

Decompose into:
  A = sum_{vol} p_g * m_bar_g * (I_g/N_g)
  B = sum_{inst} p_g * m_bar_g * (I_g/N_g)

FOC for e:     -kappa*e + V_a * (theta*A + B) = 0
               e* = V_a * (theta*A + B) / kappa, clamped to [0,1]
FOC for theta: sum_{vol} alpha_g * m_bar_g / (1 + theta * m_bar_g)
               = V_a * (1-e*) * A

Solve by 1-D root-find over theta in [eps, 1], with e* substituted in closed form.
Corner solutions occur when alpha values dominate or when V*A is small.
"""

import numpy as np
from scipy.optimize import brentq

from sir.config import ScenarioConfig
from sir.constants import GroupType


def _compute_AB(
    cfg: ScenarioConfig, prevalence_by_group_type: dict[GroupType, float]
) -> tuple[float, float]:
    """Return (A, B): exposure intensity from voluntary and institutional groups."""
    A = 0.0
    B = 0.0
    for gt in GroupType:
        p = cfg.group_p_baseline[gt]
        m_bar = cfg.group_m_bar[gt]
        prev = prevalence_by_group_type[gt]
        contribution = p * m_bar * prev
        if gt in cfg.voluntary_groups:
            A += contribution
        else:
            B += contribution
    return A, B


def _solve_one_type(cfg: ScenarioConfig, V_a: float, A: float, B: float) -> tuple[float, float]:
    """Solve (theta*, e*) for one age type. Returns clamped values in [0,1]."""
    if V_a == 0.0:
        return 1.0, 0.0
    if A == 0.0:
        # No voluntary-group exposure; theta unconstrained, e from B only
        e_star = min(1.0, max(0.0, V_a * B / cfg.kappa))
        return 1.0, e_star

    vol_alpha_m: list[tuple[float, float]] = [
        (cfg.group_alpha[gt], cfg.group_m_bar[gt]) for gt in cfg.voluntary_groups
    ]

    def theta_foc(theta: float) -> float:
        e_star = V_a * (theta * A + B) / cfg.kappa
        e_star = min(max(e_star, 0.0), 1.0)
        lhs = sum(alpha * m_bar / (1.0 + theta * m_bar) for alpha, m_bar in vol_alpha_m)
        rhs = V_a * (1.0 - e_star) * A
        return lhs - rhs

    f_at_1 = theta_foc(1.0)
    f_at_0 = theta_foc(1e-6)
    if f_at_1 >= 0:
        theta_star = 1.0
    elif f_at_0 <= 0:
        theta_star = 1e-6
    else:
        theta_star = brentq(theta_foc, 1e-6, 1.0, xtol=1e-6)

    e_star = V_a * (theta_star * A + B) / cfg.kappa
    e_star = min(max(e_star, 0.0), 1.0)
    return theta_star, e_star


def solve_foc_by_age(
    cfg: ScenarioConfig, prevalence_by_group_type: dict[GroupType, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta_by_age, e_by_age), each of shape (7,)."""
    A, B = _compute_AB(cfg, prevalence_by_group_type)
    theta = np.empty(7, dtype=np.float64)
    e = np.empty(7, dtype=np.float64)
    for a in range(7):
        V_a = cfg.V_a[a]
        theta[a], e[a] = _solve_one_type(cfg, V_a, A, B)
    return theta, e
