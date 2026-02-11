"""Aion's Edge — Colony Survival Strategy Game.

A Streamlit application with three levels:

* **Level 1** — Linear Programming (The Survival Equation)
* **Level 2** — Multi-Objective Optimisation (The Tri-Lemma)
* **Level 3** — Voting Theory & MCDA (The Council of Factions)

Run with::

    streamlit run src/app.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Polygon
from numpy.typing import NDArray

# Ensure project root is on sys.path so we can import the
# OptimizationEngine module that lives in src/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.OptimizationEngine import (  # noqa: E402
    LPSolver,
    MOOSolver,
    VotingSystem,
)


# ################################################################
#  LEVEL 1 — Linear Programming
# ################################################################

# ================================================================
# Constants — Level 1 default scenario parameters
# ================================================================

# Default objective: maximise Z = 30·x₁ + 20·x₂
DEFAULT_OBJECTIVE = np.array([30.0, 20.0])

# Constraint coefficient matrix (unchanged by events)
#   C1 (Energy):  2·x₁ + 1·x₂ ≤ b₁
#   C2 (Labour):  1·x₁ + 2·x₂ ≤ b₂
A_UB = np.array([
    [2.0, 1.0],
    [1.0, 2.0],
])

# Default right-hand-side values (modified by events)
DEFAULT_B_UB = np.array([100.0, 80.0])

# Decision-variable bounds (non-negative)
BOUNDS: List[Tuple[float, None]] = [(0, None), (0, None)]

# Total turns to survive ("one year")
TOTAL_TURNS = 12

# Axis limits for the plot
X_MAX = 70
Y_MAX = 100


# ================================================================
# Event System
# ================================================================

@dataclass
class GameEvent:
    """Describes a random event that modifies the LP.

    Attributes:
        name: Short identifier (e.g. "dust_storm").
        title: Display title shown in the UI.
        description: Flavour text explaining the event.
        icon: Emoji icon for quick visual identification.
        b_ub_override: If set, replace b_ub with these
            values while the event is active.
        coeff_multiplier: If set, multiply objective
            coefficients by this factor.
    """

    name: str
    title: str
    description: str
    icon: str
    b_ub_override: Optional[NDArray[np.float64]] = None
    coeff_multiplier: Optional[float] = None


# Catalogue of possible events.
EVENT_CATALOGUE: Dict[str, GameEvent] = {
    "dust_storm": GameEvent(
        name="dust_storm",
        title="🌪️ 沙尘暴 (Dust Storm)",
        description=(
            "一场猛烈的沙尘暴遮蔽了太阳能板！"
            "电力上限从 100 降至 60。"
        ),
        icon="🌪️",
        b_ub_override=np.array([60.0, 80.0]),
    ),
    "flu": GameEvent(
        name="flu",
        title="🤒 流感爆发 (Flu Outbreak)",
        description=(
            "殖民者大面积感染流感！"
            "可用人力从 80 降至 50。"
        ),
        icon="🤒",
        b_ub_override=np.array([100.0, 50.0]),
    ),
    "tech_breakthrough": GameEvent(
        name="tech_breakthrough",
        title="🔬 技术突破 (Tech Breakthrough)",
        description=(
            "研究团队取得重大突破！"
            "所有产出系数翻倍！(本回合有效)"
        ),
        icon="🔬",
        coeff_multiplier=2.0,
    ),
    "clear": GameEvent(
        name="clear",
        title="☀️ 风平浪静 (All Clear)",
        description="本回合一切正常，没有突发事件。",
        icon="☀️",
    ),
}

# Probabilities: 25% dust storm, 20% flu, 10% tech, 45% clear
EVENT_WEIGHTS: List[Tuple[str, float]] = [
    ("dust_storm", 0.25),
    ("flu", 0.20),
    ("tech_breakthrough", 0.10),
    ("clear", 0.45),
]


def roll_event() -> GameEvent:
    """Randomly select an event based on EVENT_WEIGHTS.

    Returns:
        The selected ``GameEvent``.
    """
    names, weights = zip(*EVENT_WEIGHTS)
    chosen = random.choices(names, weights=weights, k=1)[0]
    return EVENT_CATALOGUE[chosen]


# ================================================================
# Session-state initialisation
# ================================================================

def _init_session_state() -> None:
    """Initialise all session-state keys on first run."""
    defaults: Dict[str, object] = {
        # --- Level 1 state ---
        "turn": 1,
        "b_ub": DEFAULT_B_UB.copy(),
        "objective": DEFAULT_OBJECTIVE.copy(),
        "current_event": None,
        "event_log": [],
        "total_score": 0.0,
        "game_over": False,
        # --- Level 2 state ---
        "moo_solutions": None,
        "moo_pareto_result": None,
        "moo_selected_idx": None,
        # --- Level 3 state ---
        "voting_ballots": None,
        "voting_factions": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def advance_turn() -> None:
    """Advance the game by one turn.

    Resets constraints to defaults, then rolls a new random
    event and applies its effects.
    """
    if st.session_state.turn >= TOTAL_TURNS:
        st.session_state.game_over = True
        return

    st.session_state.b_ub = DEFAULT_B_UB.copy()
    st.session_state.objective = DEFAULT_OBJECTIVE.copy()

    event = roll_event()
    st.session_state.current_event = event

    if event.b_ub_override is not None:
        st.session_state.b_ub = event.b_ub_override.copy()
    if event.coeff_multiplier is not None:
        st.session_state.objective = (
            DEFAULT_OBJECTIVE * event.coeff_multiplier
        )

    st.session_state.event_log.append(
        f"回合 {st.session_state.turn + 1}: "
        f"{event.title}"
    )
    st.session_state.turn += 1


# ================================================================
# Level 1 helpers
# ================================================================

def _get_b_ub() -> NDArray[np.float64]:
    """Return the current right-hand-side vector."""
    return np.asarray(
        st.session_state.b_ub, dtype=np.float64
    )


def _get_objective() -> NDArray[np.float64]:
    """Return the current objective coefficients."""
    return np.asarray(
        st.session_state.objective, dtype=np.float64
    )


def compute_optimal_solution() -> Tuple[
    NDArray[np.float64], float
]:
    """Use LPSolver to find the optimum for the current LP.

    Returns:
        Tuple of (solution vector [x1, x2], optimal Z).
    """
    result = LPSolver.solve(
        c=_get_objective(),
        A_ub=A_UB,
        b_ub=_get_b_ub(),
        bounds=BOUNDS,
        maximize=True,
    )
    return result.solution, result.optimal_value


def is_feasible(x1: float, x2: float) -> bool:
    """Check whether a point satisfies all constraints."""
    point = np.array([x1, x2])
    b_ub = _get_b_ub()
    within_ub = np.all(A_UB @ point <= b_ub + 1e-9)
    return bool(within_ub and x1 >= 0 and x2 >= 0)


def compute_feasible_polygon() -> NDArray[np.float64]:
    """Compute the vertices of the feasible polygon.

    Returns:
        Array of shape (n_vertices, 2).
    """
    b_ub = _get_b_ub()

    lines = []
    for i in range(A_UB.shape[0]):
        lines.append((A_UB[i, 0], A_UB[i, 1], b_ub[i]))
    lines.append((1.0, 0.0, 0.0))
    lines.append((0.0, 1.0, 0.0))

    vertices = []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            a = np.array([
                [lines[i][0], lines[i][1]],
                [lines[j][0], lines[j][1]],
            ])
            b = np.array([lines[i][2], lines[j][2]])
            if abs(np.linalg.det(a)) < 1e-12:
                continue
            point = np.linalg.solve(a, b)
            if is_feasible(point[0], point[1]):
                vertices.append(point)

    vertices = np.array(vertices)
    centroid = vertices.mean(axis=0)
    angles = np.arctan2(
        vertices[:, 1] - centroid[1],
        vertices[:, 0] - centroid[0],
    )
    return vertices[np.argsort(angles)]


def draw_lp_plot(
    x1_player: float,
    x2_player: float,
    x1_opt: float,
    x2_opt: float,
) -> plt.Figure:
    """Render the 2-D feasible-region plot."""
    b_ub = _get_b_ub()
    obj = _get_objective()

    fig, ax = plt.subplots(figsize=(8, 6))
    x_range = np.linspace(0, X_MAX, 400)

    # Constraint lines
    y_c1 = b_ub[0] - 2 * x_range
    ax.plot(
        x_range, y_c1,
        label=(
            rf"$C_1$: $2x_1 + x_2 \leq {b_ub[0]:.0f}$"
            " (电力)"
        ),
        color="#1f77b4", linewidth=2,
    )

    y_c2 = (b_ub[1] - x_range) / 2
    ax.plot(
        x_range, y_c2,
        label=(
            rf"$C_2$: $x_1 + 2x_2 \leq {b_ub[1]:.0f}$"
            " (人力)"
        ),
        color="#ff7f0e", linewidth=2,
    )

    # Ghost lines when constraints differ from default
    if not np.allclose(b_ub, DEFAULT_B_UB):
        y_c1_d = DEFAULT_B_UB[0] - 2 * x_range
        ax.plot(
            x_range, y_c1_d,
            "--", color="#1f77b4", alpha=0.25, linewidth=1,
        )
        y_c2_d = (DEFAULT_B_UB[1] - x_range) / 2
        ax.plot(
            x_range, y_c2_d,
            "--", color="#ff7f0e", alpha=0.25, linewidth=1,
        )

    # Feasible region
    polygon_verts = compute_feasible_polygon()
    ax.add_patch(Polygon(
        polygon_verts, closed=True,
        facecolor="lightgray", edgecolor="gray",
        alpha=0.45, label="可行域 (Feasible Region)",
    ))

    # Player point
    player_color = (
        "red" if not is_feasible(x1_player, x2_player)
        else "#e74c3c"
    )
    ax.plot(
        x1_player, x2_player, "o",
        color=player_color, markersize=12,
        markeredgecolor="black", markeredgewidth=1.5,
        label=(
            f"玩家选择 ({x1_player:.0f}, {x2_player:.0f})"
        ),
        zorder=5,
    )

    # Optimal point (green star)
    ax.plot(
        x1_opt, x2_opt, "*",
        color="#2ecc71", markersize=20,
        markeredgecolor="black", markeredgewidth=1,
        label=f"最优解 ({x1_opt:.1f}, {x2_opt:.1f})",
        zorder=5,
    )

    # Iso-profit line
    z_player = obj[0] * x1_player + obj[1] * x2_player
    if obj[1] != 0:
        y_iso = (z_player - obj[0] * x_range) / obj[1]
        ax.plot(
            x_range, y_iso, "--",
            color="#9b59b6", linewidth=1, alpha=0.6,
            label=f"等利润线 Z = {z_player:.0f}",
        )

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel(r"$x_1$ — 氧气产量 (Oxygen)", fontsize=12)
    ax.set_ylabel(r"$x_2$ — 食物产量 (Food)", fontsize=12)
    ax.set_title(
        f"Aion's Edge · Level 1 — 回合 "
        f"{st.session_state.turn}/{TOTAL_TURNS}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _render_event_banner() -> None:
    """Display the current-turn event as a prominent banner."""
    event: Optional[GameEvent] = (
        st.session_state.current_event
    )
    if event is None:
        return

    if event.name in ("dust_storm", "flu"):
        st.warning(
            f"{event.title}\n\n{event.description}",
            icon=event.icon,
        )
    elif event.name == "tech_breakthrough":
        st.success(
            f"{event.title}\n\n{event.description}",
            icon="🔬",
        )
    else:
        st.info(
            f"{event.title}\n\n{event.description}",
            icon="☀️",
        )


# ================================================================
# Level 1 UI
# ================================================================

def render_level1() -> None:
    """Render Level 1 — Linear Programming survival mode."""
    _render_event_banner()

    # Game over screen
    if st.session_state.game_over:
        st.balloons()
        st.success(
            f"🎉 恭喜！你成功生存了 {TOTAL_TURNS} 回合！\n\n"
            f"累计总产值: "
            f"**{st.session_state.total_score:.0f}**"
        )
        if st.button("🔄 重新开始", key="l1_restart"):
            for k in [
                "turn", "b_ub", "objective",
                "current_event", "event_log",
                "total_score", "game_over",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        return

    b_ub = _get_b_ub()
    obj = _get_objective()

    st.markdown(
        f"""
        你是殖民地中央 AI **AION**。调整 **氧气** ($x_1$) 和
        **食物** ($x_2$) 的生产配额，在有限的 **电力** 和
        **人力** 约束下，最大化殖民地的总产值：

        $$Z = {obj[0]:.0f}\\,x_1 + {obj[1]:.0f}\\,x_2$$
        """
    )

    # Sidebar controls
    st.sidebar.header("⚙️ L1 生产控制面板")
    st.sidebar.markdown(
        f"**回合 {st.session_state.turn} / {TOTAL_TURNS}**"
    )

    x1_player = st.sidebar.slider(
        "x₁ — 氧气产量", 0, 60, 20, 1, key="l1_x1",
    )
    x2_player = st.sidebar.slider(
        "x₂ — 食物产量", 0, 80, 20, 1, key="l1_x2",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 当前约束")
    st.sidebar.latex(
        rf"C_1:\;2x_1+x_2\leq {b_ub[0]:.0f}"
    )
    st.sidebar.latex(
        rf"C_2:\;x_1+2x_2\leq {b_ub[1]:.0f}"
    )

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "⏭️ 提交并进入下一回合",
        use_container_width=True,
        key="l1_next",
    ):
        if is_feasible(x1_player, x2_player):
            st.session_state.total_score += float(
                obj[0] * x1_player + obj[1] * x2_player
            )
        advance_turn()
        st.rerun()

    st.sidebar.metric(
        "📈 累计总产值",
        f"{st.session_state.total_score:.0f}",
    )

    # Compute optimum
    opt_solution, opt_value = compute_optimal_solution()
    x1_opt, x2_opt = opt_solution[0], opt_solution[1]

    z_player = float(
        obj[0] * x1_player + obj[1] * x2_player
    )
    feasible = is_feasible(x1_player, x2_player)

    col_chart, col_info = st.columns([2, 1])

    with col_chart:
        fig = draw_lp_plot(
            x1_player, x2_player, x1_opt, x2_opt,
        )
        st.pyplot(fig)

    with col_info:
        st.markdown("### 📊 面板状态")
        if feasible:
            st.success("✅ 生产方案可行！")
        else:
            st.error("🚨 资源不足！当前方案超出约束！")

        st.metric(
            "当前总产值 Z", f"{z_player:.0f}",
            delta=(
                f"{z_player - opt_value:+.0f} vs 最优"
                if feasible else "不可行"
            ),
        )
        st.metric("最优总产值 Z*", f"{opt_value:.0f}")

        st.markdown("---")
        c1_used = 2 * x1_player + x2_player
        c2_used = x1_player + 2 * x2_player
        st.markdown(
            f"**电力** (C₁): {c1_used:.0f}/{b_ub[0]:.0f}"
        )
        st.progress(min(c1_used / b_ub[0], 1.0))
        st.markdown(
            f"**人力** (C₂): {c2_used:.0f}/{b_ub[1]:.0f}"
        )
        st.progress(min(c2_used / b_ub[1], 1.0))

        st.markdown("---")
        st.markdown(
            f"**最优解**: $x_1^*={x1_opt:.1f}$, "
            f"$x_2^*={x2_opt:.1f}$"
        )
        if feasible and opt_value > 0:
            eff = z_player / opt_value * 100
            st.markdown(f"**效率**: {eff:.1f}%")
            if eff >= 99.9:
                st.balloons()
                st.success("🎉 完美！")
            elif eff >= 90:
                st.info("👍 非常接近！")
            elif eff >= 70:
                st.warning("💡 还有优化空间。")
            else:
                st.warning("⚠️ 产值偏低。")

    if st.session_state.event_log:
        with st.expander("📜 事件日志"):
            for e in reversed(st.session_state.event_log):
                st.markdown(f"- {e}")


# ################################################################
#  LEVEL 2 — Multi-Objective Optimisation (Pareto Front)
# ################################################################

# Number of random candidate solutions to generate
N_MOO_SOLUTIONS = 50


def _generate_moo_solutions() -> NDArray[np.float64]:
    """Generate random solutions in objective space.

    Each solution has two objectives:
      - col 0: Environmental Pollution (lower is better)
      - col 1: Economic Output        (higher is better)

    A mild negative correlation makes the trade-off visible.

    Returns:
        Array of shape (N_MOO_SOLUTIONS, 2).
    """
    rng = np.random.default_rng()
    pollution = rng.uniform(10, 100, N_MOO_SOLUTIONS)
    # Economic output inversely correlated with low
    # pollution (cleaning costs money).
    output = (
        120
        - 0.6 * pollution
        + rng.normal(0, 15, N_MOO_SOLUTIONS)
    )
    output = np.clip(output, 5, 150)
    return np.column_stack([pollution, output])


def draw_pareto_plot(
    solutions: NDArray[np.float64],
    pareto_idx: List[int],
    dominated_idx: List[int],
    selected_idx: Optional[int],
) -> plt.Figure:
    """Draw the scatter plot with Pareto front highlighted.

    Args:
        solutions: (n, 2) objective-space points.
        pareto_idx: Indices on the Pareto front.
        dominated_idx: Indices of dominated solutions.
        selected_idx: Index the player selected (or None).

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Dominated points — blue
    dom = solutions[dominated_idx]
    ax.scatter(
        dom[:, 0], dom[:, 1],
        c="#3498db", s=50, alpha=0.6,
        label="被支配解 (Dominated)", zorder=3,
    )

    # Pareto front — red, connected by line
    par = solutions[pareto_idx]
    sort_order = np.argsort(par[:, 0])
    par_sorted = par[sort_order]
    ax.scatter(
        par[:, 0], par[:, 1],
        c="#e74c3c", s=80, edgecolors="black",
        linewidths=1,
        label="帕累托前沿 (Pareto Front)",
        zorder=4,
    )
    ax.plot(
        par_sorted[:, 0], par_sorted[:, 1],
        "-", color="#e74c3c", alpha=0.5, linewidth=1.5,
        zorder=3,
    )

    # Player-selected point highlight
    if selected_idx is not None:
        pt = solutions[selected_idx]
        ax.scatter(
            [pt[0]], [pt[1]],
            c="gold", s=200, marker="*",
            edgecolors="black", linewidths=1.5,
            label="你的选择", zorder=5,
        )

    # Label each point with its index
    for i, (px, py) in enumerate(solutions):
        ax.annotate(
            str(i), (px, py),
            fontsize=6, alpha=0.5,
            textcoords="offset points",
            xytext=(4, 4),
        )

    ax.set_xlabel(
        "环境污染 (← 越低越好)", fontsize=12,
    )
    ax.set_ylabel(
        "经济产出 (越高越好 →)", fontsize=12,
    )
    ax.set_title(
        "Aion's Edge · Level 2 — 帕累托前沿",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _find_dominator(
    idx: int,
    solutions: NDArray[np.float64],
    pareto_front: List[int],
) -> Optional[int]:
    """Find a Pareto-front member that dominates *idx*.

    Dominance here means: lower pollution AND higher output
    (weakly on both, strictly on at least one).

    Args:
        idx: Index of the candidate solution.
        solutions: Full solution matrix.
        pareto_front: Indices of Pareto-optimal solutions.

    Returns:
        Index of a dominator, or ``None``.
    """
    candidate = solutions[idx]
    for p_idx in pareto_front:
        other = solutions[p_idx]
        # pollution: lower is better  →  other[0] ≤ cand[0]
        # output:   higher is better  →  other[1] ≥ cand[1]
        if (
            other[0] <= candidate[0]
            and other[1] >= candidate[1]
            and (
                other[0] < candidate[0]
                or other[1] > candidate[1]
            )
        ):
            return p_idx
    return None


def render_level2() -> None:
    """Render Level 2 — Multi-Objective Optimisation."""
    st.markdown(
        """
        殖民地需要在 **经济产出** 和 **环境保护** 之间权衡。
        下方 50 个随机方案中，**红色** 点构成 **帕累托前沿**
        (Pareto Front) — 不可能在不牺牲一个目标的情况下
        改善另一个目标的最优集合。

        选择一个方案编号，系统将判断它是帕累托最优还是
        被支配解。
        """
    )

    # Sidebar controls
    st.sidebar.header("🔬 L2 帕累托分析")

    if st.sidebar.button(
        "🎲 生成新方案集",
        use_container_width=True,
        key="l2_gen",
    ):
        st.session_state.moo_solutions = None
        st.session_state.moo_pareto_result = None
        st.session_state.moo_selected_idx = None
        st.rerun()

    # Generate / retrieve data
    if st.session_state.moo_solutions is None:
        solutions = _generate_moo_solutions()
        st.session_state.moo_solutions = solutions
        result = MOOSolver.find_pareto_front(
            solutions, maximize=[False, True],
        )
        st.session_state.moo_pareto_result = result

    solutions = st.session_state.moo_solutions
    pareto_result = st.session_state.moo_pareto_result

    # Player selection
    selected_idx: int = st.sidebar.selectbox(
        "选择方案编号 (0–49)",
        options=list(range(N_MOO_SOLUTIONS)),
        index=0,
        key="l2_select",
    )

    # Layout
    col_chart, col_info = st.columns([2, 1])

    with col_chart:
        fig = draw_pareto_plot(
            solutions,
            pareto_result.pareto_front,
            pareto_result.dominated,
            selected_idx,
        )
        st.pyplot(fig)

    with col_info:
        st.markdown("### 📊 方案分析")

        pt = solutions[selected_idx]
        st.markdown(
            f"**方案 #{selected_idx}**  \n"
            f"环境污染: `{pt[0]:.1f}`  \n"
            f"经济产出: `{pt[1]:.1f}`"
        )

        is_pareto = (
            selected_idx in pareto_result.pareto_front
        )
        if is_pareto:
            st.success(
                "⭐ 帕累托最优！"
                "该方案不被任何其他方案支配。"
            )
        else:
            st.error(
                "❌ 被支配解！存在其他方案在所有目标上"
                "都不差于此方案，且至少一个目标更优。"
            )
            dom_by = _find_dominator(
                selected_idx, solutions,
                pareto_result.pareto_front,
            )
            if dom_by is not None:
                dp = solutions[dom_by]
                st.markdown(
                    f"例如，**方案 #{dom_by}** "
                    f"(污染 `{dp[0]:.1f}`, "
                    f"产出 `{dp[1]:.1f}`) 支配了你的选择。"
                )

        st.markdown("---")
        st.markdown("### 📈 帕累托前沿统计")
        st.markdown(
            f"- 帕累托最优方案数: "
            f"**{len(pareto_result.pareto_front)}**"
        )
        st.markdown(
            f"- 被支配方案数: "
            f"**{len(pareto_result.dominated)}**"
        )

        # Nadir point
        nadir = MOOSolver.nadir_point(
            pareto_result.pareto_points,
            maximize=[False, True],
        )
        st.markdown("---")
        st.markdown("### ⚠️ Nadir 点 (最坏边界)")
        st.markdown(
            f"污染最高: `{nadir[0]:.1f}`  \n"
            f"产出最低: `{nadir[1]:.1f}`"
        )
        st.caption(
            "Nadir 点代表帕累托前沿上各目标的最差值，"
            "殖民地必须远离这个灾难点。"
        )


# ################################################################
#  LEVEL 3 — Voting Theory (The Council of Factions)
# ################################################################

# Faction names and candidate plans
FACTIONS = [
    "⛏️ 矿工公会",
    "🌿 环保主义者",
    "👨\u200d👩\u200d👧 居民家庭",
]
PLANS = ["方案 A", "方案 B", "方案 C"]

# Pre-built scenarios that guarantee interesting results.
VOTING_SCENARIOS: List[Dict[str, object]] = [
    {
        "name": "经典孔多塞悖论",
        "description": (
            "矿工偏好 A>B>C，环保偏好 B>C>A，"
            "居民偏好 C>A>B — 产生循环！"
        ),
        "ballots": [
            (["方案 A", "方案 B", "方案 C"], 4),
            (["方案 B", "方案 C", "方案 A"], 3),
            (["方案 C", "方案 A", "方案 B"], 2),
        ],
        "factions": [
            ("⛏️ 矿工公会 (4票)", "A > B > C"),
            ("🌿 环保主义者 (3票)", "B > C > A"),
            ("👨\u200d👩\u200d👧 居民家庭 (2票)", "C > A > B"),
        ],
    },
    {
        "name": "多数制 vs 波达计数分歧",
        "description": (
            "多数制和波达计数产生不同赢家！"
        ),
        "ballots": [
            (["方案 A", "方案 B", "方案 C"], 5),
            (["方案 B", "方案 C", "方案 A"], 4),
            (["方案 C", "方案 B", "方案 A"], 3),
        ],
        "factions": [
            ("⛏️ 矿工公会 (5票)", "A > B > C"),
            ("🌿 环保主义者 (4票)", "B > C > A"),
            ("👨\u200d👩\u200d👧 居民家庭 (3票)", "C > B > A"),
        ],
    },
    {
        "name": "随机偏好",
        "description": "随机生成的派系偏好。",
        "ballots": None,  # generated at runtime
        "factions": None,
    },
]


def _generate_random_ballots() -> Tuple[
    List[Tuple[List[str], int]],
    List[Tuple[str, str]],
]:
    """Generate random preference ballots for 3 factions.

    Returns:
        Tuple of (ballots, faction_display_info).
    """
    rng = random.Random()
    ballots = []
    factions_info = []
    for faction_name in FACTIONS:
        ranking = PLANS.copy()
        rng.shuffle(ranking)
        weight = rng.randint(2, 6)
        ballots.append((ranking, weight))
        pref_str = " > ".join(ranking)
        factions_info.append(
            (f"{faction_name} ({weight}票)", pref_str)
        )
    return ballots, factions_info


def render_level3() -> None:
    """Render Level 3 — Voting Theory / MCDA."""
    st.markdown(
        """
        殖民地议会有三个派系，需要从三个发展方案中选出一个
        执行。不同的投票规则可能产生 **不同的赢家** ——
        这就是著名的 **投票悖论**。

        选择一个场景，然后用不同的投票方法观察结果变化。
        """
    )

    # --- Sidebar: scenario selection -------------------------
    st.sidebar.header("🏛️ L3 议会投票")

    scenario_names = [s["name"] for s in VOTING_SCENARIOS]
    chosen_idx = st.sidebar.radio(
        "选择场景",
        options=range(len(scenario_names)),
        format_func=lambda i: scenario_names[i],
        key="l3_scenario",
    )
    scenario = VOTING_SCENARIOS[chosen_idx]

    if st.sidebar.button(
        "🎲 刷新随机场景",
        use_container_width=True,
        key="l3_refresh",
    ):
        st.session_state.voting_ballots = None
        st.session_state.voting_factions = None
        st.rerun()

    # Resolve ballots
    if scenario["ballots"] is not None:
        ballots = scenario["ballots"]
        factions_info = scenario["factions"]
    else:
        if st.session_state.voting_ballots is None:
            b, f = _generate_random_ballots()
            st.session_state.voting_ballots = b
            st.session_state.voting_factions = f
        ballots = st.session_state.voting_ballots
        factions_info = st.session_state.voting_factions

    # --- Display preference table ----------------------------
    st.markdown(f"**场景: {scenario['name']}**")
    st.markdown(f"*{scenario['description']}*")
    st.markdown("#### 🗳️ 派系偏好")

    st.table({
        "派系": [f[0] for f in factions_info],
        "偏好排序": [f[1] for f in factions_info],
    })

    # --- Voting buttons in columns ---------------------------
    st.markdown("#### 🗳️ 选择投票方法")
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        run_plurality = st.button(
            "📊 多数制 (Plurality)",
            use_container_width=True,
            key="l3_plurality",
        )
    with btn2:
        run_borda = st.button(
            "📊 波达计数 (Borda)",
            use_container_width=True,
            key="l3_borda",
        )
    with btn3:
        run_condorcet = st.button(
            "📊 孔多塞 (Condorcet)",
            use_container_width=True,
            key="l3_condorcet",
        )

    run_all = st.button(
        "⚡ 同时运行所有方法 — 展示投票悖论",
        use_container_width=True,
        key="l3_all",
    )

    # --- Results display -------------------------------------
    if run_plurality or run_all:
        _show_plurality(ballots)

    if run_borda or run_all:
        _show_borda(ballots)

    if run_condorcet or run_all:
        _show_condorcet(ballots)

    if run_all:
        _detect_paradox(ballots)


# ----------------------------------------------------------------
# Voting result renderers
# ----------------------------------------------------------------

def _show_plurality(
    ballots: List[Tuple[List[str], int]],
) -> None:
    """Display Plurality voting results."""
    result = VotingSystem.plurality(ballots)
    st.markdown("---")
    st.markdown("### 📊 多数制 (Plurality) 结果")
    st.markdown(
        "每个派系的 **第一选择** 获得全部票数。"
    )

    col_s, col_c = st.columns([1, 1])
    with col_s:
        for c in sorted(result.scores.keys()):
            bar = "█" * result.scores[c]
            st.markdown(
                f"**{c}**: {result.scores[c]} 票 "
                f"`{bar}`"
            )
    with col_c:
        fig = _bar_chart(
            result.scores, "多数制得分", "#3498db",
        )
        st.pyplot(fig)

    st.success(f"🏆 多数制赢家: **{result.winner}**")


def _show_borda(
    ballots: List[Tuple[List[str], int]],
) -> None:
    """Display Borda Count voting results."""
    result = VotingSystem.borda_count(ballots)
    st.markdown("---")
    st.markdown("### 📊 波达计数 (Borda Count) 结果")
    st.markdown(
        "第 1 名得 2 分，第 2 名得 1 分，"
        "第 3 名得 0 分（乘以派系票数）。"
    )

    col_s, col_c = st.columns([1, 1])
    with col_s:
        for c in sorted(result.scores.keys()):
            st.markdown(
                f"**{c}**: {result.scores[c]:.0f} 分"
            )
    with col_c:
        fig = _bar_chart(
            {k: int(v) for k, v in result.scores.items()},
            "波达计数得分", "#2ecc71",
        )
        st.pyplot(fig)

    st.success(f"🏆 波达计数赢家: **{result.winner}**")


def _show_condorcet(
    ballots: List[Tuple[List[str], int]],
) -> None:
    """Display Condorcet pairwise comparison results."""
    result = VotingSystem.condorcet(ballots)
    st.markdown("---")
    st.markdown("### 📊 孔多塞 (Condorcet) 结果")
    st.markdown(
        "每两个方案进行一对一比较，"
        "看哪个方案能击败所有对手。"
    )

    # Pairwise comparison table
    candidates = sorted({
        c for ranking, _ in ballots for c in ranking
    })
    st.markdown("**两两对决矩阵**（行击败列的票数）:")

    header = [""] + candidates
    rows = []
    for a in candidates:
        row = [f"**{a}**"]
        for b in candidates:
            if a == b:
                row.append("—")
            else:
                wins = result.pairwise_wins.get(
                    (a, b), 0
                )
                loses = result.pairwise_wins.get(
                    (b, a), 0
                )
                marker = "✅" if wins > loses else "❌"
                row.append(f"{wins} {marker}")
        rows.append(row)

    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(
        ["---"] * len(header)
    ) + " |\n"
    for row in rows:
        md += "| " + " | ".join(row) + " |\n"
    st.markdown(md)

    if result.winner:
        st.success(
            f"🏆 孔多塞赢家: **{result.winner}** "
            f"(击败所有对手)"
        )
    else:
        st.error(
            "🔄 **孔多塞悖论！** 不存在能击败所有对手的"
            "方案 — 出现投票循环！"
        )
        if result.cycle_description:
            st.warning(
                f"循环: {result.cycle_description}"
            )


def _detect_paradox(
    ballots: List[Tuple[List[str], int]],
) -> None:
    """Compare winners across methods; highlight paradoxes."""
    plur = VotingSystem.plurality(ballots)
    borda = VotingSystem.borda_count(ballots)
    cond = VotingSystem.condorcet(ballots)

    winners = {
        "多数制": plur.winner,
        "波达计数": borda.winner,
        "孔多塞": (
            cond.winner if cond.winner else "无 (循环)"
        ),
    }

    st.markdown("---")
    st.markdown("### 🔍 投票悖论分析")

    unique = set(winners.values())
    if len(unique) == 1 and "无 (循环)" not in unique:
        st.info(
            f"三种方法产生了 **相同的赢家**: "
            f"**{list(unique)[0]}** — 没有悖论。"
        )
    else:
        st.warning(
            "⚠️ **发现投票悖论！** "
            "不同规则产生了不同赢家："
        )
        for method, winner in winners.items():
            st.markdown(f"- **{method}** → {winner}")
        st.markdown(
            "\n> 这印证了 **Arrow 不可能定理**：没有一种"
            "排序投票制度能同时满足所有公平性标准。"
        )


def _bar_chart(
    scores: Dict[str, int],
    title: str,
    color: str,
) -> plt.Figure:
    """Create a horizontal bar chart for vote scores.

    Args:
        scores: Mapping candidate → score.
        title: Chart title.
        color: Bar colour.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(4, 2.5))
    candidates = sorted(scores.keys())
    values = [scores[c] for c in candidates]

    bars = ax.barh(candidates, values, color=color, alpha=0.8)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("得分")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ################################################################
#  Main entry point
# ################################################################

def main() -> None:
    """Entry point — tab navigation across three levels."""
    st.set_page_config(
        page_title="Aion's Edge",
        page_icon="🚀",
        layout="wide",
    )

    _init_session_state()

    st.title("🚀 Aion's Edge: The Optimization Frontier")

    tab1, tab2, tab3 = st.tabs([
        "🔋 Level 1 — 线性规划",
        "🔬 Level 2 — 多目标优化",
        "🏛️ Level 3 — 议会投票",
    ])

    with tab1:
        render_level1()

    with tab2:
        render_level2()

    with tab3:
        render_level3()


if __name__ == "__main__":
    main()
