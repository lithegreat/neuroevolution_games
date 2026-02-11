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
        title="🌪️ Dust Storm",
        description=(
            "A fierce dust storm shrouds the solar arrays! "
            "Energy capacity drops from 100 to 60."
        ),
        icon="🌪️",
        b_ub_override=np.array([60.0, 80.0]),
    ),
    "flu": GameEvent(
        name="flu",
        title="🤒 Flu Outbreak",
        description=(
            "A widespread flu outbreak hits the colonists! "
            "Available labour falls from 80 to 50."
        ),
        icon="🤒",
        b_ub_override=np.array([100.0, 50.0]),
    ),
    "tech_breakthrough": GameEvent(
        name="tech_breakthrough",
        title="🔬 Tech Breakthrough",
        description=(
            "Research teams achieve a major breakthrough! "
            "All production coefficients are doubled this turn."
        ),
        icon="🔬",
        coeff_multiplier=2.0,
    ),
    "clear": GameEvent(
        name="clear",
        title="☀️ All Clear",
        description="Nothing unusual this turn; systems nominal.",
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
        f"Turn {st.session_state.turn + 1}: {event.title}"
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
            " (Energy)"
        ),
        color="#1f77b4", linewidth=2,
    )

    y_c2 = (b_ub[1] - x_range) / 2
    ax.plot(
        x_range, y_c2,
        label=(
            rf"$C_2$: $x_1 + 2x_2 \leq {b_ub[1]:.0f}$"
            " (Labour)"
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
        alpha=0.45, label="Feasible Region",
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
            f"Player choice ({x1_player:.0f}, {x2_player:.0f})"
        ),
        zorder=5,
    )

    # Optimal point (green star)
    ax.plot(
        x1_opt, x2_opt, "*",
        color="#2ecc71", markersize=20,
        markeredgecolor="black", markeredgewidth=1,
        label=f"Optimal ({x1_opt:.1f}, {x2_opt:.1f})",
        zorder=5,
    )

    # Iso-profit line
    z_player = obj[0] * x1_player + obj[1] * x2_player
    if obj[1] != 0:
        y_iso = (z_player - obj[0] * x_range) / obj[1]
        ax.plot(
            x_range, y_iso, "--",
            color="#9b59b6", linewidth=1, alpha=0.6,
            label=f"Iso-profit line Z = {z_player:.0f}",
        )

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel(r"$x_1$ — Oxygen production", fontsize=12)
    ax.set_ylabel(r"$x_2$ — Food production", fontsize=12)
    ax.set_title(
        f"Aion's Edge · Level 1 — Turn "
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
            f"🎉 Congratulations! You survived {TOTAL_TURNS} turns!\n\n"
            f"Cumulative total output: **{st.session_state.total_score:.0f}**"
        )
        if st.button("🔄 Restart", key="l1_restart"):
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
        You are the colony's central AI **AION**. Adjust production of **Oxygen** ($x_1$)
        and **Food** ($x_2$) to maximise total colony output under limited **Energy**
        and **Labour** constraints:

        $$Z = {obj[0]:.0f}\\,x_1 + {obj[1]:.0f}\\,x_2$$
        """
    )

    # Sidebar controls
    st.sidebar.header("⚙️ L1 Production Control")
    st.sidebar.markdown(
        f"**Turn {st.session_state.turn} / {TOTAL_TURNS}**"
    )

    x1_player = st.sidebar.slider(
        "x₁ — Oxygen production", 0, 60, 20, 1, key="l1_x1",
    )
    x2_player = st.sidebar.slider(
        "x₂ — Food production", 0, 80, 20, 1, key="l1_x2",
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
        "⏭️ Submit and Next Turn",
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
        "📈 Cumulative Total Output",
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
        st.markdown("### 📊 Panel Status")
        if feasible:
            st.success("✅ Production plan is feasible!")
        else:
            st.error("🚨 Not enough resources — plan exceeds constraints!")

        st.metric(
            "Current Total Output Z", f"{z_player:.0f}",
            delta=(
                f"{z_player - opt_value:+.0f} vs Optimal"
                if feasible else "Infeasible"
            ),
        )
        st.metric("Optimal Total Output Z*", f"{opt_value:.0f}")

        st.markdown("---")
        c1_used = 2 * x1_player + x2_player
        c2_used = x1_player + 2 * x2_player
        st.markdown(
            f"**Energy** (C₁): {c1_used:.0f}/{b_ub[0]:.0f}"
        )
        st.progress(min(c1_used / b_ub[0], 1.0))
        st.markdown(
            f"**Labour** (C₂): {c2_used:.0f}/{b_ub[1]:.0f}"
        )
        st.progress(min(c2_used / b_ub[1], 1.0))

        st.markdown("---")
        st.markdown(
            f"**Optimal solution**: $x_1^*={x1_opt:.1f}$, "
            f"$x_2^*={x2_opt:.1f}$"
        )
        if feasible and opt_value > 0:
            eff = z_player / opt_value * 100
            st.markdown(f"**Efficiency**: {eff:.1f}%")
            if eff >= 99.9:
                st.balloons()
                st.success("🎉 Perfect!")
            elif eff >= 90:
                st.info("👍 Very close!")
            elif eff >= 70:
                st.warning("💡 Room for improvement.")
            else:
                st.warning("⚠️ Low output.")

    if st.session_state.event_log:
        with st.expander("📜 Event Log"):
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
        label="Dominated", zorder=3,
    )

    # Pareto front — red, connected by line
    par = solutions[pareto_idx]
    sort_order = np.argsort(par[:, 0])
    par_sorted = par[sort_order]
    ax.scatter(
        par[:, 0], par[:, 1],
        c="#e74c3c", s=80, edgecolors="black",
        linewidths=1,
        label="Pareto Front",
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
            label="Your choice", zorder=5,
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
        "Pollution (lower is better)", fontsize=12,
    )
    ax.set_ylabel(
        "Economic Output (higher is better)", fontsize=12,
    )
    ax.set_title(
        "Aion's Edge · Level 2 — Pareto Front",
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
        The colony must trade off **economic output** against **environmental
        impact**. From the 50 random candidate solutions below, the **red**
        points form the **Pareto Front** — the set of solutions that cannot
        be improved on one objective without worsening another.

        Pick a solution index and the system will tell you whether it is
        Pareto-optimal or dominated.
        """
    )

    # Sidebar controls
    st.sidebar.header("🔬 L2 Pareto Analysis")

    if st.sidebar.button(
        "🎲 Generate New Solutions",
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
        "Select solution index (0–49)",
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
        st.markdown("### 📊 Solution Analysis")

        pt = solutions[selected_idx]
        st.markdown(
            f"**Solution #{selected_idx}**  \n"
            f"Pollution: `{pt[0]:.1f}`  \n"
            f"Economic Output: `{pt[1]:.1f}`"
        )

        is_pareto = (
            selected_idx in pareto_result.pareto_front
        )
        if is_pareto:
            st.success(
                "⭐ Pareto optimal! The solution is not dominated by any other."
            )
        else:
            st.error(
                "❌ Dominated solution! Another solution is at least as good"
                " on all objectives and strictly better on at least one."
            )
            dom_by = _find_dominator(
                selected_idx, solutions,
                pareto_result.pareto_front,
            )
            if dom_by is not None:
                dp = solutions[dom_by]
                st.markdown(
                    f"For example, **Solution #{dom_by}** "
                    f"(pollution `{dp[0]:.1f}`, output `{dp[1]:.1f}`) dominates your choice."
                )

        st.markdown("---")
        st.markdown("### 📈 Pareto Statistics")
        st.markdown(
            f"- Number of Pareto-optimal solutions: "
            f"**{len(pareto_result.pareto_front)}**"
        )
        st.markdown(
            f"- Number of dominated solutions: "
            f"**{len(pareto_result.dominated)}**"
        )

        # Nadir point
        nadir = MOOSolver.nadir_point(
            pareto_result.pareto_points,
            maximize=[False, True],
        )
        st.markdown("---")
        st.markdown("### ⚠️ Nadir point (worst-case)")
        st.markdown(
            f"Highest pollution: `{nadir[0]:.1f}`  \n"
            f"Lowest output: `{nadir[1]:.1f}`"
        )
        st.caption(
            "The Nadir point shows the worst value of each objective on the"
            " Pareto front — avoid this disaster point."
        )


# ################################################################
#  LEVEL 3 — Voting Theory (The Council of Factions)
# ################################################################

# Faction names and candidate plans
FACTIONS = [
    "⛏️ Miners Guild",
    "🌿 Environmentalists",
    "👨\u200d👩\u200d👧 Residents",
]
PLANS = ["Plan A", "Plan B", "Plan C"]

# Pre-built scenarios that guarantee interesting results.
VOTING_SCENARIOS: List[Dict[str, object]] = [
    {
        "name": "Classic Condorcet Paradox",
        "description": (
            "Miners prefer A>B>C, Environmentalists prefer B>C>A,"
            " Residents prefer C>A>B — this produces a cycle!"
        ),
        "ballots": [
            (["Plan A", "Plan B", "Plan C"], 4),
            (["Plan B", "Plan C", "Plan A"], 3),
            (["Plan C", "Plan A", "Plan B"], 2),
        ],
        "factions": [
            ("⛏️ Miners Guild (4 votes)", "A > B > C"),
            ("🌿 Environmentalists (3 votes)", "B > C > A"),
            ("👨\u200d👩\u200d👧 Residents (2 votes)", "C > A > B"),
        ],
    },
    {
        "name": "Plurality vs Borda Divergence",
        "description": (
            "Plurality and Borda can pick different winners!"
        ),
        "ballots": [
            (["Plan A", "Plan B", "Plan C"], 5),
            (["Plan B", "Plan C", "Plan A"], 4),
            (["Plan C", "Plan B", "Plan A"], 3),
        ],
        "factions": [
            ("⛏️ Miners Guild (5 votes)", "A > B > C"),
            ("🌿 Environmentalists (4 votes)", "B > C > A"),
            ("👨\u200d👩\u200d👧 Residents (3 votes)", "C > B > A"),
        ],
    },
    {
        "name": "Random Preferences",
        "description": "Randomly generated faction preferences.",
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
            (f"{faction_name} ({weight} votes)", pref_str)
        )
    return ballots, factions_info


def render_level3() -> None:
    """Render Level 3 — Voting Theory / MCDA."""
    st.markdown(
        """
        The colony parliament has three factions that must choose one development
        plan to implement. Different voting rules can produce **different
        winners** — this is the well-known **voting paradox**.

        Choose a scenario and try different voting methods to see how winners
        change.
        """
    )

    # --- Sidebar: scenario selection -------------------------
    st.sidebar.header("🏛️ L3 Parliamentary Voting")

    scenario_names = [s["name"] for s in VOTING_SCENARIOS]
    chosen_idx = st.sidebar.radio(
        "Choose scenario",
        options=range(len(scenario_names)),
        format_func=lambda i: scenario_names[i],
        key="l3_scenario",
    )
    scenario = VOTING_SCENARIOS[chosen_idx]

    if st.sidebar.button(
        "🎲 Refresh random scenario",
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
    st.markdown(f"**Scenario: {scenario['name']}**")
    st.markdown(f"*{scenario['description']}*")
    st.markdown("#### 🗳️ Faction preferences")

    st.table({
        "Faction": [f[0] for f in factions_info],
        "Preferences": [f[1] for f in factions_info],
    })

    # --- Voting buttons in columns ---------------------------
    st.markdown("#### 🗳️ Choose a voting method")
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        run_plurality = st.button(
            "📊 Plurality",
            use_container_width=True,
            key="l3_plurality",
        )
    with btn2:
        run_borda = st.button(
            "📊 Borda Count",
            use_container_width=True,
            key="l3_borda",
        )
    with btn3:
        run_condorcet = st.button(
            "📊 Condorcet",
            use_container_width=True,
            key="l3_condorcet",
        )

    run_all = st.button(
        "⚡ Run all methods — show voting paradox",
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
    st.markdown("### 📊 Plurality results")
    st.markdown(
        "Each faction's **first choice** receives all of its votes."
    )

    col_s, col_c = st.columns([1, 1])
    with col_s:
        for c in sorted(result.scores.keys()):
            bar = "█" * result.scores[c]
            st.markdown(
                f"**{c}**: {result.scores[c]} votes "
                f"`{bar}`"
            )
    with col_c:
        fig = _bar_chart(
            result.scores, "Plurality scores", "#3498db",
        )
        st.pyplot(fig)
    st.success(f"🏆 Plurality winner: **{result.winner}**")


def _show_borda(
    ballots: List[Tuple[List[str], int]],
) -> None:
    """Display Borda Count voting results."""
    result = VotingSystem.borda_count(ballots)
    st.markdown("---")
    st.markdown("### 📊 Borda Count results")
    st.markdown(
        "1st gets 2 points, 2nd gets 1 point, 3rd gets 0 points (times faction votes)."
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
            "Borda scores", "#2ecc71",
        )
        st.pyplot(fig)
    st.success(f"🏆 Borda winner: **{result.winner}**")


def _show_condorcet(
    ballots: List[Tuple[List[str], int]],
) -> None:
    """Display Condorcet pairwise comparison results."""
    result = VotingSystem.condorcet(ballots)
    st.markdown("---")
    st.markdown("### 📊 Condorcet results")
    st.markdown(
        "Compare each pair of plans head-to-head to see which plan beats all others."
    )

    # Pairwise comparison table
    candidates = sorted({
        c for ranking, _ in ballots for c in ranking
    })
    st.markdown("**Pairwise comparison matrix** (row beats column votes):")

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
            f"🏆 Condorcet winner: **{result.winner}** (beats all opponents)"
        )
    else:
        st.error(
            "🔄 Condorcet paradox! No plan beats all opponents — a voting cycle exists."
        )
        if result.cycle_description:
            st.warning(
                f"Cycle: {result.cycle_description}"
            )


def _detect_paradox(
    ballots: List[Tuple[List[str], int]],
) -> None:
    """Compare winners across methods; highlight paradoxes."""
    plur = VotingSystem.plurality(ballots)
    borda = VotingSystem.borda_count(ballots)
    cond = VotingSystem.condorcet(ballots)

    winners = {
        "Plurality": plur.winner,
        "Borda": borda.winner,
        "Condorcet": (
            cond.winner if cond.winner else "None (cycle)"
        ),
    }

    st.markdown("---")
    st.markdown("### 🔍 Voting paradox analysis")

    unique = set(winners.values())
    if len(unique) == 1 and "None (cycle)" not in unique:
        st.info(
            f"All methods produced the **same winner**: **{list(unique)[0]}** — no paradox."
        )
    else:
        st.warning(
            "⚠️ Voting paradox detected! Different rules produced different winners:"
        )
        for method, winner in winners.items():
            st.markdown(f"- **{method}** → {winner}")
        st.markdown(
            "\n> This illustrates Arrow's impossibility theorem: no ranked voting"
            " system can satisfy all fairness criteria simultaneously."
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
    ax.set_xlabel("Score")
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
        "🔋 Level 1 — Linear Programming",
        "🔬 Level 2 — Multi-Objective Optimisation",
        "🏛️ Level 3 — Parliamentary Voting",
    ])

    with tab1:
        render_level1()

    with tab2:
        render_level2()

    with tab3:
        render_level3()


if __name__ == "__main__":
    main()
