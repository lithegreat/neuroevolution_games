"""OptimizationEngine — Math Engine for Aion's Edge.

This module provides the three mathematical cores that drive the
game's decision-making mechanics:

1. **LPSolver** — Linear Programming (Level 1: The Survival Equation)
2. **MOOSolver** — Multi-Objective Optimization / Pareto analysis
   (Level 2: The Tri-Lemma)
3. **VotingSystem** — Voting Theory & MCDA
   (Level 3: The Council of Factions)

All solvers are stateless utility classes: create an instance, feed
it data, get results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, linprog


# ================================================================
# 1. LPSolver — Linear Programming
# ================================================================

@dataclass
class LPResult:
    """Container for a linear-programming solution.

    Attributes:
        optimal_value: The optimised objective function value.
        solution: The decision-variable vector at the optimum.
        success: Whether the solver converged.
        message: Human-readable solver status string.
    """

    optimal_value: float
    solution: NDArray[np.float64]
    success: bool
    message: str


class LPSolver:
    """Solve linear programmes of the form:

        maximise / minimise  c^T x
        subject to           A_ub @ x <= b_ub
                             A_eq @ x == b_eq   (optional)
                             bounds on each x_i  (optional)

    Internally delegates to ``scipy.optimize.linprog`` which
    expects a *minimisation* problem.  When the caller requests
    maximisation we negate c, solve, then negate the objective
    value back.

    Mathematical background
    -----------------------
    A linear programme (LP) seeks the best outcome in a
    mathematical model whose requirements are represented by
    linear relationships.  The feasible region is a convex
    polytope, and the Fundamental Theorem of LP guarantees
    that the optimum (if it exists) is at a vertex of this
    polytope.
    """

    @staticmethod
    def solve(
        c: NDArray[np.float64],
        A_ub: NDArray[np.float64],
        b_ub: NDArray[np.float64],
        A_eq: Optional[NDArray[np.float64]] = None,
        b_eq: Optional[NDArray[np.float64]] = None,
        bounds: Optional[List[Tuple[Optional[float],
                                    Optional[float]]]] = None,
        maximize: bool = True,
    ) -> LPResult:
        """Solve an LP and return structured results.

        Args:
            c: Coefficient vector of the objective function
                (length n).
            A_ub: Inequality constraint matrix (m × n).
                Each row represents one ``<=`` constraint.
            b_ub: Right-hand side of inequality constraints
                (length m).
            A_eq: Equality constraint matrix (optional).
            b_eq: Right-hand side of equality constraints
                (optional).
            bounds: Sequence of ``(min, max)`` pairs for each
                decision variable.  ``None`` entries default to
                ``(0, None)`` (non-negative).
            maximize: If ``True`` (default), *maximise* c^T x.
                Otherwise *minimise*.

        Returns:
            An ``LPResult`` with the optimal value, solution
            vector, success flag, and solver message.

        Raises:
            ValueError: If the dimensions of *c*, *A_ub*, and
                *b_ub* are inconsistent.

        Example — Aion's Edge Turn 1::

            # Maximise Z = 3x1 + 8x2 + 5x3
            c = np.array([3, 8, 5])
            A_ub = np.array([
                [2, 5, 3],   # Energy constraint
                [1, 2, 4],   # Labour constraint
                [1, 1, 1],   # Storage constraint
            ])
            b_ub = np.array([100, 80, 40])
            bounds = [(50, None), (0, None), (0, None)]
            result = LPSolver.solve(c, A_ub, b_ub,
                                    bounds=bounds)
        """
        c = np.asarray(c, dtype=np.float64)
        A_ub = np.asarray(A_ub, dtype=np.float64)
        b_ub = np.asarray(b_ub, dtype=np.float64)

        # --- dimension checks --------------------------------
        if A_ub.ndim != 2:
            raise ValueError(
                "A_ub must be a 2-D matrix."
            )
        if c.shape[0] != A_ub.shape[1]:
            raise ValueError(
                f"Dimension mismatch: c has {c.shape[0]} "
                f"variables but A_ub has "
                f"{A_ub.shape[1]} columns."
            )
        if A_ub.shape[0] != b_ub.shape[0]:
            raise ValueError(
                f"Dimension mismatch: A_ub has "
                f"{A_ub.shape[0]} rows but b_ub has "
                f"{b_ub.shape[0]} entries."
            )

        # scipy.linprog always *minimises*.
        # To maximise c^T x we minimise (-c)^T x, then negate
        # the objective value in the result.
        c_internal = -c if maximize else c

        res: OptimizeResult = linprog(
            c_internal,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        optimal_value = (
            -res.fun if (maximize and res.success) else res.fun
        )
        return LPResult(
            optimal_value=float(optimal_value),
            solution=np.array(res.x)
            if res.x is not None
            else np.array([]),
            success=bool(res.success),
            message=str(res.message),
        )


# ================================================================
# 2. MOOSolver — Multi-Objective Optimisation / Pareto Analysis
# ================================================================

@dataclass
class ParetoResult:
    """Container for Pareto-front analysis.

    Attributes:
        pareto_front: Indices of non-dominated solutions.
        dominated: Indices of dominated solutions.
        pareto_points: The actual objective-space points on
            the front.
    """

    pareto_front: List[int]
    dominated: List[int]
    pareto_points: NDArray[np.float64]


class MOOSolver:
    """Identify the Pareto front in a set of solutions.

    Mathematical background
    -----------------------
    In multi-objective optimisation (MOO) we rarely have a
    single "best" solution.  Instead, we seek the **Pareto
    front** — the set of *non-dominated* solutions.

    A solution **a** *dominates* **b** if it is at least as
    good on every objective and strictly better on at least
    one (assuming maximisation for all objectives here; the
    caller can negate minimisation objectives before passing
    them in).

    The **Nadir point** is the vector of worst objective
    values among the Pareto-optimal set; it represents the
    "doomsday" scenario the colony must avoid.
    """

    @staticmethod
    def find_pareto_front(
        solutions: NDArray[np.float64],
        maximize: Optional[List[bool]] = None,
    ) -> ParetoResult:
        """Partition *solutions* into Pareto-front and
        dominated sets.

        Args:
            solutions: 2-D array of shape (n_solutions,
                n_objectives).  Each row is one candidate
                solution evaluated on every objective.
            maximize: Per-objective direction flag.  ``True``
                means *higher is better*; ``False`` means
                *lower is better*.  Defaults to all-``True``
                (maximise every objective).

        Returns:
            A ``ParetoResult`` with index lists and the
            front's objective-space points.

        Algorithm
        ---------
        For each candidate *i* we check whether any other
        candidate *j* dominates it (j >= i on all objectives
        **and** j > i on at least one).  If no such *j*
        exists, *i* is non-dominated.

        Complexity: O(n² · m) where n = number of solutions,
        m = number of objectives.
        """
        solutions = np.asarray(solutions, dtype=np.float64)
        if solutions.ndim != 2:
            raise ValueError(
                "solutions must be a 2-D array "
                "(n_solutions × n_objectives)."
            )

        n_solutions, n_objectives = solutions.shape

        # Normalise direction: flip sign on objectives we
        # want to minimise so that "higher = better" for all.
        if maximize is None:
            maximize = [True] * n_objectives
        if len(maximize) != n_objectives:
            raise ValueError(
                f"maximize list length ({len(maximize)}) "
                f"must match n_objectives "
                f"({n_objectives})."
            )

        adjusted = solutions.copy()
        for idx, is_max in enumerate(maximize):
            if not is_max:
                adjusted[:, idx] *= -1

        # --- dominance check ---------------------------------
        pareto_mask = np.ones(n_solutions, dtype=bool)
        for i in range(n_solutions):
            if not pareto_mask[i]:
                continue
            for j in range(n_solutions):
                if i == j:
                    continue
                # Does j dominate i?
                # j >= i on all AND j > i on at least one
                if (
                    np.all(adjusted[j] >= adjusted[i])
                    and np.any(adjusted[j] > adjusted[i])
                ):
                    pareto_mask[i] = False
                    break

        pareto_indices = [
            int(i) for i in range(n_solutions)
            if pareto_mask[i]
        ]
        dominated_indices = [
            int(i) for i in range(n_solutions)
            if not pareto_mask[i]
        ]

        return ParetoResult(
            pareto_front=pareto_indices,
            dominated=dominated_indices,
            pareto_points=solutions[pareto_mask],
        )

    @staticmethod
    def is_dominated(
        candidate: NDArray[np.float64],
        reference_set: NDArray[np.float64],
        maximize: Optional[List[bool]] = None,
    ) -> bool:
        """Check whether a single *candidate* is dominated
        by any member of *reference_set*.

        Useful for the "Dominated Trap" mechanic — quickly
        rejecting scam deals offered to the player.

        Args:
            candidate: 1-D objective vector for one solution.
            reference_set: 2-D array of reference solutions.
            maximize: Per-objective direction flags.

        Returns:
            ``True`` if *candidate* is dominated by at least
            one solution in *reference_set*.
        """
        candidate = np.asarray(candidate, dtype=np.float64)
        reference_set = np.asarray(
            reference_set, dtype=np.float64
        )
        n_objectives = candidate.shape[0]

        if maximize is None:
            maximize = [True] * n_objectives

        adj_candidate = candidate.copy()
        adj_ref = reference_set.copy()
        for idx, is_max in enumerate(maximize):
            if not is_max:
                adj_candidate[idx] *= -1
                adj_ref[:, idx] *= -1

        for j in range(adj_ref.shape[0]):
            if (
                np.all(adj_ref[j] >= adj_candidate)
                and np.any(adj_ref[j] > adj_candidate)
            ):
                return True
        return False

    @staticmethod
    def nadir_point(
        pareto_points: NDArray[np.float64],
        maximize: Optional[List[bool]] = None,
    ) -> NDArray[np.float64]:
        """Compute the Nadir point of a Pareto front.

        The Nadir point contains the *worst* objective value
        among non-dominated solutions for each objective.
        In Aion's Edge the "Nadir Crisis" boss raises this
        point — the colony must keep its actual performance
        away from it.

        Args:
            pareto_points: 2-D array (n_pareto × n_obj).
            maximize: Direction flags per objective.

        Returns:
            1-D Nadir vector of length n_objectives.
        """
        pareto_points = np.asarray(
            pareto_points, dtype=np.float64
        )
        if maximize is None:
            maximize = [True] * pareto_points.shape[1]

        nadir = np.empty(pareto_points.shape[1])
        for idx, is_max in enumerate(maximize):
            if is_max:
                # worst = smallest value among front
                nadir[idx] = pareto_points[:, idx].min()
            else:
                # worst = largest value among front
                nadir[idx] = pareto_points[:, idx].max()
        return nadir


# ================================================================
# 3. VotingSystem — Voting Theory & MCDA
# ================================================================

@dataclass
class PluralityResult:
    """Result of a Plurality vote.

    Attributes:
        scores: Mapping candidate → number of first-place
            votes.
        winner: The candidate with the most first-place votes
            (ties broken alphabetically).
    """

    scores: Dict[str, int]
    winner: str


@dataclass
class BordaResult:
    """Result of a Borda Count vote.

    Attributes:
        scores: Mapping candidate → Borda score.
        winner: The candidate with the highest Borda score
            (ties broken alphabetically).
    """

    scores: Dict[str, float]
    winner: str


@dataclass
class CondorcetResult:
    """Result of a Condorcet pairwise comparison.

    Attributes:
        pairwise_wins: Dict mapping ``(a, b)`` → number of
            voters who prefer *a* over *b*.
        winner: The Condorcet winner, or ``None`` if a cycle
            exists.
        has_cycle: ``True`` when no candidate beats every
            other candidate pairwise (the Condorcet Paradox).
        cycle_description: Human-readable description of the
            cycle (empty string when no cycle).
    """

    pairwise_wins: Dict[Tuple[str, str], int]
    winner: Optional[str]
    has_cycle: bool
    cycle_description: str


class VotingSystem:
    """Implements three classical voting methods.

    Mathematical background
    -----------------------
    Voting theory studies how individual preferences aggregate
    into a collective decision.  Arrow's Impossibility Theorem
    (1951) shows that no rank-order voting system can satisfy
    all fairness criteria simultaneously — which is precisely
    the drama of Aion's Edge Level 3.

    Input format
    ------------
    A preference profile is a list of ballots.  Each ballot is
    an **ordered list of candidate names** from most preferred
    to least preferred.  Optionally each ballot carries a
    *weight* (number of voters sharing that ordering).

    Example::

        ballots = [
            (["A", "B", "C"], 4),   # 4 Miners
            (["B", "C", "A"], 3),   # 3 Terra-formers
            (["C", "A", "B"], 2),   # 2 Families
        ]
    """

    # ----------------------------------------------------------
    # Plurality (First-Past-The-Post)
    # ----------------------------------------------------------
    @staticmethod
    def plurality(
        ballots: List[Tuple[List[str], int]],
    ) -> PluralityResult:
        """Count first-place votes for each candidate.

        Each voter's *top-ranked* candidate receives one
        point (times the ballot weight).

        Args:
            ballots: List of ``(ranking, weight)`` pairs.

        Returns:
            ``PluralityResult`` with scores and winner.
        """
        scores: Dict[str, int] = {}
        for ranking, weight in ballots:
            top = ranking[0]
            scores[top] = scores.get(top, 0) + weight

        # Ensure every candidate appearing anywhere gets a
        # score entry (even if zero first-place votes).
        all_candidates = {
            c for ranking, _ in ballots for c in ranking
        }
        for c in all_candidates:
            scores.setdefault(c, 0)

        winner = max(
            sorted(scores.keys()),
            key=lambda c: scores[c],
        )
        return PluralityResult(
            scores=scores, winner=winner
        )

    # ----------------------------------------------------------
    # Borda Count
    # ----------------------------------------------------------
    @staticmethod
    def borda_count(
        ballots: List[Tuple[List[str], int]],
    ) -> BordaResult:
        """Compute Borda scores.

        With *m* candidates the first-ranked candidate
        receives (m − 1) points, the second receives
        (m − 2), … , the last receives 0.  Points are
        multiplied by ballot weight.

        Args:
            ballots: List of ``(ranking, weight)`` pairs.

        Returns:
            ``BordaResult`` with scores and winner.
        """
        all_candidates = {
            c for ranking, _ in ballots for c in ranking
        }
        m = len(all_candidates)
        scores: Dict[str, float] = {
            c: 0.0 for c in all_candidates
        }

        for ranking, weight in ballots:
            for position, candidate in enumerate(ranking):
                # position 0 → m-1 points, etc.
                scores[candidate] += (m - 1 - position) * weight

        winner = max(
            sorted(scores.keys()),
            key=lambda c: scores[c],
        )
        return BordaResult(
            scores=scores, winner=winner
        )

    # ----------------------------------------------------------
    # Condorcet Method (Pairwise Majority)
    # ----------------------------------------------------------
    @staticmethod
    def condorcet(
        ballots: List[Tuple[List[str], int]],
    ) -> CondorcetResult:
        """Run pairwise Condorcet comparisons.

        For every pair (a, b) we count how many voters
        prefer a over b.  A **Condorcet winner** beats every
        other candidate in pairwise comparison.  If no such
        candidate exists, a *cycle* is reported — this is the
        famous **Condorcet Paradox** that triggers a deadlock
        in the Colony Council.

        Cycle detection
        ---------------
        After building the pairwise-wins matrix, we attempt
        to find a candidate who beats all others.  If none
        exists, we trace the cycle by following the "beats"
        relation to produce a readable description such as
        ``A > B > C > A``.

        Args:
            ballots: List of ``(ranking, weight)`` pairs.

        Returns:
            ``CondorcetResult`` with pairwise wins, optional
            winner, cycle flag, and cycle description.
        """
        all_candidates = sorted({
            c for ranking, _ in ballots for c in ranking
        })
        n = len(all_candidates)
        idx = {c: i for i, c in enumerate(all_candidates)}

        # pairwise[i][j] = number of voters preferring
        #                   candidate i over candidate j
        pairwise = np.zeros((n, n), dtype=np.int64)

        for ranking, weight in ballots:
            for i, higher in enumerate(ranking):
                for lower in ranking[i + 1:]:
                    pairwise[idx[higher]][idx[lower]] += (
                        weight
                    )

        # Build convenient dict of pairwise wins
        pairwise_wins: Dict[Tuple[str, str], int] = {}
        for a in all_candidates:
            for b in all_candidates:
                if a != b:
                    pairwise_wins[(a, b)] = int(
                        pairwise[idx[a]][idx[b]]
                    )

        # --- find Condorcet winner ---------------------------
        # A Condorcet winner beats every other candidate in a
        # head-to-head comparison.
        condorcet_winner: Optional[str] = None
        for a in all_candidates:
            if all(
                pairwise[idx[a]][idx[b]]
                > pairwise[idx[b]][idx[a]]
                for b in all_candidates
                if b != a
            ):
                condorcet_winner = a
                break

        # --- cycle detection ---------------------------------
        has_cycle = False
        cycle_description = ""

        if condorcet_winner is None and n >= 2:
            has_cycle = True
            # Trace cycle: starting from the first candidate,
            # follow the "beats" relation.
            cycle = VotingSystem._trace_cycle(
                all_candidates, pairwise, idx
            )
            cycle_description = " > ".join(cycle)

        return CondorcetResult(
            pairwise_wins=pairwise_wins,
            winner=condorcet_winner,
            has_cycle=has_cycle,
            cycle_description=cycle_description,
        )

    @staticmethod
    def _trace_cycle(
        candidates: List[str],
        pairwise: NDArray[np.int64],
        idx: Dict[str, int],
    ) -> List[str]:
        """Trace a cycle in the pairwise-beats relation.

        Starting from the first candidate we greedily follow
        the "beats" edges until we revisit a candidate, then
        extract the cycle sub-sequence.

        Args:
            candidates: Sorted candidate name list.
            pairwise: Pairwise wins matrix (n × n).
            idx: Mapping candidate name → matrix index.

        Returns:
            List like ``["A", "B", "C", "A"]`` showing the
            cycle.
        """
        # Build "beats" adjacency: a beats b if
        # pairwise[a][b] > pairwise[b][a]
        beats: Dict[str, List[str]] = {
            c: [] for c in candidates
        }
        for a in candidates:
            for b in candidates:
                if a != b:
                    if (
                        pairwise[idx[a]][idx[b]]
                        > pairwise[idx[b]][idx[a]]
                    ):
                        beats[a].append(b)

        # Walk until we revisit a node
        visited: List[str] = []
        visited_set: set[str] = set()
        current = candidates[0]

        while current not in visited_set:
            visited.append(current)
            visited_set.add(current)
            if beats[current]:
                current = beats[current][0]
            else:
                break

        # Extract the cycle portion
        if current in visited_set:
            start = visited.index(current)
            cycle = visited[start:] + [current]
        else:
            cycle = visited + [visited[0]]

        return cycle
