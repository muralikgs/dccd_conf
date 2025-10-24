"""Implementation of the Linear Latent Causal (LLC) algorithm from
Hyttinen, Eberhardt, and Hoyer (2012).

This module follows Algorithm 1 exactly: it inspects the available
intervention experiments, constructs the linear system for the unknown
structural coefficients, and estimates the latent disturbance covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class LLCResult:
    """Container for the outputs of Algorithm 1."""

    adjacency: np.ndarray
    disturbance_covariance: np.ndarray
    unsatisfied_pair_condition: List[Tuple[int, int]]
    unsatisfied_covariance_condition: List[Tuple[int, int]]


class LLCAlgorithm:
    """Learns a linear causal model with latent confounding from interventions."""

    def __init__(self, rank_tol: float = 1e-10):
        self.rank_tol = rank_tol

    def fit(
        self,
        datasets: Sequence[np.ndarray],
        intervention_sets: Sequence[Iterable[int]],
    ) -> LLCResult:
        if len(datasets) != len(intervention_sets):
            raise ValueError("datasets and intervention_sets must have the same length")
        if not datasets:
            raise ValueError("At least one experiment is required")

        n_nodes = self._infer_n_nodes(datasets)
        experiments = self._prepare_experiments(datasets, intervention_sets, n_nodes)
        if not experiments:
            raise ValueError("No usable experiments (empty intervention sets)")

        pair_condition, covariance_condition = self._evaluate_conditions(experiments, n_nodes)
        adjacency = self._estimate_adjacency(experiments, n_nodes, pair_condition)
        disturbance_covariance = self._estimate_disturbance_covariance(
            experiments, adjacency, covariance_condition
        )

        unsatisfied_pairs = [
            (i, j)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j and not pair_condition[i, j]
        ]
        unsatisfied_covariance = [
            (i, j)
            for i in range(n_nodes)
            for j in range(i, n_nodes)
            if not covariance_condition[i, j]
        ]

        return LLCResult(
            adjacency=adjacency,
            disturbance_covariance=disturbance_covariance,
            unsatisfied_pair_condition=unsatisfied_pairs,
            unsatisfied_covariance_condition=unsatisfied_covariance,
        )

    @staticmethod
    def _infer_n_nodes(datasets: Sequence[np.ndarray]) -> int:
        widths = {data.shape[1] for data in datasets}
        if len(widths) != 1:
            raise ValueError("All datasets must have the same number of variables")
        return widths.pop()

    def _prepare_experiments(
        self,
        datasets: Sequence[np.ndarray],
        intervention_sets: Sequence[Iterable[int]],
        n_nodes: int,
    ) -> List[Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]]:
        experiments: List[Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]] = []
        all_nodes = np.arange(n_nodes)
        for data, interventions in zip(datasets, intervention_sets):
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[1] != n_nodes:
                raise ValueError("Each dataset must be two-dimensional with consistent width")
            if data.shape[0] < 2:
                continue

            j_set = tuple(sorted(int(i) for i in interventions))
            u_mask = np.ones(n_nodes, dtype=bool)
            u_mask[list(j_set)] = False
            u_set = tuple(all_nodes[u_mask].tolist())
            if not j_set or not u_set:
                continue

            centered = data - data.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / float(data.shape[0])
            experiments.append((covariance, j_set, u_set))
        return experiments

    def _evaluate_conditions(
        self,
        experiments: Sequence[Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]],
        n_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pair_condition = np.zeros((n_nodes, n_nodes), dtype=bool)
        covariance_condition = np.zeros((n_nodes, n_nodes), dtype=bool)

        for covariance, j_set, u_set in experiments:
            if j_set:
                cov_jj = covariance[np.ix_(j_set, j_set)]
                rank_ok = np.linalg.matrix_rank(cov_jj, tol=self.rank_tol) == len(j_set)
                if rank_ok:
                    for i in j_set:
                        for u in u_set:
                            pair_condition[i, u] = True
            for idx, i in enumerate(u_set):
                covariance_condition[i, i] = True
                for j in u_set[idx + 1 :]:
                    covariance_condition[i, j] = True
                    covariance_condition[j, i] = True

        return pair_condition, covariance_condition

    def _estimate_adjacency(
        self,
        experiments: Sequence[Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]],
        n_nodes: int,
        pair_condition: np.ndarray,
    ) -> np.ndarray:
        adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
        identity = np.eye(n_nodes)

        # Pre-compute total effects for experiments that satisfy the pair condition.
        effect_cache: List[Tuple[Tuple[int, ...], Tuple[int, ...], np.ndarray]] = []
        for covariance, j_set, u_set in experiments:
            cov_jj = covariance[np.ix_(j_set, j_set)]
            if np.linalg.matrix_rank(cov_jj, tol=self.rank_tol) != len(j_set):
                continue
            cov_uj = covariance[np.ix_(u_set, j_set)]
            total_effects = cov_uj / 2.0
            effect_cache.append((j_set, u_set, total_effects))

        for target in range(n_nodes):
            sources = [node for node in range(n_nodes) if node != target]
            if not sources:
                continue

            design_rows: List[np.ndarray] = []
            rhs: List[float] = []

            for j_set, u_set, total_effects in effect_cache:
                if target not in u_set:
                    continue

                u_index = {node: idx for idx, node in enumerate(u_set)}
                j_index = {node: idx for idx, node in enumerate(j_set)}

                for intervention in j_set:
                    if not pair_condition[intervention, target]:
                        continue

                    row = np.zeros(len(sources), dtype=float)
                    for idx, source in enumerate(sources):
                        if source == intervention:
                            row[idx] = 1.0
                        elif source in u_index and source != target:
                            row[idx] = total_effects[u_index[source], j_index[intervention]]
                    design_rows.append(row)
                    rhs.append(total_effects[u_index[target], j_index[intervention]])

            if design_rows:
                A = np.vstack(design_rows)
                b = np.asarray(rhs)
                solution, *_ = np.linalg.lstsq(A, b, rcond=None)
                for coeff, source in zip(solution, sources):
                    adjacency[target, source] = coeff

        # Zero the diagonal explicitly.
        adjacency = adjacency * (1 - identity)
        return adjacency

    def _estimate_disturbance_covariance(
        self,
        experiments: Sequence[Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]],
        adjacency: np.ndarray,
        covariance_condition: np.ndarray,
    ) -> np.ndarray:
        n_nodes = adjacency.shape[0]
        disturbance_sum = np.zeros((n_nodes, n_nodes), dtype=float)
        disturbance_count = np.zeros((n_nodes, n_nodes), dtype=int)
        identity = np.eye(n_nodes)

        for covariance, _, u_set in experiments:
            if not u_set:
                continue
            u_mask = np.zeros((n_nodes, n_nodes), dtype=float)
            idx = np.ix_(u_set, u_set)
            u_mask[idx] = np.eye(len(u_set))
            residual = (identity - u_mask @ adjacency) @ covariance @ (identity - u_mask @ adjacency).T
            for i_pos, i in enumerate(u_set):
                for j_pos, j in enumerate(u_set):
                    disturbance_sum[i, j] += residual[i, j]
                    disturbance_count[i, j] += 1

        disturbance_covariance = np.full((n_nodes, n_nodes), np.nan, dtype=float)
        mask = disturbance_count > 0
        disturbance_covariance[mask] = disturbance_sum[mask] / disturbance_count[mask]

        # Preserve symmetry where defined.
        upper = np.triu_indices(n_nodes, k=1)
        lower = (upper[1], upper[0])
        disturbance_covariance[lower] = disturbance_covariance[upper]

        # For pairs without data keep NaN, the caller can decide how to proceed.
        for i in range(n_nodes):
            if covariance_condition[i, i] and np.isnan(disturbance_covariance[i, i]):
                disturbance_covariance[i, i] = 0.0

        return disturbance_covariance
