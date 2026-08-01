from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hqml_spectral_qaoa import experiment


class ReproductionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = experiment.reproduce()
        cls.actual = experiment.rounded(cls.raw)
        cls.results = Path(experiment.PACKAGE_RESULTS)

    def test_locked_results_reproduce(self) -> None:
        expected = json.loads((self.results / "results.json").read_text())
        experiment._compare(self.actual, expected)

    def test_authenticated_source_contract(self) -> None:
        self.assertEqual(
            self.actual["dataset_sha256"],
            "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665",
        )
        self.assertEqual(self.actual["records"], 800)
        self.assertEqual(self.actual["edges"], 19176)
        self.assertEqual(self.actual["unique_edge_weights"], [1.0])

    def test_parser_rejects_self_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad"
            path.write_text("2 1\n1 1 1\n")
            with self.assertRaisesRegex(ValueError, "self-loop"):
                experiment.parse_gset(path)

    def test_data_are_not_bundled(self) -> None:
        code_root = Path(experiment.CODE_ROOT)
        self.assertFalse((code_root / "data" / "G1").exists())
        self.assertFalse((code_root / "data" / "source" / "G1").exists())

    def test_isomorphism_classes_and_balancing(self) -> None:
        self.assertEqual(self.actual["isomorphism_class_count"], 16)
        self.assertEqual(
            self.actual["isomorphism_class_sizes"],
            [2, 7, 5, 11, 4, 1, 7, 1, 2, 1, 2, 1, 1, 1, 1, 1],
        )
        self.assertEqual(self.actual["grouped_fold_loads"], [12, 12, 12, 12])

    def test_no_isomorphism_group_crosses_outer_folds(self) -> None:
        ownership = {}
        for fold in self.actual["grouped_folds"]:
            for index in fold["test_indices"]:
                self.assertNotIn(index, ownership)
                ownership[index] = fold["fold"]
        self.assertEqual(set(ownership), set(range(48)))
        for group in self.actual["isomorphism_groups"]:
            self.assertEqual(len({ownership[index] for index in group}), 1)

    def test_old_row_split_collision_counts(self) -> None:
        rows = self.actual["old_row_fold_audit"]
        self.assertEqual([row["topology_matches"] for row in rows], [9, 8, 10, 11])
        self.assertEqual([row["feature_matches"] for row in rows], [9, 9, 10, 11])
        self.assertEqual([row["ordered_edge_repeats"] for row in rows], [2, 2, 2, 5])

    def test_exact_formula_matches_every_statevector_value(self) -> None:
        check = self.actual["formula_validation"]
        self.assertEqual(check["values_checked"], 48 * 81)
        self.assertLessEqual(check["max_absolute_error"], 1e-12)
        self.assertEqual(check["target_quantum_objective_queries"], 0)

    def test_grouped_fold_headline_values(self) -> None:
        expected = [
            (0.006896176937, 0.007144462614, 0.0),
            (0.000970230484, 0.000163452370, 0.000163452370),
            (0.049991766495, 0.029476195506, 0.031484961031),
            (0.005025260511, 0.0, 0.0),
        ]
        for fold, target in zip(self.actual["grouped_folds"], expected):
            actual = tuple(fold["mean_curves"][name][0] for name in ("mean", "ridge", "spectral"))
            np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-12)
            self.assertLessEqual(fold["mean_curves"]["closed_form"][0], 1e-12)

    def test_residual_gate_fails_two_grouped_folds(self) -> None:
        residuals = [
            fold["operator"]["sampled_row_relative_error"]
            for fold in self.actual["grouped_folds"]
        ]
        np.testing.assert_allclose(
            residuals,
            [0.026808925816, 0.020233169524, 0.157186401944, 0.076240732314],
            rtol=1e-8,
            atol=1e-10,
        )
        self.assertEqual(sum(value <= 0.05 for value in residuals), 2)
        self.assertEqual(self.actual["spectral_residual_passes"], 2)

    def test_psd_spike_noncertification_construction(self) -> None:
        for row in self.actual["spike_counterexample"]:
            self.assertEqual(row["sampled_row_operator_error"], 0.0)
            self.assertEqual(row["global_operator_error"], row["tau"])

    def test_singular_values_do_not_identify_ranking(self) -> None:
        left = np.array([[1.0, 0.0]])
        right = np.array([[0.0, 1.0]])
        np.testing.assert_allclose(np.linalg.svd(left, compute_uv=False), np.linalg.svd(right, compute_uv=False))
        self.assertNotEqual(int(np.argmax(left)), int(np.argmax(right)))

    def test_two_epsilon_ranking_bound(self) -> None:
        rng = np.random.default_rng(91)
        for _ in range(200):
            truth = rng.normal(size=17)
            perturbation = rng.uniform(-0.03, 0.03, size=17)
            predicted = truth + perturbation
            regret = truth.max() - truth[int(np.argmax(predicted))]
            self.assertLessEqual(regret, 2 * np.max(np.abs(perturbation)) + 1e-15)

    def test_matched_runtime_scope_and_accuracy_are_recorded(self) -> None:
        rows = self.actual["timing_kernel_runtime"]
        self.assertEqual([row["records"] for row in rows], [100, 200, 400, 800])
        for row in rows:
            self.assertGreater(row["timing_dense_seconds"], 0.0)
            self.assertGreater(row["timing_spectral_seconds"], 0.0)
            self.assertGreater(row["application_relative_error"], 0.0)
        self.assertEqual(
            self.actual["runtime_environment"]["timing_scope"],
            "matched construction plus evaluation/application",
        )

    def test_figure_contract(self) -> None:
        figure_dir = self.results / "figures"
        expected = {
            "collision_audit",
            "formula_validation_runtime",
            "grouped_query_curves",
            "matched_kernel_runtime",
            "residual_noncertification",
            "split_sensitivity",
        }
        self.assertEqual({path.stem for path in figure_dir.glob("*.pdf")}, expected)
        self.assertEqual({path.stem for path in figure_dir.glob("*.png")}, expected)

    def test_manifest_hashes(self) -> None:
        manifest = json.loads((self.results / "manifest.json").read_text())
        for record in manifest["files"]:
            path = self.results / record["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, record["sha256"])
            self.assertEqual(path.stat().st_size, record["bytes"])


if __name__ == "__main__":
    unittest.main()
