"""Tests for dataset drift statistics and domain classifier (M19)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelroot.stats.drift import (
    compute_domain_classifier,
    compute_drift_report,
    compute_univariate_drift,
)


class TestUnivariateDrift:
    """Tests for univariate drift functions."""

    def test_numeric_no_drift(self) -> None:
        rng = np.random.default_rng(2026)
        ref = pd.Series(rng.normal(0, 1, size=500), name="x")
        cur = pd.Series(rng.normal(0, 1, size=500), name="x")

        res = compute_univariate_drift(ref, cur, feature_name="x", is_numeric=True)
        assert res.feature == "x"
        assert res.feature_type == "numeric"
        assert res.statistic_name == "ks"
        assert res.p_value > 0.05
        assert res.psi < 0.10

    def test_numeric_strong_drift(self) -> None:
        rng = np.random.default_rng(2026)
        ref = pd.Series(rng.normal(0, 1, size=500), name="x")
        cur = pd.Series(rng.normal(3, 1, size=500), name="x")  # Large mean shift

        res = compute_univariate_drift(ref, cur, feature_name="x", is_numeric=True)
        assert res.statistic_name == "ks"
        assert res.statistic_value > 0.50
        assert res.p_value < 1e-4
        assert res.psi > 0.50

    def test_categorical_no_drift(self) -> None:
        rng = np.random.default_rng(2026)
        ref = pd.Series(rng.choice(["A", "B", "C"], size=500, p=[0.5, 0.3, 0.2]), name="cat")
        cur = pd.Series(rng.choice(["A", "B", "C"], size=500, p=[0.5, 0.3, 0.2]), name="cat")

        res = compute_univariate_drift(ref, cur, feature_name="cat", is_numeric=False)
        assert res.feature_type == "categorical"
        assert res.statistic_name == "chi2"
        assert res.p_value > 0.01
        assert res.psi < 0.10
        assert res.js_divergence is not None
        assert res.js_divergence < 0.05

    def test_categorical_strong_drift(self) -> None:
        rng = np.random.default_rng(2026)
        ref = pd.Series(rng.choice(["A", "B", "C"], size=500, p=[0.8, 0.1, 0.1]), name="cat")
        cur = pd.Series(rng.choice(["A", "B", "C"], size=500, p=[0.1, 0.1, 0.8]), name="cat")

        res = compute_univariate_drift(ref, cur, feature_name="cat", is_numeric=False)
        assert res.statistic_name == "chi2"
        assert res.p_value < 1e-4
        assert res.psi > 0.50
        assert res.js_divergence is not None
        assert res.js_divergence > 0.20

    def test_missing_rate_tracking(self) -> None:
        ref = pd.Series([1.0, 2.0, np.nan, 4.0], name="m")
        cur = pd.Series([1.0, np.nan, np.nan, np.nan], name="m")

        res = compute_univariate_drift(ref, cur, feature_name="m", is_numeric=True)
        assert abs(res.missing_rate_ref - 0.25) < 1e-5
        assert abs(res.missing_rate_cur - 0.75) < 1e-5


class TestDomainClassifier:
    """Tests for multidimensional domain classifier."""

    def test_domain_classifier_no_drift(self) -> None:
        rng = np.random.default_rng(2026)
        df_ref = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, size=200),
                "f2": rng.normal(0, 1, size=200),
            }
        )
        df_cur = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, size=200),
                "f2": rng.normal(0, 1, size=200),
            }
        )

        res = compute_domain_classifier(
            df_ref, df_cur, seed=2026, n_permutations=20, max_samples_per_domain=200
        )
        assert 0.35 <= res.auc <= 0.65
        assert res.p_value > 0.01

    def test_domain_classifier_strong_drift(self) -> None:
        rng = np.random.default_rng(2026)
        df_ref = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, size=200),
                "f2": rng.normal(0, 1, size=200),
            }
        )
        df_cur = pd.DataFrame(
            {
                "f1": rng.normal(5, 1, size=200),
                "f2": rng.normal(5, 1, size=200),
            }
        )

        res = compute_domain_classifier(
            df_ref, df_cur, seed=2026, n_permutations=20, max_samples_per_domain=200
        )
        assert res.auc >= 0.85
        assert res.p_value <= 0.10

    def test_domain_classifier_consistent_seed_and_refit(self) -> None:
        rng = np.random.default_rng(2026)
        df_ref = pd.DataFrame(
            {
                "f1": rng.normal(0, 1, size=100),
                "f2": rng.normal(0, 1, size=100),
            }
        )
        df_cur = pd.DataFrame(
            {
                "f1": rng.normal(1, 1, size=100),
                "f2": rng.normal(1, 1, size=100),
            }
        )
        # Running twice with same seed must produce identical permuted AUCs
        res1 = compute_domain_classifier(df_ref, df_cur, seed=42, n_permutations=5)
        res2 = compute_domain_classifier(df_ref, df_cur, seed=42, n_permutations=5)
        assert res1.auc == res2.auc
        assert res1.p_value == res2.p_value
        assert res1.permuted_aucs == res2.permuted_aucs
        assert len(res1.permuted_aucs) == 5


class TestDriftReport:
    """Tests for full drift report compilation."""

    def test_full_report_generation(self) -> None:
        rng = np.random.default_rng(2026)
        df_ref = pd.DataFrame(
            {
                "num1": rng.normal(0, 1, size=150),
                "cat1": rng.choice(["low", "high"], size=150),
                "target": rng.choice([0, 1], size=150),
            }
        )
        df_cur = pd.DataFrame(
            {
                "num1": rng.normal(2, 1, size=150),
                "cat1": rng.choice(["low", "high"], size=150),
                "target": rng.choice([0, 1], size=150),
            }
        )

        report = compute_drift_report(
            df_ref, df_cur, target_column="target", seed=2026, n_permutations=10
        )
        assert "target" not in report.univariate
        assert "num1" in report.univariate
        assert "cat1" in report.univariate
        assert report.top_drifted_features[0] == "num1"

        data_dict = report.to_dict()
        assert "domain_classifier" in data_dict
        assert "univariate_drift" in data_dict
