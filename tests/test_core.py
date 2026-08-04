"""Core algorithm tests for geographically-weighted-regression."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conftest import mod


class TestKernels:
    def test_bisquare_at_zero(self):
        assert mod.bisquare_kernel(np.array([0.0]), 1.0)[0] == 1.0

    def test_bisquare_cutoff(self):
        """超出带宽权重为 0。"""
        w = mod.bisquare_kernel(np.array([0.5, 1.0, 1.5]), 1.0)
        assert w[0] > 0
        assert w[1] == 0.0
        assert w[2] == 0.0

    def test_bisquare_monotonic(self):
        d = np.linspace(0, 0.99, 20)
        w = mod.bisquare_kernel(d, 1.0)
        assert np.all(np.diff(w) < 0)

    def test_gaussian_positive_always(self):
        w = mod.gaussian_kernel(np.array([0.0, 1.0, 10.0]), 1.0)
        assert np.all(w > 0)
        assert w[0] == 1.0


class TestGWRFit:
    def test_global_linear_constant_coefficients(self):
        """全局线性关系 + 大带宽 → 局部系数应接近全局 OLS 系数。"""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 1, (80, 2))
        x1 = rng.normal(0, 1, 80)
        X = np.column_stack([np.ones(80), x1])
        y = 2.0 + 3.0 * x1 + rng.normal(0, 0.1, 80)
        res = mod.gwr_fit(coords, X, y, bandwidth=2.0, kernel="gaussian")
        # 大带宽 → 系数接近全局
        np.testing.assert_allclose(res["local_beta"][:, 1], 3.0, atol=0.3)
        assert res["r2"] > 0.9

    def test_local_coefficients_track_truth(self):
        """合成空间中变系数应被 GWR 捕捉：西侧系数低、东侧高。"""
        coords, X, y, beta1_true, _ = mod.generate_synthetic(
            [0, 0, 1, 1], n_points=200, seed=3)
        res = mod.gwr_fit(coords, X, y, bandwidth=0.3, kernel="gaussian")
        b1 = res["local_beta"][:, 1]
        west = b1[coords[:, 0] < 0.3].mean()
        east = b1[coords[:, 0] > 0.7].mean()
        assert east - west > 1.0  # 真值差 2.5*0.4=1.0+

    def test_local_r2_range(self):
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 1, (60, 2))
        X = np.column_stack([np.ones(60), rng.normal(0, 1, 60)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.2, 60)
        res = mod.gwr_fit(coords, X, y, bandwidth=0.4)
        assert res["local_r2"].min() >= 0
        assert res["local_r2"].max() <= 1

    def test_shapes(self):
        rng = np.random.default_rng(2)
        coords = rng.uniform(0, 1, (50, 2))
        X = np.column_stack([np.ones(50), rng.normal(0, 1, 50)])
        y = rng.normal(0, 1, 50)
        res = mod.gwr_fit(coords, X, y, bandwidth=0.5)
        assert res["local_beta"].shape == (50, 2)
        assert res["yhat"].shape == (50,)
        assert res["resid"].shape == (50,)


class TestBandwidthSelection:
    def test_returns_valid_bandwidth(self):
        coords, X, y, _, _ = mod.generate_synthetic([0, 0, 1, 1], n_points=100, seed=5)
        candidates = np.array([0.1, 0.3, 0.6, 1.0])
        best, records = mod.select_bandwidth(coords, X, y, candidates)
        assert best in candidates.tolist()
        assert len(records) == 4
        for rec in records:
            assert np.isfinite(rec["aicc"])

    def test_smooth_data_prefers_larger_bandwidth(self):
        """全局平滑关系（无空间变化）→ 大带宽 AICc 更优。"""
        rng = np.random.default_rng(6)
        coords = rng.uniform(0, 1, (120, 2))
        x1 = rng.normal(0, 1, 120)
        X = np.column_stack([np.ones(120), x1])
        y = 1.0 + 2.0 * x1 + rng.normal(0, 0.2, 120)
        best, _ = mod.select_bandwidth(coords, X, y, np.array([0.05, 0.1, 0.5, 1.0]),
                                       kernel="gaussian")
        assert best >= 0.5


class TestSynthetic:
    def test_shapes_and_truth(self):
        coords, X, y, beta1, info = mod.generate_synthetic([116, 39, 117, 40], n_points=80)
        assert coords.shape == (80, 2)
        assert X.shape == (80, 2)
        assert beta1.shape == (80,)
        assert beta1.min() < 1.5 < beta1.max()
