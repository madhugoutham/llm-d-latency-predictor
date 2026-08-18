#!/usr/bin/env python3
"""Unit tests for common/topology_correction.py and its wiring into the
training and prediction servers.

Core invariant under test: when topology_distance is absent (the case for
every existing caller today), correction_for() must return exactly 0.0 --
a complete no-op. This is what makes the feature safe to add without
breaking any current behavior while it stays disabled by default.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.topology_correction import TopologyCorrectionTable  # noqa: E402


def _model_factory():
    return xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=0)


class TestTopologyCorrectionTable:
    def test_correction_for_none_is_always_zero(self):
        table = TopologyCorrectionTable(quantile_alpha=0.9)
        table.corrections = {"zone": 42.0, "region": -13.0}
        assert table.correction_for(None) == 0.0

    def test_correction_for_unknown_class_is_zero(self):
        table = TopologyCorrectionTable(quantile_alpha=0.9)
        table.corrections = {"zone": 42.0}
        assert table.correction_for("region") == 0.0
        assert table.correction_for("nonexistent_class") == 0.0

    def test_empty_table_is_always_a_no_op(self):
        """An unfitted table (e.g. feature disabled, or fit() never called) must
        never raise and must always return 0.0, for any input."""
        table = TopologyCorrectionTable(quantile_alpha=0.9)
        assert table.correction_for(None) == 0.0
        assert table.correction_for("zone") == 0.0
        assert table.correction_for("anything") == 0.0

    def test_invalid_quantile_alpha_rejected(self):
        with pytest.raises(ValueError):
            TopologyCorrectionTable(quantile_alpha=0.0)
        with pytest.raises(ValueError):
            TopologyCorrectionTable(quantile_alpha=1.0)
        with pytest.raises(ValueError):
            TopologyCorrectionTable(quantile_alpha=1.5)

    def test_fit_recovers_known_bias_direction(self):
        """Inject a known systematic bias for one class only, fit against a
        model with NO topology feature, and confirm the two classes' corrections
        land on opposite sides of zero, matching the size of the injected gap.

        The base model has no topology feature, so it learns one blended
        average across both classes -- each class's out-of-fold residual is
        its deviation FROM that shared average, not from its own true value.
        For a 50-point gap split 50/50 between two classes, the model settles
        near the midpoint, so each class's correction should land near +/-25,
        not near 0/+50.
        """
        rng = np.random.RandomState(0)
        n = 2000
        x = rng.uniform(0, 10, n)
        topo = rng.choice(["zone", "region"], n)
        base = 100 + 5 * x
        noise = rng.normal(0, 2, n)
        injected_bias = np.where(topo == "region", 50.0, 0.0)
        y = base + noise + injected_bias

        features = pd.DataFrame({"x": x})
        table = TopologyCorrectionTable(quantile_alpha=0.5, min_samples_per_class=30)
        table.fit(features, pd.Series(y), pd.Series(topo), model_factory=_model_factory)

        assert set(table.corrections.keys()) == {"zone", "region"}
        # region (the higher-true-value class) must correct upward, zone downward
        assert table.corrections["region"] > 0
        assert table.corrections["zone"] < 0
        # the gap between the two corrections should approximate the injected 50-point split
        gap = table.corrections["region"] - table.corrections["zone"]
        assert 40 < gap < 60

    def test_classes_below_min_samples_are_skipped_not_corrected(self):
        rng = np.random.RandomState(1)
        n = 500
        x = rng.uniform(0, 10, n)
        # 'rare' class has only 5 samples -- below any reasonable min_samples_per_class
        topo = np.array(["common"] * (n - 5) + ["rare"] * 5)
        y = 100 + 5 * x + rng.normal(0, 2, n)

        features = pd.DataFrame({"x": x})
        table = TopologyCorrectionTable(quantile_alpha=0.9, min_samples_per_class=30)
        table.fit(features, pd.Series(y), pd.Series(topo), model_factory=_model_factory)

        assert "rare" not in table.corrections
        assert "rare" in table.skipped_classes
        assert table.skipped_classes["rare"] == 5
        # rare class must still be a safe no-op at lookup time
        assert table.correction_for("rare") == 0.0

    def test_serialization_round_trip(self):
        rng = np.random.RandomState(2)
        n = 1000
        x = rng.uniform(0, 10, n)
        topo = rng.choice(["host", "rack", "zone"], n)
        y = 100 + 5 * x + rng.normal(0, 2, n)

        table = TopologyCorrectionTable(quantile_alpha=0.9, min_samples_per_class=30)
        table.fit(pd.DataFrame({"x": x}), pd.Series(y), pd.Series(topo), model_factory=_model_factory)
        assert len(table.corrections) > 0

        restored = TopologyCorrectionTable.from_dict(table.to_dict())
        assert restored.corrections == table.corrections
        assert restored.quantile_alpha == table.quantile_alpha
        for cls in table.corrections:
            assert restored.correction_for(cls) == table.correction_for(cls)

    def test_mismatched_lengths_raise(self):
        table = TopologyCorrectionTable(quantile_alpha=0.9)
        with pytest.raises(ValueError):
            table.fit(
                pd.DataFrame({"x": [1, 2, 3]}),
                pd.Series([1, 2]),
                pd.Series(["zone", "zone", "zone"]),
                model_factory=_model_factory,
            )
