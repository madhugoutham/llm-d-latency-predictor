# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-topology-class residual correction, applied after a frozen base model's
prediction rather than as a training feature.

Why this exists, and why it is NOT a raw XGBoost feature: feeding a
low-cardinality topology label directly into the tree model lets it steal
decision-splitting capacity from higher-value continuous features (queue
depth, token count), which was measured to improve one topology class while
regressing another on real P/D traffic (see predictor issue #30). This module
instead leaves the base model untouched and adds a small, independently
computed, per-class correction after the fact, which cannot cause that kind
of cross-class regression because no class's correction is estimated from,
or affects, another class's data.

The correction must be estimated out-of-fold (leakage-free) and aligned to
the base model's own quantile objective. A mean/median correction on a p90
model is invalid: it improves MAE while destroying p90 pinball, because a
p90 model's residuals are large and negative by design (it intentionally
over-predicts). See docstring on `fit()` for the exact reasoning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class TopologyCorrectionTable:
    """Per-class p90-aligned residual correction, fit out-of-fold on a frozen model.

    Usage:
        table = TopologyCorrectionTable(quantile_alpha=0.9, min_samples_per_class=30)
        table.fit(raw_df, target, topology_column="topology_distance",
                  model_factory=lambda: xgb.XGBRegressor(...), feature_cols=[...])
        corrected = base_prediction + table.correction_for("zone")
    """

    def __init__(self, quantile_alpha: float, min_samples_per_class: int = 30, n_folds: int = 5):
        if not 0.0 < quantile_alpha < 1.0:
            raise ValueError(f"quantile_alpha must be in (0, 1), got {quantile_alpha}")
        if min_samples_per_class < 1:
            raise ValueError(f"min_samples_per_class must be >= 1, got {min_samples_per_class}")
        self.quantile_alpha = quantile_alpha
        self.min_samples_per_class = min_samples_per_class
        self.n_folds = n_folds
        self.corrections: dict[str, float] = {}
        self.skipped_classes: dict[str, int] = {}  # class -> sample count, for classes below min_samples

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        topology_labels: pd.Series,
        model_factory,
    ) -> TopologyCorrectionTable:
        """Compute one correction value per topology class from out-of-fold residuals.

        Out-of-fold, not in-sample: an in-sample residual (predict on the same
        rows the model was trained on) is artificially small because the model
        has already seen and partially fit to those exact points, which
        inflates the apparent correction. K-fold cross-validation holds each
        fold out during its own prediction, giving an honest estimate of how
        the model performs on data it has not seen -- the same standard the
        model itself will be judged against in production.

        p90-aligned, not mean/median: this predictor's objective is the 90th
        percentile, so it deliberately over-predicts (roughly 90% of actuals
        fall below its prediction). A mean or median correction assumes the
        model is unbiased around the center of the distribution, which is
        false by construction for a quantile model -- applying one shifts
        predictions toward the mean and destroys the model's actual coverage
        guarantee even though it looks like an improvement under MAE.
        """
        if len(features) != len(target) or len(features) != len(topology_labels):
            raise ValueError("features, target, and topology_labels must have equal length")

        self.corrections = {}
        self.skipped_classes = {}

        n_samples = len(features)
        if n_samples < self.n_folds:
            return self

        oof_pred = np.full(n_samples, np.nan)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        features_reset = features.reset_index(drop=True)
        target_reset = target.reset_index(drop=True)

        for train_idx, val_idx in kf.split(features_reset):
            model = model_factory()
            model.fit(features_reset.iloc[train_idx], target_reset.iloc[train_idx])
            oof_pred[val_idx] = model.predict(features_reset.iloc[val_idx])

        oof_residual = target_reset.to_numpy() - oof_pred
        labels_reset = topology_labels.reset_index(drop=True)

        for cls in labels_reset.dropna().unique():
            mask = (labels_reset == cls).to_numpy()
            n_class = int(mask.sum())
            if n_class < self.min_samples_per_class:
                self.skipped_classes[cls] = n_class
                continue
            self.corrections[cls] = float(np.quantile(oof_residual[mask], self.quantile_alpha))

        return self

    def correction_for(self, topology_class: str | None) -> float:
        """Return the stored correction for a class, or 0.0 (no-op) if unknown/absent.

        Absent or unrecognized topology_class must never raise -- every
        existing caller that does not send this field relies on this method
        being a pure no-op.
        """
        if topology_class is None:
            return 0.0
        return self.corrections.get(topology_class, 0.0)

    def to_dict(self) -> dict:
        """Serializable representation for joblib persistence."""
        return {
            "quantile_alpha": self.quantile_alpha,
            "min_samples_per_class": self.min_samples_per_class,
            "n_folds": self.n_folds,
            "corrections": dict(self.corrections),
            "skipped_classes": dict(self.skipped_classes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TopologyCorrectionTable:
        table = cls(
            quantile_alpha=data["quantile_alpha"],
            min_samples_per_class=data.get("min_samples_per_class", 30),
            n_folds=data.get("n_folds", 5),
        )
        table.corrections = dict(data.get("corrections", {}))
        table.skipped_classes = dict(data.get("skipped_classes", {}))
        return table
