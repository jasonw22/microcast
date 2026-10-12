"""NBM benchmark: cell weighting, and scoring on the backtest's own rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcast.backtest import nbm

T0 = pd.Timestamp("2026-07-01T12:00Z")
T1 = pd.Timestamp("2026-08-01T12:00Z")


def _bronze() -> pd.DataFrame:
    rows = []
    for var, vals in (("t2m", [290.15, 292.15]), ("t2m_sd", [1.0, 2.0]), ("gust", [5.0, 8.0]), ("gust_sd", [0.5, 0.5])):
        for k, (v, d) in enumerate(zip(vals, [1000.0, 3000.0], strict=True)):
            rows.append(
                dict(
                    model="nbm",
                    init_time=T0,
                    lead_min=60,
                    point_id="A",
                    variable=var,
                    k=k,
                    value=v,
                    grid_dist_m=d,
                    ingested_at=T0,
                )
            )
    rows.append({**rows[0], "value": 0.0, "ingested_at": T0 - pd.Timedelta("1D")})  # stale: superseded
    return pd.DataFrame(rows)


def test_aligned_weights_cells_by_inverse_distance_in_celsius():
    out = nbm.aligned(_bronze()).iloc[0]
    w1, w2 = 1 / 1000, 1 / 3000
    assert out.t2m == pytest.approx((17 * w1 + 19 * w2) / (w1 + w2))
    assert out.gust == pytest.approx((5 * w1 + 8 * w2) / (w1 + w2))
    assert out.t2m_sd == pytest.approx((1 * w1 + 2 * w2) / (w1 + w2))


def _scores() -> pd.DataFrame:
    rows = []
    for init, obs in ((T0, 18.0), (T1, 20.0)):
        for model in ("raw_hrrr", "gbm_terrain"):
            rows.append(
                dict(
                    model_name=model,
                    variable="t2m",
                    point_id="A",
                    init_time=init,
                    lead_min=60,
                    valid_time=init + pd.Timedelta("1h"),
                    obs=obs,
                )
            )
    return pd.DataFrame(rows)


def test_scored_uses_backtest_truth_and_both_sigmas():
    runs = pd.DataFrame(
        dict(
            init_time=[T0, T1, T1],
            lead_min=[60, 60, 120],  # lead 120 has no backtest row: dropped
            point_id=["A", "A", "A"],
            t2m=[17.0, 23.0, 0.0],
            t2m_sd=[0.5, 0.0, 1.0],
            gust=[np.nan] * 3,
            gust_sd=[np.nan] * 3,
        )
    )
    out = nbm.scored(runs, _scores()).set_index(["model_name", "init_time"])
    assert len(out) == 4
    assert out.loc[("nbm", T0), "obs"] == 18.0
    assert out.loc[("nbm", T0), "error"] == pytest.approx(-1.0)
    assert out.loc[("nbm", T1), "sigma"] == nbm.SIGMA_FLOOR  # zero spread floored
    # Calibrated sigma for July is the RMS error of the other month (August: 3.0), and vice versa.
    assert out.loc[("nbm_cal", T0), "sigma"] == pytest.approx(3.0)
    assert out.loc[("nbm_cal", T1), "sigma"] == pytest.approx(1.0)
    assert {"crps", "in_p10_p90"} <= set(out.columns)
