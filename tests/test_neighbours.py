"""Neighbour obs features: ranks, self-exclusion, inverse-distance weighting."""

from __future__ import annotations

import pandas as pd
import pytest

from microcast import models as zoo
from microcast.transform import neighbours

T0 = pd.Timestamp("2026-07-01T00:00Z")

# Along a line of latitude, ~0.885 km per 0.01 degree of longitude at 37.77.
POS = pd.DataFrame(
    {
        "point_id": ["A", "B", "C", "D"],
        "lat": [37.77] * 4,
        "lon": [-122.50, -122.49, -122.47, -122.43],
    }
)


def _gold(errs: dict[str, float | None], lead: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                point_id=p,
                init_time=T0,
                lead_min=lead,
                err_at_init_t2m=e,
                err_at_init_gust=None if e is None else 10 * e,
                bias14_t2m=0.5,
                bias14_gust=None,
            )
            for p, e in errs.items()
        ]
    )


def test_pairs_exclude_self_and_rank_by_distance():
    p = neighbours.pairs(POS, ["A", "B", "C", "D"])
    a = p[p.point_id == "A"].sort_values("rank")
    assert a.nb_id.tolist() == ["B", "C", "D"]  # only 3 others, fewer than N_IDW
    assert "A" not in a.nb_id.tolist()
    assert a.dist_km.iloc[0] == pytest.approx(0.88, abs=0.01)
    c = p[p.point_id == "C"].sort_values("rank")
    assert c.nb_id.tolist() == ["B", "A", "D"]


def test_ranked_features_are_the_neighbours_values_not_the_points_own():
    out = neighbours.features(_gold({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}), POS).set_index("point_id")
    assert out.loc["A", "nb1_err_at_init_t2m"] == 2.0
    assert out.loc["A", "nb2_err_at_init_t2m"] == 3.0
    assert out.loc["A", "nb3_err_at_init_gust"] == 40.0
    assert out.loc["C", "nb1_err_at_init_t2m"] == 2.0
    assert out.loc["A", "nb1_dist_km"] == pytest.approx(0.88, abs=0.01)


def test_idw_skips_missing_neighbours():
    out = neighbours.features(_gold({"A": 0.0, "B": None, "C": 3.0, "D": 6.0}), POS).set_index("point_id")
    # B is down: A's weighted mean is over C (2.66 km) and D (6.20 km) only.
    w_c, w_d = 1 / 2.655, 1 / 6.195
    assert out.loc["A", "nbw_err_at_init_t2m"] == pytest.approx((3 * w_c + 6 * w_d) / (w_c + w_d), rel=1e-2)
    assert pd.isna(out.loc["A", "nb1_err_at_init_t2m"])
    assert pd.isna(out.loc["A", "nbw_bias14_gust"])  # no neighbour has it


def test_features_need_every_stations_position():
    with pytest.raises(ValueError, match="no position"):
        neighbours.features(_gold({"A": 1.0, "Z": 2.0}), POS)


def test_neighbour_models_use_no_own_obs_without_them():
    df = neighbours.features(_gold({"A": 1.0, "B": 2.0, "C": 3.0}), POS).assign(st_elev_m=1.0)
    cols = zoo.make("gbm_nwp_neighbours", "t2m").columns(df)
    assert "nb1_err_at_init_t2m" in cols and "nbw_bias14_t2m" in cols
    assert not any(c.startswith(("obs_last", "err_at_init", "bias14", "std14")) for c in cols)
    assert "point" not in cols
    own = zoo.make("gbm_terrain_neighbours", "t2m").columns(df)
    assert "err_at_init_t2m" in own and "nb1_err_at_init_t2m" in own
