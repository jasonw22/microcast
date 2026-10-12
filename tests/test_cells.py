"""HRRR's 4 cells kept apart: ranking by distance, deltas from the blend, land-only blend."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from microcast.transform import cells

T0 = datetime(2026, 7, 1, tzinfo=UTC)


def _bronze() -> pa.Table:
    rows = []
    # k order deliberately not distance order: k=0 is the farthest cell.
    for k, dist in enumerate([3000.0, 1000.0, 2000.0, 2500.0]):
        for var, val in (("t2m", 290.15 + k), ("gust", 5.0 + k), ("u10", 3.0), ("v10", 4.0)):
            rows.append(
                dict(
                    model="hrrr",
                    product="sfc",
                    init_time=T0,
                    lead_min=60,
                    point_id="A",
                    variable=var,
                    level="",
                    member=0,
                    k=k,
                    value=val,
                    grid_dist_m=dist,
                    ingested_at=T0,
                )
            )
    # A stale duplicate of one cell: the later ingest wins.
    rows.append({**rows[0], "value": 0.0, "ingested_at": datetime(2026, 6, 30, tzinfo=UTC)})
    return pa.Table.from_pylist(rows)


def test_build_ranks_cells_by_distance():
    out = cells.build(_bronze()).to_pandas()
    assert len(out) == 1
    row = out.iloc[0]
    # Nearest is k=1 (1000 m), then k=2, k=3, k=0.
    assert [row[f"cell{r}_t2m"] for r in range(1, 5)] == pytest.approx([18.0, 19.0, 20.0, 17.0])
    assert row.cell1_gust == 6.0
    assert row.cell4_wind10 == pytest.approx(5.0)


def test_features_deltas_statics_and_land_blend(monkeypatch):
    stat = pd.DataFrame(
        {
            "point_id": ["A"],
            **{f"cell{r}_hgt_m": [h] for r, h in zip(range(1, 5), [0.0, 10.0, 50.0, 100.0], strict=True)},
            **{f"cell{r}_land": [lnd] for r, lnd in zip(range(1, 5), [0.0, 1.0, 1.0, 0.0], strict=True)},
            **{f"cell{r}_dist_m": [d] for r, d in zip(range(1, 5), [1000.0, 2000.0, 2500.0, 3000.0], strict=True)},
        }
    )
    monkeypatch.setattr(cells, "statics", lambda positions: stat)
    vals = cells.build(_bronze()).to_pandas()
    gold = pd.DataFrame(
        dict(
            point_id=["A"],
            init_time=[pd.Timestamp(T0)],
            lead_min=[60],
            nwp_t2m=[18.5],
            nwp_gust=[6.5],
            nwp_wind10=[5.0],
            st_elev_m=[30.0],
        )
    )
    pos = pd.DataFrame({"point_id": ["A"], "lat": [37.77], "lon": [-122.5]})
    out = cells.features(gold, vals, pos).iloc[0]
    assert out.cell1_t2m_d == pytest.approx(-0.5)
    assert out.cell2_dz_m == pytest.approx(20.0)
    # Land cells are ranks 2 (19 C, 2000 m) and 3 (20 C, 2500 m).
    w2, w3 = 1 / 2000, 1 / 2500
    assert out.cellland_t2m_d == pytest.approx((19 * w2 + 20 * w3) / (w2 + w3) - 18.5)
    assert "cell1_t2m" not in out.index
    cols = cells.columns(out.to_frame().T)
    assert "cell3_land" in cols and "cellland_gust_d" in cols


def test_land_blend_is_empty_over_water(monkeypatch):
    stat = pd.DataFrame({"point_id": ["A"], **{f"cell{r}_{a}": [0.0] for r in range(1, 5) for a in ("hgt_m", "land")}})
    for r in range(1, 5):
        stat[f"cell{r}_dist_m"] = 1000.0 * r
    monkeypatch.setattr(cells, "statics", lambda positions: stat)
    gold = pd.DataFrame(
        dict(point_id=["A"], init_time=[pd.Timestamp(T0)], lead_min=[60], nwp_t2m=[18.0], nwp_gust=[6.0])
    ).assign(nwp_wind10=5.0, st_elev_m=0.0)
    out = cells.features(gold, cells.build(_bronze()).to_pandas(), pd.DataFrame({"point_id": ["A"]}))
    assert np.isnan(out.cellland_t2m_d.iloc[0])
