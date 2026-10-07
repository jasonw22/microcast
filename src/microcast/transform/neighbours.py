"""Neighbour obs as features: what the nearest stations saw at a cycle's init.

For a point with no station of its own (the house, the ride), the closest
substitute for its own last obs is its neighbours'. Leave-stations-out showed a
point's own obs carry most of the skill; these features carry the neighbours'.

Per (point, cycle, lead), from the other truth stations, never the point itself:

* ``nb{1..K}_*``: the k-th nearest station's HRRR error at init and its
  trailing 14-day bias at the same lead, plus its distance. Fixed ranks, so a
  neighbour that isn't reporting leaves its columns empty.
* ``nbw_*``: the same errors, inverse-distance weighted over the ``N_IDW``
  nearest that have a value. Robust to one neighbour being down.

As-of: every input is a gold as-of feature of the neighbour's own row for the
same cycle and lead, so it was visible at the same issue time.
"""

from __future__ import annotations

import re

import duckdb
import numpy as np
import pandas as pd

K = 3
N_IDW = 5
FIELDS = ("err_at_init_t2m", "err_at_init_gust", "bias14_t2m", "bias14_gust")
MIN_DIST_KM = 0.2  # caps one neighbour's IDW weight: 604PG and D5422 sit 230 m apart


def pairs(positions: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    """Each station's ``N_IDW`` nearest other stations: point_id, nb_id, rank (1 = nearest), dist_km."""
    p = positions.set_index("point_id").loc[sorted(stations)]
    lat, lon = np.radians(p.lat.to_numpy()), np.radians(p.lon.to_numpy())
    cos = np.sin(lat[:, None]) * np.sin(lat) + np.cos(lat[:, None]) * np.cos(lat) * np.cos(lon[:, None] - lon)
    d = 6371.0 * np.arccos(np.clip(cos, -1.0, 1.0))
    np.fill_diagonal(d, np.inf)
    ids = p.index.to_numpy()
    rows = []
    for i, pid in enumerate(ids):
        for rank, j in enumerate(np.argsort(d[i], kind="stable")[: min(N_IDW, len(ids) - 1)], start=1):
            rows.append((pid, ids[j], rank, float(d[i, j])))
    return pd.DataFrame(rows, columns=["point_id", "nb_id", "rank", "dist_km"])


def _sql() -> str:
    ranked = [f"first(g.{f}) FILTER (WHERE p.rank = {k}) AS nb{k}_{f}" for k in range(1, K + 1) for f in FIELDS] + [
        f"first(p.dist_km) FILTER (WHERE p.rank = {k}) AS nb{k}_dist_km" for k in range(1, K + 1)
    ]
    idw = [f"sum(p.w * g.{f}) / sum(p.w) FILTER (WHERE g.{f} IS NOT NULL) AS nbw_{f}" for f in FIELDS]
    return f"""
    SELECT p.point_id, g.init_time, g.lead_min,
        {", ".join(ranked + idw)}
    FROM (SELECT *, 1.0 / greatest(dist_km, {MIN_DIST_KM}) AS w FROM pairs) p
    JOIN gold g ON g.point_id = p.nb_id
    GROUP BY ALL
    """


def features(gold: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Neighbour columns keyed by (point_id, init_time, lead_min), for every station in ``gold``."""
    stations = sorted(gold.point_id.unique())
    if missing := sorted(set(stations) - set(positions.point_id)):
        raise ValueError(f"no position for {missing}")
    con = duckdb.connect()
    con.register("gold", gold[["point_id", "init_time", "lead_min", *FIELDS]])
    con.register("pairs", pairs(positions, stations))
    return con.sql(_sql()).df()


def columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(c for c in df.columns if re.match(r"nb(\d+|w)_", c))
    if not cols:
        raise ValueError("neighbour models need the nb* features joined onto gold")
    return cols
