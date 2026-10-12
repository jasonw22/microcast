"""HRRR's 4 nearest cells kept apart, instead of only their distance-weighted blend.

silver.nwp_aligned blends the 4 cells around a point by distance alone, so a
point beside the ocean gets a mix of sea and land cells that may be true
nowhere, and the model can't tell which cell resembled it. These features keep
each cell, ranked by distance (1 = nearest):

* ``cell{r}_{f}_d``: the cell's value minus the blend, for t2m, gust, wind10.
* ``cell{r}_hgt_m``, ``cell{r}_land``, ``cell{r}_dist_m``: HRRR's terrain
  height and land mask for that cell, and its distance from the point.
* ``cell{r}_dz_m``: the point's real ground (``st_elev_m``) above the cell.
* ``cellland_{f}_d``: the blend over land cells only, minus the full blend;
  empty where every cell is water.

silver.nwp_cells holds the per-cell values (``build``); the static cell
attributes are read from HRRR's own HGT and LAND fields when gold is loaded.
As-of: the same forecast values as the blend, so visible at the same issue time.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
from pyiceberg.expressions import And, EqualTo, In

from microcast.lake.schemas import NWP_CELLS_SCHEMA

K = 4
FIELDS = ("t2m", "gust", "wind10")

NWP_CELLS_SQL = f"""
WITH latest AS (
    SELECT * FROM bronze
    WHERE member = 0 AND model = 'hrrr' AND variable IN ('t2m', 'gust', 'u10', 'v10')
    QUALIFY row_number() OVER (
        PARTITION BY model, product, init_time, lead_min, point_id, variable, level, member, k
        ORDER BY ingested_at DESC) = 1
),
cell AS (
    PIVOT (SELECT model, init_time, lead_min, point_id, k, grid_dist_m, variable, value FROM latest)
    ON variable IN ('t2m', 'gust', 'u10', 'v10')
    USING first(value)
    GROUP BY model, init_time, lead_min, point_id, k, grid_dist_m
),
ranked AS (
    SELECT *, row_number() OVER (
        PARTITION BY model, init_time, lead_min, point_id ORDER BY grid_dist_m, k) AS r
    FROM cell
)
SELECT model, init_time, lead_min, point_id,
    {
    ", ".join(
        f"first(t2m - 273.15) FILTER (WHERE r = {r}) AS cell{r}_t2m, "
        f"first(gust) FILTER (WHERE r = {r}) AS cell{r}_gust, "
        f"first(sqrt(u10 * u10 + v10 * v10)) FILTER (WHERE r = {r}) AS cell{r}_wind10"
        for r in range(1, K + 1)
    )
}
FROM ranked
GROUP BY ALL
"""


def build(bronze: pa.Table) -> pa.Table:
    """Bronze nwp rows -> one wide row per (model, cycle, lead, point) with each cell's values."""
    con = duckdb.connect()
    con.register("bronze", bronze)
    return (
        con.sql(NWP_CELLS_SQL).to_arrow_table().select(NWP_CELLS_SCHEMA.column_names).cast(NWP_CELLS_SCHEMA.as_arrow())
    )


def rebuild(start: date, end: date, log: Callable[[str], None] = print) -> None:
    """silver.nwp_cells for every month in [start, end), from bronze.nwp_point."""
    from microcast.ingest.obs import months
    from microcast.lake.catalog import get_catalog, replace_partition
    from microcast.transform.silver import _range, _utc

    nwp = get_catalog().load_table("bronze.nwp_point")
    cols = ("model", "product", "init_time", "lead_min", "point_id", "variable", "level", "member", "k")
    cols += ("value", "grid_dist_m", "ingested_at")
    for first, last in months(start, end):
        lo, hi = _utc(first), _utc(last)
        rows = And(_range("init_time", lo, hi), EqualTo("model", "hrrr"), In("variable", ["t2m", "gust", "u10", "v10"]))
        b = nwp.scan(row_filter=rows, selected_fields=cols).to_arrow()
        if b.num_rows:
            out = build(b)
            replace_partition("silver.nwp_cells", out, _range("init_time", lo, hi))
            log(f"  silver.nwp_cells {first:%Y-%m}: {b.num_rows} bronze -> {out.num_rows} rows")


def statics(positions: pd.DataFrame) -> pd.DataFrame:
    """Per point: each ranked cell's HRRR height, land mask and distance (``cell{r}_hgt_m`` ...)."""
    from microcast.ingest import hrrr_zarr
    from microcast.registry import VirtualPoint

    url = hrrr_zarr._cycle_url(pd.Timestamp("2025-07-15T00:00Z").to_pydatetime(), "anl")
    hgt = hrrr_zarr._read_window(f"{url}/surface/HGT/surface/HGT", with_time=False)[0]
    land = hrrr_zarr._read_window(f"{url}/surface/LAND/surface/LAND", with_time=False)[0]
    r0, _, c0, _ = hrrr_zarr.window()
    vps = [VirtualPoint(p.point_id, p.point_id, 0, p.lat, p.lon, None) for p in positions.itertuples()]
    rows = {}
    for n in hrrr_zarr.point_neighbours(vps, k=K):
        r = n.k + 1
        row = rows.setdefault(n.point_id, {"point_id": n.point_id})
        row[f"cell{r}_hgt_m"] = float(hgt[n.row - r0, n.col - c0])
        row[f"cell{r}_land"] = float(land[n.row - r0, n.col - c0])
        row[f"cell{r}_dist_m"] = float(n.dist_m)
    return pd.DataFrame(list(rows.values()))


def features(gold: pd.DataFrame, cells: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """``gold`` with the cell columns joined: deltas from the blend, cell statics, land-only blend.

    ``cells`` is silver.nwp_cells for HRRR; ``gold`` needs ``st_elev_m`` (the static features).
    """
    out = gold.merge(cells.drop(columns="model"), on=["point_id", "init_time", "lead_min"], how="left")
    out = out.merge(statics(positions[positions.point_id.isin(gold.point_id.unique())]), on="point_id", how="left")
    for f in FIELDS:
        vals = np.stack([out[f"cell{r}_{f}"].to_numpy(dtype="float64") for r in range(1, K + 1)], axis=1)
        land_w = [out[f"cell{r}_land"] / out[f"cell{r}_dist_m"].clip(lower=100.0) for r in range(1, K + 1)]
        w = np.stack([x.to_numpy(dtype="float64") for x in land_w], axis=1)
        w = np.where(np.isnan(vals), 0.0, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            land = np.nansum(w * np.nan_to_num(vals), axis=1) / w.sum(axis=1)
        out[f"cellland_{f}_d"] = land - out[f"nwp_{f}"]
        for r in range(1, K + 1):
            out[f"cell{r}_{f}_d"] = out.pop(f"cell{r}_{f}") - out[f"nwp_{f}"]
    for r in range(1, K + 1):
        out[f"cell{r}_dz_m"] = out.st_elev_m - out[f"cell{r}_hgt_m"]
    return out


def columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(c for c in df.columns if c.startswith("cell"))
    if not cols:
        raise ValueError("cell models need the cell* features joined onto gold")
    return cols
