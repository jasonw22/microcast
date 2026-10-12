"""NBM (National Blend of Models) at the truth stations: the operational benchmark.

NBM is NWS's calibrated multi-model blend on a 2.5 km grid, already bias
corrected against the URMA analysis, so it is the forecast a researcher
expects any point product to beat. It is a comparison only, never a model
input. The ``core`` product carries the blend's mean and its ensemble
spread (``ens std dev``) for 2 m temperature and 10 m gust, so it can be
scored as a probabilistic forecast like ours.

Cells are picked as for HRRR (the 4 nearest, ``nwp.K_NEIGHBOURS``) and land in
bronze.nwp_point as ``model = 'nbm'``, variables ``t2m``, ``gust`` and their
``_sd``. A run is published 42-62 min after its nominal hour (S3 Last-Modified,
checked 2026-10-12), so the NBM run of the same hour as a HRRR cycle is what a
forecaster had at our issue time (init + 55 min), give or take a late run.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import pandas as pd
import pyarrow as pa

from microcast import settings
from microcast.ingest import nwp

PRODUCT = "co"
MEAN = r":(?:TMP:2 m|GUST:10 m) above ground:\d+ hour fcst:$"
SPREAD = r":(?:TMP:2 m|GUST:10 m) above ground:\d+ hour fcst:ens std dev:$"
NAMES = {"t2m": "t2m", "i10fg": "gust"}


def batch_id(init: datetime) -> str:
    return f"nbm/{init:%Y-%m-%dT%H}"


def fetch_cycle(init: datetime, points: pd.DataFrame, leads: list[int]) -> pa.Table:
    """One NBM run, mean and spread for each lead, at ``points`` (point_id, lat, lon)."""
    import herbie
    from herbie import Herbie

    cache = settings.data_dir() / "grib"
    herbie.config["default"]["save_dir"] = cache
    pts = points.rename(columns={"point_id": "id", "lat": "latitude", "lon": "longitude"})
    frames = []
    for fxx in leads:
        H = Herbie(
            init.replace(tzinfo=None),
            model="nbm",
            product=PRODUCT,
            fxx=fxx,
            save_dir=cache,
            priority=["aws"],
            verbose=False,
        )
        for search, suffix in ((MEAN, ""), (SPREAD, "_sd")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                ds = H.xarray(search, remove_grib=True)
            ds = ds if isinstance(ds, list) else [ds]
            picked = [
                d.herbie.pick_points(pts, method="nearest", k=nwp.K_NEIGHBOURS, tree_name="nbm_conus") for d in ds
            ]
            df = nwp.to_long(picked, model="nbm", product=PRODUCT, ingest_batch=batch_id(init)).to_pandas()
            df["variable"] = df.variable.map(NAMES) + suffix
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return pa.Table.from_pandas(out, schema=nwp.NWP_POINT_SCHEMA.as_arrow(), preserve_index=False)


def backfill(
    points: pd.DataFrame,
    start: datetime,
    end: datetime,
    leads: list[int],
    *,
    stride_h: int = 24,
    workers: int = 4,
    commit_every: int = 24,
    log=print,
) -> dict[str, int]:
    """Every missing NBM run in [start, end), one in ``stride_h`` (rotating hour), into bronze.nwp_point.

    Resumable like the HRRR backfill: runs recorded in bronze snapshot
    summaries are skipped. Fetches run in worker processes (cfgrib decodes
    a CONUS grid per message); appends happen here, one writer.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from microcast.ingest.hrrr_zarr import cycles
    from microcast.lake.catalog import append_bronze, recorded_batches

    done = recorded_batches("bronze.nwp_point")
    every = cycles(start, end, stride_h)
    todo = [c for c in every if batch_id(c) not in done]
    stats = {"skipped": len(every) - len(todo), "fetched": 0, "failed": 0, "rows": 0}
    log(f"{len(todo)} NBM runs to fetch, {stats['skipped']} already in bronze")
    pending: list[pa.Table] = []
    ids: list[str] = []

    def flush() -> None:
        if pending:
            table = pa.concat_tables(pending)
            append_bronze("bronze.nwp_point", table, batches=ids)
            stats["rows"] += table.num_rows
            log(f"  committed {len(ids)} runs ({table.num_rows} rows), last {ids[-1]}")
            pending.clear()
            ids.clear()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_cycle, c, points, leads): c for c in todo}
        for fut in as_completed(futures):
            init = futures[fut]
            try:
                pending.append(fut.result())
                ids.append(batch_id(init))
                stats["fetched"] += 1
            except Exception as e:  # a missing or truncated run: log it, rerun later
                stats["failed"] += 1
                log(f"  failed {init:%Y-%m-%dT%HZ}: {type(e).__name__}: {e}")
            if len(pending) >= commit_every:
                flush()
    flush()
    return stats
