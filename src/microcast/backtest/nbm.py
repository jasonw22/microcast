"""Score NBM on the same rows as the rolling backtest (gold.scores).

Two NBM forecasts, so the comparison can't be blamed on how NBM's
uncertainty was set:

* ``nbm``: the blend's mean, with its own ensemble spread as sigma.
* ``nbm_cal``: the same mean, sigma = NBM's RMS error per (variable, lead)
  over every other month: the treatment raw_hrrr gets, and generous to NBM
  (it sees errors from after the test month). With a single month there are
  no others, and that month's own RMS is used.

Truth, valid time and the rows themselves come from the backtest's raw_hrrr
rows, so NBM is scored on exactly the station-hours our models were.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from microcast.backtest import _score

SIGMA_FLOOR = 0.05

ALIGNED_SQL = """
WITH latest AS (
    SELECT * FROM bronze
    WHERE model = 'nbm'
    QUALIFY row_number() OVER (
        PARTITION BY init_time, lead_min, point_id, variable, k ORDER BY ingested_at DESC) = 1
)
SELECT init_time, lead_min, point_id,
    sum(w * value) FILTER (WHERE variable = 't2m') / sum(w) FILTER (WHERE variable = 't2m') - 273.15 AS t2m,
    sum(w * value) FILTER (WHERE variable = 't2m_sd') / sum(w) FILTER (WHERE variable = 't2m_sd') AS t2m_sd,
    sum(w * value) FILTER (WHERE variable = 'gust') / sum(w) FILTER (WHERE variable = 'gust') AS gust,
    sum(w * value) FILTER (WHERE variable = 'gust_sd') / sum(w) FILTER (WHERE variable = 'gust_sd') AS gust_sd
FROM (SELECT *, 1.0 / greatest(grid_dist_m, 100.0) AS w FROM latest)
GROUP BY ALL
"""


def aligned(bronze: pd.DataFrame) -> pd.DataFrame:
    """NBM bronze rows -> one row per (run, lead, point): 4 cells inverse-distance weighted, degC."""
    con = duckdb.connect()
    con.register("bronze", bronze)
    return con.sql(ALIGNED_SQL).df()


def scored(nbm: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """``nbm`` and ``nbm_cal`` rows shaped like gold.scores, on the raw_hrrr rows NBM covers."""
    base = scores[scores.model_name == "raw_hrrr"]
    out = []
    for target in ("t2m", "gust"):
        b = base[base.variable == target][["variable", "point_id", "init_time", "lead_min", "valid_time", "obs"]]
        n = nbm[["point_id", "init_time", "lead_min", target, f"{target}_sd"]].dropna()
        m = b.merge(n.rename(columns={target: "mu", f"{target}_sd": "sd"}), on=["point_id", "init_time", "lead_min"])
        own = m.assign(model_name="nbm", sigma=m.sd.clip(lower=SIGMA_FLOOR))
        month = m.init_time.dt.tz_convert(None).dt.to_period("M")
        sq = (m.mu - m.obs) ** 2
        tot = sq.groupby(m.lead_min).transform("sum")
        cnt = sq.groupby(m.lead_min).transform("count")
        in_month = sq.groupby([m.lead_min, month]).transform("sum")
        n_month = sq.groupby([m.lead_min, month]).transform("count")
        others = cnt - n_month
        # A single month has no others: fall back to its own RMS (only in short smoke runs).
        rms_other = np.sqrt(((tot - in_month) / others).where(others > 0, tot / cnt))
        cal = m.assign(model_name="nbm_cal", sigma=rms_other.clip(lower=SIGMA_FLOOR))
        out += [own, cal]
    df = pd.concat(out, ignore_index=True).drop(columns="sd")
    return _score(df)
