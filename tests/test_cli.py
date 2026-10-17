"""CLI argument defaults."""

from __future__ import annotations

from datetime import UTC, date, datetime

from microcast import cli


def test_end_day_explicit():
    assert cli._end_day("2026-10-18") == date(2026, 10, 18)


def test_end_day_default_is_utc_tomorrow(monkeypatch):
    # 03:40Z on Oct 17 is still Oct 16 in San Francisco; the default must follow UTC
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 17, 3, 40, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(cli, "datetime", FrozenDatetime)
    assert cli._end_day(None) == date(2026, 10, 18)
