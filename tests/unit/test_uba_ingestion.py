import pandas as pd

from aqcascade.ingestion.uba import _parse_uba_timestamp


def test_parses_normal_timestamp():
    assert _parse_uba_timestamp("2026-09-20 05:00:00") == pd.Timestamp("2026-09-20 05:00:00")


def test_parses_hour_24_as_next_day_midnight():
    # UBA represents the end of the last hour of a day as "24:00:00" rather
    # than rolling over to the next date — confirmed against a real API
    # response during Phase 2 ingestion.
    assert _parse_uba_timestamp("2026-09-20 24:00:00") == pd.Timestamp("2026-09-21 00:00:00")
