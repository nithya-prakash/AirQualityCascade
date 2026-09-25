import pytest

from aqcascade.common.geo import haversine_km


def test_haversine_zero_distance():
    assert haversine_km(52.5, 13.4, 52.5, 13.4) == pytest.approx(0.0, abs=1e-9)


def test_haversine_berlin_to_munich():
    # Known real-world great-circle distance, ~504 km.
    berlin = (52.5200, 13.4050)
    munich = (48.1351, 11.5820)
    dist = haversine_km(*berlin, *munich)
    assert dist == pytest.approx(504, abs=10)
