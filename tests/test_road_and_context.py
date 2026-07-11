"""Road perception, complexity, providers."""
from pathlib import Path

from context_providers import SensorHub, WeatherProvider, validate_sensor_packet
from road_perception.detection import Detection, MockDetector
from road_perception.road_complexity import RoadComplexityIndex
from road_perception.tracking import IouTracker

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "digital_twin" / "scenarios"


def test_mock_detector_playback_and_distance_closing():
    det = MockDetector(SCEN / "motorcycle_cut_in.json")
    early = det.detect(None, 2.5)
    later = det.detect(None, 5.0)
    moto_e = next(d for d in early if d.cls == "motorcycle")
    moto_l = next(d for d in later if d.cls == "motorcycle")
    assert moto_l.distance_m < moto_e.distance_m  # approaching
    assert moto_e.ttc_s is not None and moto_e.ttc_s > 0
    assert det.detect(None, 0.5) is not None
    assert all(d.cls != "motorcycle" for d in det.detect(None, 1.0))  # appears at t=2


def test_tracker_assigns_stable_ids():
    tr = IouTracker()
    d1 = [Detection("car", 0.9, (0.4, 0.4, 0.2, 0.2))]
    d2 = [Detection("car", 0.9, (0.41, 0.4, 0.21, 0.21))]
    a = tr.update(d1, 0.0)[0].track_id
    b = tr.update(d2, 0.1)[0].track_id
    assert a == b


def test_complexity_explainable_and_bounded():
    rci = RoadComplexityIndex()
    empty = rci.compute([], visibility="good", weather="clear")
    assert empty.score <= 10
    dets = [Detection("motorcycle", 0.9, (0, 0, 0.1, 0.1)) for _ in range(4)] + \
           [Detection("pedestrian", 0.9, (0, 0, 0.1, 0.1)) for _ in range(3)] + \
           [Detection("wrong_side_vehicle", 0.9, (0, 0, 0.1, 0.1)),
            Detection("pothole", 0.8, (0, 0, 0.1, 0.1))]
    busy = rci.compute(dets, road_width_m=5.0, lane_marked=False, zone="market",
                       weather="rain", speed_kmh=55)
    assert busy.score >= 60
    assert 0 <= busy.score <= 100
    assert any("wrong-side" in r.lower() for r in busy.reasons)
    assert len(busy.reasons) >= 4


def test_weather_multiplier():
    wx = WeatherProvider().get("heavy_rain")
    assert wx.data["risk_multiplier"] > 1.3
    assert wx.data["friction_estimate"] < 0.5
    assert wx.available


def test_sensor_packet_validation():
    good = {"timestamp": "t", "accelerometer": {"x": 0.1, "y": -0.04, "z": 9.72},
            "gyroscope": {"x": 0.01, "y": 0.03, "z": -0.02}, "speed_kmh": 46.5}
    assert validate_sensor_packet(good) is None
    assert validate_sensor_packet({"timestamp": "t"}) is not None
    bad_speed = dict(good, speed_kmh=900)
    assert validate_sensor_packet(bad_speed) is not None
    hub = SensorHub()
    assert hub.ingest(good) and not hub.ingest(bad_speed)
    assert hub.get().available
    assert hub.get().data["label"] == "experimental"
