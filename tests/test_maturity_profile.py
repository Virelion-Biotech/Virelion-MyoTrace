import pytest
from fusion.maturity_profile import MaturityProfileError, build_maturity_profile, video_only_profile

HASH="a"*64

def measured(**kw):
    return {"status":"measured","value":1.2,"unit":"ratio","modality":"imaging","method":"alpha-actinin sarcomere image quantification","calibration":{"method":"scale bar","pixel_size_um":0.2},"sample_id":"S1","timepoint":"day14","qc":{"status":"pass"},"source_artifact_sha256":HASH,**kw}

def test_video_only_leaves_nonmechanical_dimensions_unmeasured():
    p=video_only_profile({"n_beats":4,"mean_bpm":60,"measurement_status":"motion_events_not_verified_cardiac_beats"},sample_id="S1",timepoint="day7",source_artifact_sha256=HASH)
    assert p["dimensions"]["contractility_beat_mechanics"]["status"]=="proxy"
    assert p["dimensions"]["electrophysiological_maturity"]["status"]=="not_measured"
    assert p["dimensions"]["calcium_transient_maturity"]["value"] is None
    assert p["dimensions"]["metabolic_maturation"]["status"]=="not_measured"
    assert p["composite"]["status"]=="not_computed"

def test_static_video_has_not_estimable_mechanics_not_zero_maturity():
    p=video_only_profile({"n_beats":0,"mean_bpm":None},sample_id="S1",timepoint="day0",source_artifact_sha256=HASH)
    assert p["dimensions"]["contractility_beat_mechanics"]["status"]=="not_estimable"
    assert p["dimensions"]["contractility_beat_mechanics"]["value"] is None

def test_missing_calibration_for_physical_force_is_rejected():
    force=measured(modality="mechanical",unit="uN",value=3.0,calibration=None)
    with pytest.raises(MaturityProfileError,match="require explicit calibration"):
        build_maturity_profile(sample_id="S1",timepoint="day1",measurements={"contractility_beat_mechanics":force})

def test_modality_leakage_is_rejected():
    bad=measured(modality="mechanical")
    with pytest.raises(MaturityProfileError,match="incompatible"):
        build_maturity_profile(sample_id="S1",timepoint="day1",measurements={"sarcomeric_organization":bad})

def test_unit_mismatch_is_rejected():
    bad=measured(unit="mV")
    with pytest.raises(MaturityProfileError,match="unit"):
        build_maturity_profile(sample_id="S1",timepoint="day1",measurements={"sarcomeric_organization":bad})

def test_missing_measurement_stays_null_not_zero():
    p=build_maturity_profile(sample_id="S1",timepoint="day14")
    assert all(x["value"] is None for x in p["dimensions"].values())
