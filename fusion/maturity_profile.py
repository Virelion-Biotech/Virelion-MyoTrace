"""Multidimensional maturity evidence profile; intentionally no composite score."""
from __future__ import annotations
import hashlib, json, math
from typing import Any

PROFILE_VERSION="myotrace-maturity-profile/1.0.0"
DIMENSIONS={
 "sarcomeric_organization":{"modalities":{"imaging","structural"},"units":{"dimensionless","fraction","ratio","um","µm"}},
 "contractility_beat_mechanics":{"modalities":{"mechanical"},"units":{"motion_index","pixels_per_frame","um_per_s","µm/s","uN","mN","normalized_motion_index"}},
 "electrophysiological_maturity":{"modalities":{"electrical"},"units":{"ms","mV","Hz","dimensionless"}},
 "calcium_transient_maturity":{"modalities":{"calcium","imaging"},"units":{"dF/F0","ΔF/F0","ratio","ms"}},
 "metabolic_maturation":{"modalities":{"metabolic"},"units":{"pmol/min/ug","pmol/min/µg","relative","ratio"}},
 "cell_cycle_proliferation":{"modalities":{"cell_cycle","imaging","molecular"},"units":{"fraction","%","positive_cells_per_total","cells_per_field"}},
 "electrical_coupling_synchrony":{"modalities":{"electrical","imaging"},"units":{"ms","m_per_s","synchrony_index","dimensionless"}},
 "tissue_architecture_alignment":{"modalities":{"imaging","structural"},"units":{"dimensionless","fraction","degrees","um","µm"}}
}
STATUSES={"measured","proxy","not_measured","not_estimable","not_applicable"}

class MaturityProfileError(ValueError): pass

def _is_text(x): return isinstance(x,str) and bool(x.strip())

def _valid_hash(x): return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdefABCDEF" for c in x)

def _empty(dimension:str,status:str="not_measured",reason:str="no modality-specific measurement supplied")->dict[str,Any]:
    return {"dimension":dimension,"status":status,"value":None,"unit":None,"modality":None,"method":None,"calibration":None,"sample_id":None,"timepoint":None,"qc":None,"source_artifact_sha256":None,"reason":reason}

def validate_dimension(dimension:str,measurement:dict[str,Any])->list[str]:
    errors=[]
    if dimension not in DIMENSIONS: return [f"unknown maturity dimension: {dimension}"]
    status=measurement.get("status")
    if status not in STATUSES: errors.append(f"{dimension}: unsupported status")
    if status in {"not_measured","not_estimable","not_applicable"}:
        if measurement.get("value") is not None: errors.append(f"{dimension}: missing/not-estimable dimensions must not carry numeric values")
        return errors
    modality=measurement.get("modality")
    if modality not in DIMENSIONS[dimension]["modalities"]: errors.append(f"{dimension}: modality {modality!r} is incompatible")
    unit=measurement.get("unit")
    if unit not in DIMENSIONS[dimension]["units"]: errors.append(f"{dimension}: unit {unit!r} is incompatible or unspecified")
    value=measurement.get("value")
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)): errors.append(f"{dimension}: measured/proxy value must be finite numeric")
    for key in ("method","sample_id","timepoint"):
        if not _is_text(measurement.get(key)): errors.append(f"{dimension}: {key} is required")
    if not _valid_hash(measurement.get("source_artifact_sha256")): errors.append(f"{dimension}: source_artifact_sha256 must be SHA-256")
    if not isinstance(measurement.get("qc"),dict) or not measurement["qc"]: errors.append(f"{dimension}: qc metadata is required")
    calibration=measurement.get("calibration")
    if unit in {"uN","mN","um_per_s","µm/s","m_per_s"} and not calibration:
        errors.append(f"{dimension}: physical mechanical units require explicit calibration metadata")
    if status=="proxy" and not _is_text(measurement.get("interpretation_limit")):
        errors.append(f"{dimension}: proxy requires interpretation_limit")
    return errors

def build_maturity_profile(*,sample_id:str,timepoint:str,measurements:dict[str,dict[str,Any]]|None=None,context:dict[str,Any]|None=None)->dict[str,Any]:
    if not _is_text(sample_id) or not _is_text(timepoint): raise MaturityProfileError("sample_id and timepoint are required")
    supplied=measurements or {}
    unknown=set(supplied)-set(DIMENSIONS)
    if unknown: raise MaturityProfileError("unknown dimensions: "+", ".join(sorted(unknown)))
    dimensions={name:_empty(name) for name in DIMENSIONS}
    errors=[]
    for name,item in supplied.items():
        if not isinstance(item,dict): errors.append(f"{name}: measurement must be an object"); continue
        row={**_empty(name),**item,"dimension":name}
        row.setdefault("sample_id",sample_id); row.setdefault("timepoint",timepoint)
        errors.extend(validate_dimension(name,row)); dimensions[name]=row
    if errors: raise MaturityProfileError("\n".join(errors))
    manifest={"schema_version":"1.0","profile_version":PROFILE_VERSION,"sample_id":sample_id,"timepoint":timepoint,"dimensions":dimensions,"context":context or {},"composite":{"status":"not_computed","score":None,"reason":"no externally justified weighting and uncertainty model supplied"},"interpretation_policy":{"missingness_is_explicit":True,"early_marker_expression_does_not_equal_mature_function":True,"video_does_not_infer_electrophysiology_calcium_or_metabolism":True,"motion_index_is_not_force_without_calibration":True}}
    raw=json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    manifest["profile_sha256"]=hashlib.sha256(raw).hexdigest()
    return manifest

def video_only_profile(video_summary:dict[str,Any],*,sample_id:str,timepoint:str,source_artifact_sha256:str,method:str="MyoTrace optical-flow pipeline",qc:dict[str,Any]|None=None)->dict[str,Any]:
    """Project video output into mechanical proxy only; all other dimensions stay missing."""
    if not _valid_hash(source_artifact_sha256): raise MaturityProfileError("source_artifact_sha256 must be SHA-256")
    n=int(video_summary.get("n_beats",video_summary.get("n_events",0)) or 0)
    value=video_summary.get("mean_bpm")
    if n<=0 or value is None:
        mechanical={"status":"not_estimable","value":None,"reason":"static/no verified motion events; no beat-rate estimate"}
    else:
        mechanical={"status":"proxy","value":float(value),"reason":"motion-event rate proxy; events are not verified cardiac beats","interpretation_limit":"motion events are not verified cardiac beats or calibrated contractility"}
    if mechanical["status"]=="proxy":
        mechanical.update({"unit":"motion_index","modality":"mechanical","method":method,"calibration":None,"sample_id":sample_id,"timepoint":timepoint,"qc":qc or {"status":"not_supplied"},"source_artifact_sha256":source_artifact_sha256})
    else:
        mechanical.update({"dimension":"contractility_beat_mechanics"})
    return build_maturity_profile(sample_id=sample_id,timepoint=timepoint,measurements={"contractility_beat_mechanics":mechanical},context={"source":"video_only","video_measurement_status":video_summary.get("measurement_status","unknown")})
