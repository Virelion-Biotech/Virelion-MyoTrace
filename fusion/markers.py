from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarkerDefinition:
    name: str
    mature_direction: str = "higher"
    unit: str = "relative"
    description: str = ""

    def __post_init__(self) -> None:
        if self.mature_direction not in {"higher", "lower"}:
            raise ValueError("mature_direction must be 'higher' or 'lower'")


DEFAULT_MARKERS: tuple[MarkerDefinition, ...] = (
    MarkerDefinition("MYH7_MYH6_ratio", "higher", "ratio", "MYH7/MYH6 expression ratio"),
    MarkerDefinition("TNNI3_TNNI1_ratio", "higher", "ratio", "TNNI3/TNNI1 expression ratio"),
    MarkerDefinition("ATP2A2_SERCA2A", "higher", "relative", "SERCA2A expression/readout"),
    MarkerDefinition("GJA1_CX43", "higher", "relative", "GJA1/Cx43 expression/readout"),
)


def marker_panel_dict(values: dict[str, float]) -> dict[str, float]:
    """Validate and normalize a marker dictionary without imposing biological cutoffs."""
    clean: dict[str, float] = {}
    for key, value in values.items():
        v = float(value)
        if v != v:  # NaN is allowed for missing downstream imputation, but retained explicitly.
            clean[key] = v
        else:
            clean[key] = v
    return clean
