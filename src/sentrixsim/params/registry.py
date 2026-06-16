"""Parameter registry: the structural guarantee that no UNKNOWN parameter is
silently invented.

Every simulator parameter is loaded from configs/parameters.yaml as a
``Parameter`` carrying its tier (KNOWN/ESTIMATED/UNKNOWN), confidence, units and
provenance.  Reading a disabled UNKNOWN parameter raises ``UnknownParameterError``
unless ``allow_placeholders`` is set, in which case the run is marked
``physics_fidelity = placeholder`` and the placeholder_range midpoint is used.

Assumptions
-----------
* The YAML registry is authoritative; code never hard-codes a physical value.

Limitations
-----------
* Confidence scores are human-assigned (see README confidence scale), not learned.

Hardware-upgrade path
---------------------
* When a measurement lands, edit the YAML entry: set ``value``, raise
  ``confidence``, change ``tier`` to KNOWN, set ``enabled: true``.  No code change.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ParamTier(str, Enum):
    KNOWN = "KNOWN"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


class UnknownParameterError(RuntimeError):
    """Raised when a disabled UNKNOWN parameter is read without placeholders."""


class Parameter(BaseModel):
    name: str
    value: Any = None
    units: str = "none"
    tier: ParamTier = ParamTier.ESTIMATED
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    origin: str = "unspecified"
    enabled: bool = True
    placeholder_range: list[float] | None = None
    notes: str | None = None

    def resolved(self, allow_placeholders: bool = False) -> Any:
        """Return the usable value, enforcing the no-silent-invention rule."""
        if self.tier == ParamTier.UNKNOWN and (not self.enabled or self.value is None):
            if not allow_placeholders:
                raise UnknownParameterError(
                    f"Parameter '{self.name}' is UNKNOWN and disabled "
                    f"(origin: {self.origin}). Enable --allow-placeholders to use "
                    f"its placeholder_range, or supply a measured value."
                )
            if self.placeholder_range:
                lo, hi = self.placeholder_range
                return 0.5 * (lo + hi)
            return self.value
        return self.value


class ParameterRegistry:
    def __init__(self, params: dict[str, Parameter], allow_placeholders: bool = False):
        self._p = params
        self.allow_placeholders = allow_placeholders
        self.used_placeholder = False

    @classmethod
    def load(cls, path: str | Path, allow_placeholders: bool = False) -> "ParameterRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        params = {name: Parameter(name=name, **spec) for name, spec in raw.items()}
        return cls(params, allow_placeholders=allow_placeholders)

    def param(self, name: str) -> Parameter:
        if name not in self._p:
            raise KeyError(f"Unknown parameter key: {name}")
        return self._p[name]

    def get(self, name: str) -> Any:
        p = self.param(name)
        val = p.resolved(self.allow_placeholders)
        if p.tier == ParamTier.UNKNOWN and self.allow_placeholders and (
            not p.enabled or p.value is None
        ):
            self.used_placeholder = True
        return val

    def physics_fidelity(self) -> str:
        """`relative` by default; `placeholder` if any UNKNOWN placeholder was used."""
        return "placeholder" if self.used_placeholder else "relative"

    def provenance_table(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "value": p.value,
                "units": p.units,
                "tier": p.tier.value,
                "confidence": p.confidence,
                "origin": p.origin,
                "enabled": p.enabled,
            }
            for p in self._p.values()
        ]

    def counts(self) -> dict[str, int]:
        out = {t.value: 0 for t in ParamTier}
        for p in self._p.values():
            out[p.tier.value] += 1
        return out
