"""MCAP exporter (self-describing multimodal log, ENGINE 1.1).

Logs three native-rate channels using JSON-Schema-encoded messages:
  * tactile_field  - BMM350 frames at the field rate (bmm_valid instants)
  * dynamics_accel - LIS2DTW12 acceleration at the master/accel rate
  * dynamics_temp  - LIS2DTW12 temperature at its ODR (temp_valid instants)
plus a metadata record carrying parameter provenance.

Timestamps are the hub microsecond clock (converted to ns for MCAP log_time).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...core_types import Episode

_TACTILE_SCHEMA = {
    "type": "object",
    "properties": {
        "t_us": {"type": "integer"},
        "B_uT": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "saturated": {"type": "boolean"},
    },
}
_ACCEL_SCHEMA = {
    "type": "object",
    "properties": {
        "t_us": {"type": "integer"},
        "accel_g": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    },
}
_TEMP_SCHEMA = {
    "type": "object",
    "properties": {
        "t_us": {"type": "integer"},
        "temp_degC": {"type": "array", "items": {"type": "number"}},
    },
}


def write(ep: Episode, out_dir: str | Path) -> Path:
    from mcap.writer import Writer  # imported lazily so the dep is optional at import time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ep.name}.mcap"

    t_us = ep.t_master_us
    B = ep.aligned["B_read_uT"]
    A = ep.aligned["accel_read_g"]
    Temp = ep.aligned["temp_read_c"]
    sat = ep.aligned["sat_flag"]
    bmm_valid = ep.aligned["bmm_valid"]
    temp_valid = ep.aligned["temp_valid"]

    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()

        def reg_channel(topic, schema):
            sid = w.register_schema(name=topic, encoding="jsonschema",
                                    data=json.dumps(schema).encode())
            return w.register_channel(topic=topic, message_encoding="json", schema_id=sid)

        ch_tac = reg_channel("tactile_field", _TACTILE_SCHEMA)
        ch_acc = reg_channel("dynamics_accel", _ACCEL_SCHEMA)
        ch_tmp = reg_channel("dynamics_temp", _TEMP_SCHEMA)

        w.add_metadata("sentrixsim", {
            "meta": json.dumps(ep.meta),
            "label_meta": json.dumps(ep.label_meta),
        })

        for t in range(ep.n_samples):
            ns = int(t_us[t]) * 1000
            if bmm_valid[t]:
                msg = {"t_us": int(t_us[t]), "B_uT": B[t].tolist(),
                       "saturated": bool(sat[t].any())}
                w.add_message(ch_tac, log_time=ns, publish_time=ns,
                              data=json.dumps(msg).encode())
            # accel logged every master sample
            amsg = {"t_us": int(t_us[t]), "accel_g": A[t].tolist()}
            w.add_message(ch_acc, log_time=ns, publish_time=ns,
                          data=json.dumps(amsg).encode())
            if temp_valid[t]:
                tmsg = {"t_us": int(t_us[t]), "temp_degC": Temp[t].tolist()}
                w.add_message(ch_tmp, log_time=ns, publish_time=ns,
                              data=json.dumps(tmsg).encode())
        w.finish()
    return path
