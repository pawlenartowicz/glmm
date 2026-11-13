"""The Python front end of the input port gate: every input of `inputs.json`
made through the Python port, each written to stdout as one JSON line: the
input's `id`, `status` ("accepted" or "refused"), `message` (the refusal
text, "" when accepted) and `warnings` (the text of every warning the call
raised, in order). Mirrors `input_ports.R` input for input: change together.
Each builder below passes its input in this port's own idiom (numpy scalars,
a non-string dict key); the formula, family and table come from
`inputs.json`.

Run from the repository root:
    glmm-py/venv/bin/python validation/tools/input_ports/input_ports.py > input_py.jsonl
"""

import json
import os
import sys
import warnings

import glmm
import numpy as np

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs.json")) as f:
    SPEC = json.load(f)
DATA = {
    name: col if isinstance(col[0], str) else np.array([np.nan if v is None else v for v in col])
    for name, col in SPEC["data"].items()
}
N = len(SPEC["subset"])


def fit_input(row, data=DATA, **kwargs):
    return glmm.fit(data, row["formula"], row["family"], **kwargs)


def with_na_at_row_4(value):
    out = np.full(N, value, dtype=float)
    out[3] = np.nan
    return out


# Python has no `subset=`: these are the rows R's subset keeps, its NA dropping that row.
KEPT = [i for i, s in enumerate(SPEC["subset"]) if s is True]

BUILD = {
    "nagq_numpy_integer": lambda row: fit_input(row, nagq=np.int64(5)),
    "init_theta_numpy_integer": lambda row: fit_input(row, init_theta=np.int64(3)),
    "dispersion_numpy_integer": lambda row: fit_input(row, dispersion=np.int64(2)),
    "column_name_not_a_string": lambda row: fit_input(row, data={**DATA, 0: DATA["x"]}),
    "subset_with_na": lambda row: fit_input(
        row,
        data={k: [v[i] for i in KEPT] if isinstance(v, list) else v[KEPT] for k, v in DATA.items()},
    ),
    "weights_with_na": lambda row: fit_input(row, weights=with_na_at_row_4(1.0)),
    "offset_with_na": lambda row: fit_input(row, offset=with_na_at_row_4(0.0)),
    "interaction_slope": fit_input,
}

ids = [row["id"] for row in SPEC["inputs"]]
if set(BUILD) != set(ids):
    sys.exit(f"builders and inputs.json differ: {sorted(set(BUILD) ^ set(ids))}")

for row in SPEC["inputs"]:
    status, message = "accepted", ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            BUILD[row["id"]](row)
        except Exception as e:  # noqa: BLE001 - a refusal of any class is recorded by its text
            status, message = "refused", str(e)
    rec = {"id": row["id"], "status": status, "message": message}
    rec["warnings"] = [str(w.message) for w in caught]
    print(json.dumps(rec))
