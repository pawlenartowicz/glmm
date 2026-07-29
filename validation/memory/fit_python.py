#!/usr/bin/env python3
"""Python-port runner for the large synthetic memory-measurement models
(models.json). NOT a validation rung -- see models.json's header. Fits an
arbitrary generated CSV + formula through the installed `glmm` wheel (the same
package engines/glmm_python.py drives for the 43 curated rungs), so memory.sh
can wrap this in `/usr/bin/time -f '%M'` one process per (engine, model).

    python fit_python.py <csv> <formula> <family> <link> <factors_csv> <nagq>

factors_csv: comma-separated grouping column names, read as strings so they
lower to factors with lexicographic level order (glmm.fit's convention for a
plain string column). Prints one status line -- stdout is not compared to
anything.
"""

import sys

import pandas as pd

import glmm


def main():
    csv, formula, family, link, factors_csv, nagq = sys.argv[1:7]
    factors = [f for f in factors_csv.split(",") if f]
    nagq = int(nagq)

    dtype = {f: str for f in factors}
    df = pd.read_csv(csv, dtype=dtype)

    kwargs = {}
    if link:
        kwargs["link"] = link
    if nagq != 1:
        kwargs["nagq"] = nagq

    m = glmm.fit(df, formula, family, **kwargs)
    print(f"n={len(df)} converged={m.converged}")


if __name__ == "__main__":
    main()
