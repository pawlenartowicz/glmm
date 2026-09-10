"""Shared helper for the worked-examples recipes: load a frozen golden JSON
from validation/goldens/ and print a pass/fail comparison against a fitted
value. Not part of the public glmm API — a doc-build convenience so every
recipe's printed comparison numbers come from the golden file, never typed by
hand.

Tolerance constants mirror validation/tol.R (an R script, not machine-parsed
from here, hence duplicated as constants with the source noted).
"""

import json
from pathlib import Path

GOLDENS_DIR = Path(__file__).resolve().parents[3] / "validation" / "goldens"

# validation/tol.R
TOL_BETA_REL = 1e-3
TOL_SE_REL = 1e-3
TOL_SE_HESSIAN_REL = 1e-3
TOL_STDDEV_REL = 1e-3
TOL_LOGLIK_ABS_LMM = 2e-6
TOL_LOGLIK_ABS_GLMM = 1e-3


def load_golden(name):
    with open(GOLDENS_DIR / f"{name}.json") as f:
        return json.load(f)


def rel_err(got, ref):
    return abs(got - ref) / abs(ref) if ref != 0 else abs(got - ref)


def check_rel(label, got, ref, tol):
    err = rel_err(got, ref)
    status = "PASS" if err <= tol else "FAIL"
    print(f"  {label:<28} got={got:<18.10g} ref={ref:<18.10g} rel_err={err:.3g}  tol={tol:.3g}  [{status}]")
    return err <= tol


def check_abs(label, got, ref, tol):
    err = abs(got - ref)
    status = "PASS" if err <= tol else "FAIL"
    print(f"  {label:<28} got={got:<18.10g} ref={ref:<18.10g} abs_err={err:.3g}  tol={tol:.3g}  [{status}]")
    return err <= tol
