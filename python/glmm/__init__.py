"""GLMM Python port — formula + data -> fit.

`glmm.fit` parses `formula` against `data`'s columns through the Rust
`glmm-formula` parser, fits via the `glmm` kernel (through the `glmm._native`
PyO3 extension), and returns a `Fit`. Four combinations from the API spec's
family/link table are GLMM 0.1.1 ("approved design, not yet implemented in the
kernel" — docs/GLMM/0.1.1/2026-07-04-glmm-0.1.1-families-links-spec.md) and
raise a clean `NotImplementedError`: `family="inversegaussian"`,
`link="cloglog"`, quasi-likelihood `dispersion=` on binomial/poisson, and
`theta=<float>` (no kernel hook exists yet to seed the negative-binomial search
— only the default `theta=None` cold-start is supported).

Public surface is exactly two names: `fit` and `Fit`.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from glmm import _native

__all__ = ["fit", "Fit"]


# Family table — mirrors the API spec §3.2 and GLMM/src/family.rs.
_FAMILIES = {
    "gaussian": {"default_link": "identity", "links": {"identity"}},
    "binomial": {"default_link": "logit", "links": {"logit", "probit", "cloglog"}},
    "poisson": {"default_link": "log", "links": {"log"}},
    "gamma": {"default_link": "log", "links": {"log", "inverse"}},
    "negativebinomial": {"default_link": "log", "links": {"log"}},
    "inversegaussian": {"default_link": "log", "links": {"log", "inverse_squared"}},
}

# Families where `dispersion=` is meaningful: phi families (gamma,
# inversegaussian) plus binomial/poisson, where "estimate"/float means
# quasi-likelihood (GLM only). gaussian and negativebinomial have no phi
# knob (negbin's parameter is theta).
_DISPERSION_FAMILIES = {"binomial", "poisson", "gamma", "inversegaussian"}

_MAX_NAGQ = 25  # mirrors GLMM/src/consts.rs::MAX_NAGQ — change together


def _columns(data):
    """Extract {name: list-of-values} from dict / pandas / polars / pyarrow
    (spec §3.1) without a hard dependency on any dataframe library."""
    if isinstance(data, dict):
        return {k: list(v) for k, v in data.items()}
    if hasattr(data, "column_names"):  # pyarrow Table
        return {c: data.column(c).to_pylist() for c in data.column_names}
    if hasattr(data, "columns"):  # pandas / polars DataFrame
        return {c: list(data[c]) for c in data.columns}
    raise TypeError(f"data must be a dict, DataFrame, or pyarrow Table; got {type(data).__name__}")


@dataclass
class Fit:
    """Fit result — mirrors the Rust `Fit` (GLMM/src/fit.rs), plus coefficient
    names from the formula. Returned by `fit`, not constructed by callers."""

    beta: np.ndarray  # (p,) fixed-effect estimates
    se: np.ndarray  # (p,) standard errors; NaN for non-targets
    tau2: np.ndarray  # legacy per-element RE variances (q=1 only) — prefer varcorr
    varcorr: list  # per grouping: vech-packed (column-major lower-tri) RE covariance
    stddev_se: (
        np.ndarray
    )  # SE of each RE stddev, theta layout (not beta-aligned); NaN where unavailable
    aliased: np.ndarray  # (p,) bool — rank-deficient columns dropped (lme4's NA coefficients)
    dispersion: float  # phi (gamma / inverse-gaussian) / theta (negbin) / 1.0 otherwise
    converged: bool
    names: list  # coefficient names, aligned with beta

    def stddev_corr(self, group_idx):
        """Split grouping `group_idx`'s vech-packed covariance into
        (stddevs, correlation matrix) — mirrors Rust `Fit::stddev_corr`
        (GLMM/src/fit.rs): column-major lower-triangular vech."""
        vech = np.asarray(self.varcorr[group_idx], dtype=float)
        m = len(vech)
        q = (math.isqrt(1 + 8 * m) - 1) // 2
        if q * (q + 1) // 2 != m:
            raise ValueError(f"varcorr[{group_idx}] is not a valid vech (len {m})")

        def idx(r, c):
            return c * q - (c * c - c) // 2 + (r - c)

        stddev = np.array([math.sqrt(vech[idx(i, i)]) for i in range(q)])
        corr = np.eye(q)
        for c in range(q):
            for r in range(c + 1, q):
                rho = vech[idx(r, c)] / (stddev[r] * stddev[c])
                corr[r, c] = corr[c, r] = rho
        return stddev, corr

    def summary(self):
        """Build the coefficient table, print it, and return it as text.

        z/p are Wald (z = beta/se, p = 2*(1 - Phi(|z|)) = erfc(|z|/sqrt 2)),
        derived here — the Rust `Fit` carries no p-values. Wald-z (not t)
        matches the GLM/GLMM convention and the absence of a residual-df
        field on the kernel output."""
        beta = np.asarray(self.beta, dtype=float)
        se = np.asarray(self.se, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = beta / se
        p = np.array(
            [math.erfc(abs(v) / math.sqrt(2.0)) if math.isfinite(v) else math.nan for v in z]
        )

        name_w = max(12, max((len(n) for n in self.names), default=4))
        lines = [f"{'name':<{name_w}} {'estimate':>12} {'std.error':>12} {'z':>10} {'p':>10}"]
        for i, name in enumerate(self.names):
            if self.aliased[i]:
                est = se_i = z_i = p_i = math.nan  # lme4 prints NA here
            else:
                est, se_i, z_i, p_i = beta[i], se[i], z[i], p[i]
            lines.append(f"{name:<{name_w}} {est:>12.4g} {se_i:>12.4g} {z_i:>10.4g} {p_i:>10.4g}")

        if len(self.varcorr):
            lines.append("")
            lines.append("Random effects:")
            # stddev_se is theta-layout: one entry per vech element per
            # grouping; a stddev's SE sits at its diagonal vech position.
            theta_off = 0
            for g in range(len(self.varcorr)):
                sd, corr = self.stddev_corr(g)
                q = len(sd)
                lines.append(f"  group {g}:")
                for i in range(q):
                    di = theta_off + (i * q - (i * i - i) // 2)
                    sd_se = self.stddev_se[di] if di < len(self.stddev_se) else math.nan
                    corr_cells = " ".join(f"{corr[i][j]:>8.3f}" for j in range(i + 1))
                    lines.append(f"    sd {sd[i]:>10.4g}  se {sd_se:>10.4g}  corr {corr_cells}")
                theta_off += q * (q + 1) // 2

        lines.append("")
        lines.append(f"dispersion: {self.dispersion:.6g}   converged: {self.converged}")
        text = "\n".join(lines)
        print(text)
        return text


def fit(
    data,
    formula,
    family="gaussian",
    *,
    link=None,
    dispersion=None,
    theta=None,
    weights=None,
    wald_se="hessian",
    nagq=1,
    targets=None,
    warm_start=None,
):
    """Fit `formula` against `data`'s columns and return a `Fit`.

    data: dict[str, array-like]; DataFrame / Arrow Table also accepted (duck-typed).
    formula: R-style string, e.g. "y ~ x + z + (1 + x | g)".
    family: gaussian | binomial | poisson | gamma | negativebinomial | inversegaussian.
    See the API spec for the remaining knobs.
    """
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(_FAMILIES)}")
    fam = _FAMILIES[family]
    if link is None:
        link = fam["default_link"]
    elif link not in fam["links"]:
        raise ValueError(
            f"family {family!r} does not support link {link!r}; "
            f"expected one of {sorted(fam['links'])}"
        )

    # `|` marks a random-effect term, so its presence is the mixed/GLM split
    # — decidable without the (M6, Rust-side) formula parser.
    mixed = "|" in formula

    if family == "inversegaussian" and mixed:
        raise ValueError(
            "family 'inversegaussian' is GLM-only: random-effect terms "
            "(`(... | g)`) are not supported"
        )

    if wald_se not in ("hessian", "rx"):
        raise ValueError(f"wald_se must be 'hessian' or 'rx', got {wald_se!r}")

    if not (
        isinstance(nagq, int)
        and not isinstance(nagq, bool)
        and 1 <= nagq <= _MAX_NAGQ
        and nagq % 2 == 1
    ):
        raise ValueError(f"nagq must be an odd integer in 1..={_MAX_NAGQ}, got {nagq!r}")

    # Valid-but-inapplicable options: warn and strip (spec §3.5). The kernel
    # boundary-faults on inapplicable options and a Rust panic across the FFI
    # is not an acceptable user error, so nothing inapplicable may reach it.
    if dispersion is not None and family not in _DISPERSION_FAMILIES:
        warnings.warn(
            f"dispersion= is not applicable to family {family!r}; ignored",
            stacklevel=2,
        )
        dispersion = None
    if dispersion is not None:
        if not (
            dispersion == "estimate"
            or (isinstance(dispersion, (int, float)) and not isinstance(dispersion, bool))
        ):
            raise ValueError(f"dispersion must be None, 'estimate', or a float, got {dispersion!r}")
        if family in ("binomial", "poisson") and mixed:
            warnings.warn(
                "quasi-likelihood dispersion on binomial/poisson is GLM-only; "
                "ignored for a mixed formula",
                stacklevel=2,
            )
            dispersion = None
    if theta is not None and family != "negativebinomial":
        warnings.warn(
            "theta= applies only to family 'negativebinomial'; ignored",
            stacklevel=2,
        )
        theta = None
    if warm_start is not None:
        if not isinstance(warm_start, dict):
            raise TypeError(
                "warm_start must be a dict with keys 'beta'/'theta', "
                f"got {type(warm_start).__name__}"
            )
        unknown = set(warm_start) - {"beta", "theta"}
        if unknown:
            warnings.warn(f"warm_start keys ignored: {sorted(unknown)}", stacklevel=2)
            warm_start = {k: v for k, v in warm_start.items() if k in ("beta", "theta")}

    # --- kernel-gap checks: GLMM 0.1.1 not yet implemented (see module docstring) ---
    if family == "inversegaussian":
        raise NotImplementedError(
            "family 'inversegaussian' requires GLMM 0.1.1; not yet implemented in the kernel"
        )
    if link == "cloglog":
        raise NotImplementedError(
            "link 'cloglog' requires GLMM 0.1.1; not yet implemented in the kernel"
        )
    if dispersion == "estimate" and family == "gamma":
        # gamma's family default (dispersion=None) already computes the
        # Pearson estimate, so "estimate" needs no distinct kernel state.
        dispersion = None
    if family in ("binomial", "poisson") and dispersion is not None:
        raise NotImplementedError(
            f"quasi-likelihood dispersion on family {family!r} requires GLMM "
            "0.1.1; not yet implemented in the kernel"
        )
    if theta is not None:
        raise NotImplementedError(
            "theta= (negative-binomial shape seed) has no kernel hook yet; "
            "only theta=None (cold-start search) is supported"
        )

    numeric_columns = {}
    factor_columns = {}
    for name, values in _columns(data).items():
        if values and isinstance(values[0], str):
            factor_columns[name] = [str(v) for v in values]
        else:
            numeric_columns[name] = [float(v) for v in values]

    warm_start_pair = None
    if warm_start is not None:
        warm_start_pair = (
            [float(v) for v in warm_start.get("beta", [])],
            [float(v) for v in warm_start.get("theta", [])],
        )

    beta, se, tau2, varcorr, stddev_se, aliased, disp, converged, names = _native.fit(
        formula,
        numeric_columns,
        factor_columns,
        family,
        link,
        wald_se,
        nagq,
        dispersion,
        [float(w) for w in weights] if weights is not None else None,
        list(targets) if targets is not None else None,
        warm_start_pair,
    )
    # Wrap PyO3's plain-list tuple fields back into the array types Fit's
    # dataclass documents (no numpy Rust dep — the native call returns plain lists).
    return Fit(
        np.asarray(beta, dtype=float),
        np.asarray(se, dtype=float),
        np.asarray(tau2, dtype=float),
        varcorr,
        np.asarray(stddev_se, dtype=float),
        np.asarray(aliased, dtype=bool),
        disp,
        converged,
        names,
    )
