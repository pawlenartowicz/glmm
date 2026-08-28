"""Namespace alias: `import faststats.glmm` yields the `glmm` module itself
(sys.modules alias — one source of truth, no duplicated surface)."""

import sys

import glmm as _glmm

sys.modules[__name__] = _glmm
