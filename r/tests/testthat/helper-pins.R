# The band every pin in this suite is compared at.
#
# These are pins of THIS PACKAGE'S OWN OUTPUT, not agreement with another
# engine: the same kernel, the same data, the same design. So the band is a
# same-answer band, and it has to cover only two things -- a different CPU, and
# a different build of the same sources. It is the crate kernel's measured
# cross-architecture band for the iterative fit paths.
#
# One number, three homes: TOL$ci_ref_rel in the grid's tolerance table, the
# CI_REF_REL constant in the crate's own reference check, and this one. Change
# all three together. The grid's table carries the measurement.
#
# Never widened to turn a red run green. A pin outside this band means the
# answer moved or the band does not cover the machine, and both are reported,
# not absorbed.
CI_REF_REL <- 1e-7
