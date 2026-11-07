// Install a glmm wheel into Pyodide under Node and run two seeded glmm.fit
// calls. Usage: node pyodide-smoke.mjs <wheel>, run from a directory where the
// `pyodide` npm package is installed: it is resolved from the working
// directory, not from this file, so the repository needs no node_modules.
// Exits non-zero when the install or a fit fails.
import { createRequire } from "node:module";
import path from "node:path";

const { loadPyodide } = createRequire(path.join(process.cwd(), "/"))("pyodide");
const wheel = path.resolve(process.argv[2]);
const py = await loadPyodide();
py.FS.mkdirTree("/wheel");
py.FS.mount(py.FS.filesystems.NODEFS, { root: path.dirname(wheel) }, "/wheel");
await py.loadPackage(["numpy", "micropip"]);
py.setStdout({ batched: (s) => console.log(s) });
await py.runPythonAsync(`
import micropip
await micropip.install("emfs:/wheel/${path.basename(wheel)}")
import numpy as np, glmm
rng = np.random.default_rng(0)
n_id, per = 30, 20
ID = np.repeat(np.arange(n_id), per)
x = rng.normal(size=n_id * per)
eta = 0.5 + 0.8 * x + rng.normal(scale=0.7, size=n_id)[ID]
gaussian = {"y": eta + rng.normal(size=n_id * per), "x": x, "ID": ID}
# The binomial fit runs the GLMM path (PIRLS) and the logit link code, which
# wasm32 builds without fused multiply-add.
binomial = {"y": (rng.random(n_id * per) < 1 / (1 + np.exp(-eta))).astype(float), "x": x, "ID": ID}
for name, data, kw in [
    ("gaussian", gaussian, {}),
    ("binomial", binomial, {"family": "binomial", "link": "logit"}),
]:
    m = glmm.fit(data, "y ~ x + (1 | ID)", **kw)
    beta = np.asarray(m.beta)
    assert m.converged, name
    assert np.isfinite(beta).all(), (name, beta)
    print("glmm", name, "beta", beta)
`);
