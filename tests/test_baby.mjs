// Unit tests: node tests/test_baby.mjs
import { GridRing, GridCode, makeWorld, trueSR, N, mulberry32, bfs, idx, FOOD } from "../baby.js";

let fails = 0;
const check = (name, ok, info = "") => { console.log(`${ok ? "ok  " : "FAIL"} ${name} ${info}`); if (!ok) fails++; };
const circ = (d) => Math.abs(((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);

// 1. genome-generated ring path-integrates exactly
{
  const rng = mulberry32(7), ring = new GridRing(5);
  let x = 0, maxErr = 0;
  ring.anchor(0);
  for (let i = 0; i < 2000; i++) {
    const v = [-1, 0, 1][Math.floor(rng() * 3)];
    ring.step(v); x += v;
    maxErr = Math.max(maxErr, circ(ring.phase() - 2 * Math.PI * x / 5));
  }
  check("genome ring W = X R(wv) X^T path-integrates", maxErr < 1e-9, `max phase error ${maxErr.toExponential(2)} rad over 2000 steps`);
}

// 2. generated weights equal the factored update
{
  const ring = new GridRing(7), Wm = ring.weights(1);
  ring.anchor(2);
  const r0 = ring.r.slice(), out = new Float64Array(16);
  for (let i = 0; i < 16; i++) { let t = 0; for (let j = 0; j < 16; j++) t += Wm[i * 16 + j] * r0[j]; out[i] = Math.max(0, t); }
  const s = out.reduce((a, b) => a + b, 0);
  ring.step(1);
  const d = Math.max(...out.map((v, i) => Math.abs(v / s - ring.r[i])));
  check("materialised W(v) matches the factored step", d < 1e-12, d.toExponential(2));
}

// 3. control: a random matrix with the same norm does not path-integrate
{
  const rng = mulberry32(3), ring = new GridRing(5), W0 = ring.weights(0);
  const fro = Math.sqrt(W0.reduce((a, v) => a + v * v, 0));
  const R = Float64Array.from({ length: 256 }, () => { const u = rng() || 1e-9, v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); });
  const fr = Math.sqrt(R.reduce((a, v) => a + v * v, 0));
  ring.randomW = R.map((v) => v * fro / fr);
  ring.anchor(0);
  let x = 0, err = 0;
  for (let i = 0; i < 200; i++) { ring.step(1); x += 1; err += circ(ring.phase() - 2 * Math.PI * x / 5); }
  check("random W of the same norm fails", err / 200 > 0.5, `mean phase error ${(err / 200).toFixed(2)} rad`);
}

// 4. displacement decode over the full world range
{
  const g = new GridCode();
  let ok = true;
  for (let x0 = 1; x0 < 12; x0++) for (let x1 = 1; x1 < 12; x1++) {
    g.anchor(idx(x1, 3)); const goal = g.phases();
    g.anchor(idx(x0, 3));
    if (g.displacement(goal)[0] !== x1 - x0) ok = false;
  }
  check("grid displacement decode is exact on every pair", ok);
}

// 5. true successor map rows sum to 1/(1-gamma)
{
  const w = makeWorld(), M = trueSR(w, 0.9);
  const s = idx(3, 3); let t = 0; for (let j = 0; j < N; j++) t += M[s * N + j];
  check("true SR row sum = 1/(1-γ)", Math.abs(t - 10) < 1e-6, t.toFixed(6));
  const d = bfs(w, idx(...FOOD));
  check("food reachable from the left room", d[idx(2, 10)] > 0, `${d[idx(2, 10)]} steps`);
}
process.exit(fails ? 1 : 0);
