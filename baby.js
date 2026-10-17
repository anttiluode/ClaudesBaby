// ClaudesBaby — a small creature built as nested operators (see PREREG.md).
// One module, used by the live page (index.html) and by the gates (gates.mjs) alike.
//
//   DNA (lifetime)        grid rings: W(v) = X R(ωv) Xᵀ, never stored entry by entry
//   hippocampus           successor map M, learned by TD
//   medial septum         theta: encoding phase writes experience, retrieval phase reads the map
//   striatum              reward map per need, learned by a dopamine prediction error
//   hypothalamus          drives -> mixing weights over needs
//   cortex (L5 gate)      V(s) = Σ_k m_k (M r_k)(s): basal map value x apical drive gate
//   lipids / myelin       not modelled (see README)

export const W = 13, H = 13, N = W * H;
export const ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];   // N E S W
export const ACTION_NAMES = ["N", "E", "S", "W"];
export const DOORS = { A: [6, 2], B: [6, 6], C: [6, 10] };
export const WATER = [2, 2], FOOD = [10, 10];
export const idx = (x, y) => y * W + x;
export const xy = (s) => [s % W, Math.floor(s / W)];

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ world
export function makeWorld(open = { A: true, B: false, C: false }) {
  const wall = new Uint8Array(N);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (x === 0 || y === 0 || x === W - 1 || y === H - 1 || x === 6) wall[idx(x, y)] = 1;
  const world = { wall, open: { ...open }, food: idx(...FOOD), water: idx(...WATER) };
  for (const [k, [x, y]] of Object.entries(DOORS)) wall[idx(x, y)] = open[k] ? 0 : 1;
  return world;
}

export function setDoor(world, name, isOpen) {
  world.open[name] = isOpen;
  world.wall[idx(...DOORS[name])] = isOpen ? 0 : 1;
}

export function move(world, s, a) {
  const [x, y] = xy(s), [dx, dy] = ACTIONS[a];
  const t = idx(x + dx, y + dy);
  return world.wall[t] ? s : t;
}

export function freeCells(world) {
  const out = [];
  for (let s = 0; s < N; s++) if (!world.wall[s]) out.push(s);
  return out;
}

export function bfs(world, from) {
  const d = new Int32Array(N).fill(-1), q = [from];
  d[from] = 0;
  while (q.length) {
    const s = q.shift();
    for (let a = 0; a < 4; a++) {
      const t = move(world, s, a);
      if (t !== s && d[t] < 0) { d[t] = d[s] + 1; q.push(t); }
    }
  }
  return d;
}

// True successor map of a uniform random walk: M = (I - γP)^-1, by iteration.
export function trueSR(world, gamma = 0.9, iters = 300) {
  let M = new Float64Array(N * N);
  for (let s = 0; s < N; s++) M[s * N + s] = 1;
  const nxt = new Int32Array(N * 4);
  for (let s = 0; s < N; s++) for (let a = 0; a < 4; a++) nxt[s * 4 + a] = move(world, s, a);
  for (let it = 0; it < iters; it++) {
    const M2 = new Float64Array(N * N);
    for (let s = 0; s < N; s++) {
      if (world.wall[s]) { M2[s * N + s] = 1; continue; }
      const row = s * N;
      M2[row + s] += 1;
      for (let a = 0; a < 4; a++) {
        const t = nxt[s * 4 + a] * N;
        for (let j = 0; j < N; j++) M2[row + j] += 0.25 * gamma * M[t + j];
      }
    }
    M = M2;
  }
  return M;
}

export function mapCorrelation(Mlearned, Mtrue, world) {
  const free = freeCells(world), a = [], b = [];
  for (const s of free) for (const t of free) { a.push(Mlearned[s * N + t]); b.push(Mtrue[s * N + t]); }
  const n = a.length, ma = a.reduce((p, v) => p + v, 0) / n, mb = b.reduce((p, v) => p + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const u = a[i] - ma, v = b[i] - mb; sab += u * v; saa += u * u; sbb += v * v; }
  return sab / Math.sqrt(saa * sbb);
}

// ------------------------------------------------------------------ entorhinal grid rings (DNA row)
// Each neuron i carries two "gene" features X_i = (cos θ_i, sin θ_i). The connection matrix is
// generated, not stored: W(v) = (2/n) X R(2πv/period) Xᵀ, a bilinear form of the two ends'
// features with a 2x2 rule matrix that running velocity rotates.
export class GridRing {
  constructor(period, n = 16) {
    this.period = period; this.n = n;
    this.cos = Float64Array.from({ length: n }, (_, i) => Math.cos(2 * Math.PI * i / n));
    this.sin = Float64Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * i / n));
    this.r = new Float64Array(n);
    this.randomW = null;
    this.anchor(0);
  }
  anchor(pos) {                       // landmark reset: bump at the phase of position pos
    const ph = 2 * Math.PI * pos / this.period;
    for (let i = 0; i < this.n; i++) this.r[i] = Math.max(0, Math.cos(2 * Math.PI * i / this.n - ph));
    this.norm();
  }
  norm() { let s = 0; for (const v of this.r) s += v; if (s > 0) for (let i = 0; i < this.n; i++) this.r[i] /= s; }
  weights(v) {                        // the generated matrix, materialised only for inspection
    const w = 2 * Math.PI * v / this.period, c = Math.cos(w), s = Math.sin(w), n = this.n;
    const M = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      // X_i R X_jᵀ with R = [[c,-s],[s,c]]
      M[i * n + j] = (2 / n) * (this.cos[i] * (c * this.cos[j] - s * this.sin[j]) + this.sin[i] * (s * this.cos[j] + c * this.sin[j]));
    }
    return M;
  }
  step(v) {
    const n = this.n, out = new Float64Array(n);
    if (this.randomW) {               // control: a random matrix with the same norm
      for (let i = 0; i < n; i++) { let t = 0; for (let j = 0; j < n; j++) t += this.randomW[i * n + j] * this.r[j]; out[i] = Math.max(0, t); }
    } else {                          // apply W(v) through its factors: X (R (Xᵀ r))
      let a = 0, b = 0;
      for (let j = 0; j < n; j++) { a += this.cos[j] * this.r[j]; b += this.sin[j] * this.r[j]; }
      const w = 2 * Math.PI * v / this.period, c = Math.cos(w), s = Math.sin(w);
      const a2 = c * a - s * b, b2 = s * a + c * b;
      for (let i = 0; i < n; i++) out[i] = Math.max(0, (2 / n) * (this.cos[i] * a2 + this.sin[i] * b2));
    }
    this.r = out; this.norm();
  }
  phase() {
    let a = 0, b = 0;
    for (let j = 0; j < this.n; j++) { a += this.cos[j] * this.r[j]; b += this.sin[j] * this.r[j]; }
    return Math.atan2(b, a);
  }
}

const PERIODS = [3, 5, 7];
const circ = (d) => { d = ((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; return Math.abs(d); };

export class GridCode {
  constructor() { this.rx = PERIODS.map((p) => new GridRing(p)); this.ry = PERIODS.map((p) => new GridRing(p)); }
  anchor(s) { const [x, y] = xy(s); this.rx.forEach((r) => r.anchor(x)); this.ry.forEach((r) => r.anchor(y)); }
  step(dx, dy) { this.rx.forEach((r) => r.step(dx)); this.ry.forEach((r) => r.step(dy)); }
  phases() { return { x: this.rx.map((r) => r.phase()), y: this.ry.map((r) => r.phase()) }; }
  static decodeAxis(cur, goal) {     // displacement d in -12..12 with cur + 2πd/λ ≈ goal in every module
    let best = 0, be = Infinity;
    for (let d = -12; d <= 12; d++) {
      let e = 0;
      for (let k = 0; k < PERIODS.length; k++) e += circ(cur[k] + 2 * Math.PI * d / PERIODS[k] - goal[k]);
      if (e < be) { be = e; best = d; }
    }
    return best;
  }
  displacement(goal) {
    const p = this.phases();
    return [GridCode.decodeAxis(p.x, goal.x), GridCode.decodeAxis(p.y, goal.y)];
  }
}

// ------------------------------------------------------------------ the creature
export class Baby {
  constructor(opts = {}) {
    this.rng = mulberry32(opts.seed ?? 1);
    this.alpha = 0.1; this.gamma = 0.9; this.eta = 0.5;
    this.septum = opts.septum ?? true;      // theta separation of write and read
    this.gate = opts.gate ?? true;          // hypothalamic drive gate on the cortical readout
    this.gridNav = opts.grid ?? true;       // entorhinal vector navigation
    this.M = new Float64Array(N * N);
    for (let s = 0; s < N; s++) this.M[s * N + s] = 1;
    this.T = new Int32Array(N * 4).fill(-1);
    this.r = { food: new Float64Array(N), water: new Float64Array(N) };
    this.goalPhase = { food: null, water: null };
    this.grid = new GridCode();
    this.drives = { food: 0, water: 0 };    // hunger, thirst
    this.lastDelta = 0;
    this.theta = 0;                          // 0 = encoding phase, 1 = retrieval phase (for display)
  }

  // ---- septum: one theta cycle per behavioural step
  encode(s, a, s2) {                         // encoding phase: write what happened
    this.theta = 0;
    this.T[s * 4 + a] = s2;
    const M = this.M, row = s * N, row2 = s2 * N, al = this.alpha, g = this.gamma;
    const upd = new Float64Array(N);
    for (let j = 0; j < N; j++) upd[j] = (j === s ? 1 : 0) + g * M[row2 + j] - M[row + j];
    for (let j = 0; j < N; j++) M[row + j] += al * upd[j];
  }
  retrieve(s) {                              // retrieval phase: read; without the septum, also write the read-out
    this.theta = 1;
    if (this.septum) return;
    const M = this.M, row = s * N, al = this.alpha, g = this.gamma;
    const p = M.slice(row, row + N);
    for (let j = 0; j < N; j++) M[row + j] += al * ((j === s ? 1 : 0) + g * p[j] - p[j]);
  }

  // ---- striatum: dopamine prediction error at the visited place
  taste(s, kind, R = 1) {
    const d = R - this.r[kind][s];
    this.r[kind][s] += this.eta * d;
    this.lastDelta = d;
    this.goalPhase[kind] = this.grid.phases();
    return d;
  }

  // ---- hypothalamus -> cortex gate
  mix() {
    if (!this.gate) return { food: 0.5, water: 0.5 };
    const t = this.drives.food + this.drives.water;
    return t > 0 ? { food: this.drives.food / t, water: this.drives.water / t } : { food: 0.5, water: 0.5 };
  }
  valueField() {
    const m = this.mix(), w = new Float64Array(N);
    for (let j = 0; j < N; j++) w[j] = m.food * this.r.food[j] + m.water * this.r.water[j];
    const V = new Float64Array(N);
    for (let s = 0; s < N; s++) { let v = 0; const row = s * N; for (let j = 0; j < N; j++) if (w[j]) v += this.M[row + j] * w[j]; V[s] = v; }
    return V;
  }
  actionValue(V, s, a) {
    const t = this.T[s * 4 + a];
    if (t >= 0) return V[t];
    const [x, y] = xy(s), [dx, dy] = ACTIONS[a];
    return V[idx(x + dx, y + dy)];
  }
  greedy(V, s) {
    let best = -Infinity, cand = [];
    for (let a = 0; a < 4; a++) {
      const v = this.actionValue(V, s, a);
      if (v > best + 1e-12) { best = v; cand = [a]; } else if (Math.abs(v - best) <= 1e-12) cand.push(a);
    }
    return cand[Math.floor(this.rng() * cand.length)];
  }

  // ---- one step in the world (moves, integrates the grid, optionally learns)
  act(world, s, a, learn) {
    const s2 = move(world, s, a);
    if (s2 !== s) this.grid.step(ACTIONS[a][0], ACTIONS[a][1]);
    this.T[s * 4 + a] = s2;                 // noticing where a move led is perception, always on
    if (learn) { this.encode(s, a, s2); this.retrieve(s2); }
    return s2;
  }

  explore(world, s, steps) {
    for (let i = 0; i < steps; i++) s = this.act(world, s, Math.floor(this.rng() * 4), true);
    return s;
  }

  // Seek `kind` (or whichever need is highest). Returns {steps, reached, path}.
  seek(world, start, kind, { learn = false, cap = 150, useGrid = this.gridNav } = {}) {
    let s = start, nav = useGrid && this.goalPhase[kind] !== null;
    this.grid.anchor(start);
    const goals = { food: world.food, water: world.water }, path = [s];
    const V = this.valueField();
    for (let step = 0; step < cap; step++) {
      for (const [k, g] of Object.entries(goals)) if (s === g) return { steps: step, reached: k, path };
      let a;
      if (nav) {
        const [dx, dy] = this.grid.displacement(this.goalPhase[kind]);
        a = Math.abs(dx) >= Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0);
        if ((dx === 0 && dy === 0) || this.T[s * 4 + a] === s) { nav = false; a = this.greedy(V, s); }
      } else a = this.greedy(learn ? this.valueField() : V, s);
      const s2 = this.act(world, s, a, learn);
      if (nav && s2 === s) nav = false;        // vector heading blocked: follow the map from now on
      s = s2; path.push(s);
    }
    for (const [k, g] of Object.entries(goals)) if (s === g) return { steps: cap, reached: k, path };
    return { steps: cap, reached: null, path };
  }
}

// ------------------------------------------------------------------ baseline: model-free striatum only
export class QLearner {
  constructor(seed = 1) { this.rng = mulberry32(seed); this.Q = new Float64Array(N * 4); this.alpha = 0.1; this.gamma = 0.9; }
  update(s, a, R, s2) {
    let mx = -Infinity; for (let b = 0; b < 4; b++) mx = Math.max(mx, this.Q[s2 * 4 + b]);
    this.Q[s * 4 + a] += this.alpha * (R + this.gamma * mx - this.Q[s * 4 + a]);
  }
  greedy(s) {
    let best = -Infinity, cand = [];
    for (let a = 0; a < 4; a++) { const v = this.Q[s * 4 + a]; if (v > best + 1e-12) { best = v; cand = [a]; } else if (Math.abs(v - best) <= 1e-12) cand.push(a); }
    return cand[Math.floor(this.rng() * cand.length)];
  }
  seek(world, start, goal, cap = 150) {
    let s = start;
    for (let step = 0; step < cap; step++) { if (s === goal) return step; s = move(world, s, this.greedy(s)); }
    return s === goal ? cap : cap;
  }
}

// A cell next to a resource and the action that steps onto it (used for "tasting").
export function approach(world, target) {
  for (let a = 0; a < 4; a++) {
    const [dx, dy] = ACTIONS[a], [x, y] = xy(target), s = idx(x - dx, y - dy);
    if (!world.wall[s]) return { s, a };
  }
  return null;
}
