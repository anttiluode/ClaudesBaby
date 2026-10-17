// Pre-registered gates (PREREG.md).  node gates.mjs   -> results/results.json, results/gate_log.txt
import { writeFileSync, mkdirSync, appendFileSync } from "node:fs";
import { Baby, QLearner, makeWorld, setDoor, move, freeCells, bfs, trueSR, mapCorrelation, approach, mulberry32, idx, N } from "./baby.js";

mkdirSync("results", { recursive: true });
const LOG = "results/gate_log.txt";
const log = (s) => { console.log(s); appendFileSync(LOG, s + "\n"); };
const median = (a) => { const b = [...a].sort((x, y) => x - y), n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };
const mean = (a) => a.reduce((p, v) => p + v, 0) / a.length;
const SEEDS = [...Array(10).keys()];
const EXPLORE = 3000, CAP = 150;

function randomFree(world, rng) { const f = freeCells(world); return f[Math.floor(rng() * f.length)]; }
function startsFar(world, goal, rng, n, minDist = 4) {
  const d = bfs(world, goal), f = freeCells(world).filter((s) => d[s] >= minDist), out = [];
  for (let i = 0; i < n; i++) out.push(f[Math.floor(rng() * f.length)]);
  return out;
}
// Both agents see the same exploration trajectory.
function exploreBoth(world, baby, q, rng) {
  let s = randomFree(world, rng);
  baby.grid.anchor(s);
  for (let i = 0; i < EXPLORE; i++) {
    const a = Math.floor(rng() * 4), s2 = baby.act(world, s, a, true);
    if (q) q.update(s, a, 0, s2);
    s = s2;
  }
  return s;
}
function taste(world, baby, kind, q) {
  const target = kind === "food" ? world.food : world.water, ap = approach(world, target);
  baby.grid.anchor(ap.s);
  baby.act(world, ap.s, ap.a, true);
  baby.taste(target, kind, 1);
  if (q) q.update(ap.s, ap.a, 1, target);
}

log(`=== gates ${new Date().toISOString()}`);
const R = {};

// ---------------------------------------------------------------- G1 latent learning
{
  const sr = [], ql = [];
  for (const seed of SEEDS) {
    const rng = mulberry32(1000 + seed), world = makeWorld();
    const baby = new Baby({ seed, grid: false }), q = new QLearner(seed);
    exploreBoth(world, baby, q, rng);
    taste(world, baby, "food", q);
    baby.drives = { food: 1, water: 0 };
    const d = bfs(world, world.food);
    for (const s0 of startsFar(world, world.food, rng, 20)) {
      sr.push(baby.seek(world, s0, "food", { useGrid: false }).steps / d[s0]);
      ql.push(q.seek(world, s0, world.food, CAP) / d[s0]);
    }
  }
  const mS = median(sr), mQ = median(ql);
  R.G1 = { sr_median_ratio: mS, q_median_ratio: mQ, sr_mean_ratio: mean(sr), q_mean_ratio: mean(ql), pass: mS <= 1.5 && mQ >= 3 };
  log(`G1 latent learning: median steps/shortest  map ${mS.toFixed(2)}  model-free Q ${mQ.toFixed(2)}  -> ${R.G1.pass ? "PASS" : "FAIL"}`);
}

// ---------------------------------------------------------------- G2 need switch
{
  const res = { on: [], off: [] };
  for (const seed of SEEDS) {
    for (const gate of [true, false]) {
      const rng = mulberry32(2000 + seed), world = makeWorld();
      const baby = new Baby({ seed, gate, grid: false });
      exploreBoth(world, baby, null, rng);
      taste(world, baby, "food"); taste(world, baby, "water");
      for (const s0 of startsFar(world, world.food, rng, 20).filter((s) => s !== world.water)) {
        for (const need of ["food", "water"]) {
          baby.drives = need === "food" ? { food: 1, water: 0 } : { food: 0, water: 1 };
          const r = baby.seek(world, s0, need, { useGrid: false });
          res[gate ? "on" : "off"].push(r.reached === need ? 1 : 0);
        }
      }
    }
  }
  const on = mean(res.on), off = mean(res.off);
  R.G2 = { gate_on_correct: on, gate_off_correct: off, pass: on >= 0.9 && off <= 0.7 };
  log(`G2 need switch: correct resource first  gate on ${(100 * on).toFixed(1)}%  gate off ${(100 * off).toFixed(1)}%  -> ${R.G2.pass ? "PASS" : "FAIL"}`);
}

// ---------------------------------------------------------------- G3 theta separation
{
  const rows = [];
  for (const seed of SEEDS) {
    const row = { seed };
    for (const septum of [true, false]) {
      const rng = mulberry32(3000 + seed), world = makeWorld();
      const baby = new Baby({ seed, septum, grid: false });
      let s = exploreBoth(world, baby, null, rng);
      row[`pre_${septum ? "on" : "off"}`] = mapCorrelation(baby.M, trueSR(world), world);
      setDoor(world, "A", false); setDoor(world, "B", true);
      if (world.wall[s]) s = idx(5, 2);
      for (let i = 0; i < 600; i++) s = baby.act(world, s, Math.floor(rng() * 4), true);
      row[septum ? "on" : "off"] = mapCorrelation(baby.M, trueSR(world), world);
    }
    rows.push(row);
  }
  const wins = rows.filter((r) => r.on - r.off >= 0.05).length;
  R.G3 = { per_seed: rows, wins, mean_on: mean(rows.map((r) => r.on)), mean_off: mean(rows.map((r) => r.off)),
           mean_pre_on: mean(rows.map((r) => r.pre_on)), mean_pre_off: mean(rows.map((r) => r.pre_off)), pass: wins >= 8 };
  log(`G3 theta: map correlation after the change  septum on ${R.G3.mean_on.toFixed(3)}  off ${R.G3.mean_off.toFixed(3)}  (before the change: ${R.G3.mean_pre_on.toFixed(3)} / ${R.G3.mean_pre_off.toFixed(3)})  wins ${wins}/10 -> ${R.G3.pass ? "PASS" : "FAIL"}`);
}

// ---------------------------------------------------------------- G3b behaviour after the change
{
  const curves = { on: Array(10).fill(0), off: Array(10).fill(0) };
  for (const seed of SEEDS) {
    for (const septum of [true, false]) {
      const rng = mulberry32(3500 + seed), world = makeWorld();
      const baby = new Baby({ seed, septum, grid: false });
      exploreBoth(world, baby, null, rng);
      taste(world, baby, "food");
      setDoor(world, "A", false); setDoor(world, "B", true);
      baby.drives = { food: 1, water: 0 };
      for (let t = 0; t < 10; t++) {
        const s0 = idx(2, 2 + Math.floor(rng() * 9));
        curves[septum ? "on" : "off"][t] += baby.seek(world, s0, "food", { learn: true, useGrid: false }).steps / SEEDS.length;
      }
    }
  }
  R.G3b = curves;
  log(`G3b steps per trial after the change, septum on:  ${curves.on.map((v) => v.toFixed(0)).join(" ")}`);
  log(`G3b steps per trial after the change, septum off: ${curves.off.map((v) => v.toFixed(0)).join(" ")}`);
}

// ---------------------------------------------------------------- G4 shortcut, G4b no-shortcut cost
{
  const g4 = { grid: [], map: [] }, g4b = { grid: [], map: [] };
  for (const seed of SEEDS) {
    for (const shortcut of [true, false]) {
      for (const useGrid of [true, false]) {
        const rng = mulberry32(4000 + seed), world = makeWorld();
        const baby = new Baby({ seed, grid: useGrid });
        exploreBoth(world, baby, null, rng);
        taste(world, baby, "food");
        if (shortcut) setDoor(world, "C", true);
        baby.drives = { food: 1, water: 0 };
        const s0 = idx(2, 10), d = bfs(world, world.food)[s0];
        const steps = baby.seek(world, s0, "food", { useGrid }).steps;
        (shortcut ? g4 : g4b)[useGrid ? "grid" : "map"].push(shortcut ? steps / d : steps);
      }
    }
  }
  const gm = median(g4.grid), mm = median(g4.map);
  R.G4 = { grid_median_ratio: gm, map_median_ratio: mm, pass: gm <= 1.3 && mm >= 1.3 };
  const ratio = median(g4b.grid.map((v, i) => v / g4b.map[i]));
  R.G4b = { grid_steps: g4b.grid, map_steps: g4b.map, median_grid_over_map: ratio, pass: ratio <= 1.3 };
  log(`G4 shortcut through door C: steps/new shortest  grid+map ${gm.toFixed(2)}  map alone ${mm.toFixed(2)}  -> ${R.G4.pass ? "PASS" : "FAIL"}`);
  log(`G4b no shortcut: grid+map steps / map steps (median) ${ratio.toFixed(2)}  -> ${R.G4b.pass ? "PASS" : "FAIL"}`);
}

writeFileSync("results/results.json", JSON.stringify(R, null, 1));
log("done");
