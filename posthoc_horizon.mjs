// POST-HOC (not pre-registered): is the map's horizon (gamma) what limits G2 and G4b?
// Same protocol as gates.mjs G2 / G4b, with gamma = 0.99 and a longer exploration (6000 steps).
import { appendFileSync } from "node:fs";
import { Baby, makeWorld, bfs, freeCells, mulberry32, approach, idx } from "./baby.js";
const log = (s) => { console.log(s); appendFileSync("results/posthoc_log.txt", s + "\n"); };
const mean = (a) => a.reduce((p, v) => p + v, 0) / a.length;
for (const [gamma, explore] of [[0.9, 3000], [0.99, 3000], [0.99, 6000]]) {
  const g2 = [], near = [], far = [], g4b = [];
  for (let seed = 0; seed < 10; seed++) {
    const rng = mulberry32(2000 + seed), w = makeWorld(), b = new Baby({ seed, grid: false });
    b.gamma = gamma;
    let s = freeCells(w)[Math.floor(rng() * freeCells(w).length)]; b.grid.anchor(s);
    for (let i = 0; i < explore; i++) s = b.act(w, s, Math.floor(rng() * 4), true);
    for (const k of ["food", "water"]) { const t = k == "food" ? w.food : w.water, ap = approach(w, t); b.grid.anchor(ap.s); b.act(w, ap.s, ap.a, true); b.taste(t, k, 1); }
    const d = bfs(w, w.food), f = freeCells(w).filter((x) => d[x] >= 4);
    for (let i = 0; i < 20; i++) {
      const s0 = f[Math.floor(rng() * f.length)]; if (s0 === w.water) continue;
      for (const need of ["food", "water"]) {
        b.drives = need == "food" ? { food: 1, water: 0 } : { food: 0, water: 1 };
        const r = b.seek(w, s0, need, { useGrid: false }), ok = r.reached === need ? 1 : 0;
        g2.push(ok); (bfs(w, need == "food" ? w.food : w.water)[s0] <= 12 ? near : far).push(ok);
      }
    }
    b.drives = { food: 1, water: 0 };
    g4b.push(b.seek(w, idx(2, 10), "food", { useGrid: false }).steps);
  }
  log(`POSTHOC horizon gamma=${gamma} explore=${explore}: G2-style correct ${(100 * mean(g2)).toFixed(1)}% (near ${(100 * mean(near)).toFixed(1)}%, far ${(100 * mean(far)).toFixed(1)}%); map from (2,10) to food: ${g4b.join(" ")} steps (shortest 24)`);
}
