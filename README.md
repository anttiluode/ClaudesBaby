# ClaudesBaby

**A creature made of operators.** It lives in a two-room world, explores, gets hungry and thirsty,
and finds food and water. Each brain region is one switchable part with its own matrix, taken from
the "how does the brain translate to matrices" conversation (17 Oct 2026).

[Watch it live](https://anttiluode.github.io/ClaudesBaby/): open doors, switch parts off, make it hungry.

```
node tests/test_baby.mjs     # unit tests (grid rings, decoding, true successor map)
node gates.mjs               # PREREG.md gates, ~10 s
node posthoc_horizon.mjs     # the post-hoc horizon check
python3 -m http.server       # then open http://localhost:8000
```

No dependencies: one ES module (`baby.js`) runs in Node for the gates and in the browser for the page.

## The parts

| timescale | region | its matrix in the creature |
|---|---|---|
| lifetime | entorhinal grid (DNA) | W(v) = X R(ωv) Xᵀ. 16 neurons per ring, each with two "gene" features (cos θ, sin θ). A 2×2 rule matrix rotated by running speed generates every connection; nothing is stored entry by entry. 3 modules (periods 3, 5, 7 cells) × 2 axes. |
| minutes–days | hippocampus | successor map M (169×169), learned by TD from every step |
| ~100 ms | medial septum | one theta cycle per step: encoding phase writes the experienced transition, retrieval phase reads the map. Off: the read-out is also written back as if experienced. |
| minutes | striatum | one reward map per need, changed by a dopamine prediction error δ = reward − expectation |
| seconds–hours | hypothalamus | hunger and thirst become mixing weights mₖ |
| milliseconds | cortex (L5 gate) | V(s) = Σₖ mₖ (M rₖ)(s): basal map value per need × apical drive gate |
| days–months | lipids, myelin | **not modelled**. They would set time constants and conduction delays; the 2026 lipid atlas (539 lipizones, Nature) has no functional data to constrain them. |

The DNA row is the bilinear "genetic model of the connectome" form (W ≈ X O Xᵀ, Kovács, Barabási &
Barabási), which is also the form of attention scores (hᵢ W_Q W_Kᵀ hⱼ). Here it does exact path
integration: 2 000 steps with 5e-14 rad phase error; a random matrix of the same norm fails
(`tests/test_baby.mjs`).

## Results (PREREG.md, committed before any code ran; 10 seeds each)

These show each part doing the job attributed to its region, *in a model built to have those parts*.
They do not show that the brain works this way.

| gate | part | measured | pass if | |
|---|---|---|---|---|
| G1 latent learning | hippocampus | steps / shortest: map 1.14 · model-free Q 8.33 | map ≤ 1.5, Q ≥ 3 | **PASS** |
| G2 need switch | hypothalamus + cortex gate | right resource first: gate on 69.0% · off 49.2% | on ≥ 90%, off ≤ 70% | FAIL |
| G3 theta separation | medial septum | map accuracy after a door change: on 0.906 · off 0.889, wins 0/10 | on ≥ off + 0.05 in 8/10 | **FAIL, kill condition** |
| G3b behaviour after the change | medial septum | steps per trial nearly identical (reported only) | | |
| G4 shortcut | entorhinal grid | steps / new shortest: grid 1.00 · map alone 18.75 | grid ≤ 1.3, map ≥ 1.3 | **PASS** |
| G4b grid cost, no shortcut | entorhinal grid | grid / map steps 1.00 | ≤ 1.3 | pass, **vacuous** |

### Reading it

- **The map does latent learning.** It explores with no reward, tastes food once, then walks to it
  at 1.14× the shortest path. Model-free learning on the identical experience wanders (8.3×).
- **The map's knowledge fades with distance.** The need gate itself works: the creature picks the
  needed resource 92.5% of the time when it is within 12 steps. Beyond that, it picks it only 19% of the
  time, because the map's value gradient is lost in noise there. G2 failed on distance, not on the gate
  (post-hoc breakdown in `results/posthoc_log.txt`).
- **The grid does what the map can't.** Door C opens in a wall the creature only ever bumped into.
  The grid creature walks straight through at the shortest path on its first try. The map alone has
  no experience of that passage.
- **G4b passed for the wrong reason.** In 7 of 10 seeds both creatures hit the 150-step cap from the far
  corner, so the ratio is 1.00 because both failed: the same distance limit as G2.
- **The septum, as modelled, is decoration.** Writing the retrieved read-out back into the map
  lowered its accuracy by under 0.02, and behaviour after a door change was almost the same. The
  write-back term is α·(e_s − (1 − γ)·p), a weak pull toward self-occupancy, too weak to matter.
  A stronger interference model might differ. Choosing one after seeing this result would be
  fitting the test, so this row stays a failure.

### After the tests (post-hoc, not pre-registered): what limits the map

`posthoc_horizon.mjs`, same protocol as G2:

| map setting | right resource first | near (≤ 12 steps) | far |
|---|---|---|---|
| γ 0.9, 3 000 exploration steps (as registered) | 69.0% | 92.5% | 19.0% |
| γ 0.99, 3 000 steps | 69.0% | 91.4% | 21.4% |
| γ 0.99, 6 000 steps | 89.7% | 95.1% | 76.5% |

A longer horizon alone does nothing. The long-range map has to *propagate* through experience
before it exists. In the brain, that is the job of hippocampal replay: offline sweeps during rest that
propagate the map without walking. It is the natural next part to add, as its own pre-registered gate.

## Corrections during the run

- The first gate run used a grid-navigation rule that also stopped at *remembered* walls. PREREG.md
  says "until a move fails". Door C had been a wall during exploration, so the creature trusted stale
  memory and never tried the opening (G4 1.94, FAIL). Fixed to the registered rule and rerun: G4 1.00,
  PASS. Every other gate was identical in both runs. Both runs are in `results/gate_log.txt`.
- The transition memory T (where a move led) updates during test trials too. PREREG.md did not say;
  it is perception, not plasticity of the map, and the map M does not learn during test trials.

## Prior art

Successor representation: Dayan 1993; Stachenfeld, Botvinick & Gershman 2017. Latent learning and
revaluation with the SR: Momennejad et al. 2017. Grid cells as continuous attractors and vector
navigation: Burak & Fiete 2009; Bush, Barry, Manson & Burgess 2015. Theta separation of encoding and
retrieval: Hasselmo, Bodelón & Wyble 2002. Dopamine prediction error: Schultz, Dayan & Montague
1997. Genetic model of the connectome: Kovács, Barabási & Barabási 2020. Transformers and the
hippocampal formation: Whittington, Warren & Behrens 2022. Each part is known. What is this repo's: all of
them as one running creature, each with a pre-registered behaviour test and a switch to turn it off.
