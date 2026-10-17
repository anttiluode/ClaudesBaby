# PREREG — ClaudesBaby

Committed before any code ran. Thresholds and predictions are not moved afterwards.

## What this is

A small creature in a two-room world, built as the nested operators from the "how does the brain
translate to matrices" conversation (17 Oct 2026). Each brain region is one switchable part. Each
part gets one behaviour test it should pass, and an ablation that should break it.

These tests show each part doing the job attributed to the region, *in a model built to have those
parts*. They do not show the brain works this way.

| row | part | in the baby |
|---|---|---|
| lifetime (DNA) | entorhinal grid rings | ring attractors whose weights are never stored entry by entry: W(v) = X R(ωv) Xᵀ, with X = each neuron's two "gene" features (cos θᵢ, sin θᵢ) and R a 2x2 rule matrix rotated by running velocity |
| minutes-days | hippocampus | successor map M (place x place), learned by TD from experience |
| ~100 ms | medial septum | theta cycle: each step has an encoding phase (write the experienced transition) and a retrieval phase (read the map). Ablation: retrieval-phase activity is also written |
| minutes | striatum | reward map per need, rₖ, learned by a dopamine prediction error δ at the visited place |
| seconds-hours | hypothalamus | drives (hunger, thirst) give mixing weights mₖ |
| ms | cortex (L5 gate) | value V(s) = Σₖ mₖ · (M rₖ)(s): basal = map value per need, apical = drive gate, multiplied |
| days-months | lipids / myelin | **not modelled**: they would set time constants and delays, and the lipid atlas has no functional data to constrain them |

## World

13 x 13 grid, outer walls, a vertical wall at x = 6 splitting two rooms. Doors in that wall:
A (6,2) open, B (6,6) closed, C (6,10) closed. Water at (2,2), food at (10,10). Four moves; moving
into a wall leaves the creature in place.

## Fixed settings

- α = 0.1, γ = 0.9 (successor map); striatum learning rate η = 0.5; Q-learning α = 0.1, γ = 0.9.
- Exploration: 3 000 steps of uniform random walk from a random free cell, no rewards (drives off).
- One-step transition memory T[s, a] (last observed next state) for action selection:
  the value of an action is V(T[s, a]), or V of the geometric neighbour if (s, a) is untried.
- Test trials: greedy on action value, ties broken at random, cap 150 steps, no learning unless
  stated. 20 trials per seed from random free cells at least 4 steps from the goal.
- 10 seeds (0-9) per gate.
- Grid modules: periods 3, 5, 7 cells, two rings (x, y) each, 16 neurons per ring, update
  r <- relu(W(v) r), normalised. Position decoded from ring phases by search over
  displacements -12..12 (unique, 3·5·7 = 105).
- Septum ablation: in the retrieval phase the map's own prediction p = M(s,:) is written as if it
  were experience: M(s,:) += α·[e_s + γ·p − M(s,:)], in addition to the normal encoding write.

## Gates

| gate | test | pass if | my prediction |
|---|---|---|---|
| G1 latent learning (hippocampus) | explore without reward, taste food once, then seek it | median steps / shortest path: successor map ≤ 1.5 and model-free Q-learning (same experience, same taste) ≥ 3 | SR ~1.1, Q mostly at the cap |
| G2 need switch (hypothalamus + cortex gate) | both resources tasted; set hunger or thirst; which is reached first | gate on: ≥ 90% correct; gate off (equal weights): ≤ 70% | ~100% vs ~55% |
| G3 theta separation (septum) | after exploration, door A closes and B opens; 600 more exploration steps; correlation of learned M with the true successor map of the new maze (uniform random walk, (I − γP)⁻¹ rows) | septum on beats septum off by ≥ 0.05 in ≥ 8 of 10 seeds | on ~0.9, off ~0.7 |
| G3b behaviour after the change | 10 food trials with learning on after the change | reported only: mean steps, on vs off | on recovers faster |
| G4 shortcut (entorhinal grid) | after exploration with tasting, door C opens (never experienced); first food trial from (2,10) | grid + map ≤ 1.3 x new shortest path, map alone ≥ 1.3 x | grid ~1.0, map ~2.5 |
| G4b cost of the grid when there is no shortcut | same, door C closed | grid + map ≤ 1.3 x the map-alone steps | ~1.1 |

Grid navigation rule: head along the decoded displacement to the remembered goal (larger axis first)
until a move fails, then follow the map for the rest of the trial.

Unit test (not a gate): the genome-generated rings path-integrate exactly in a 2 000-step random
walk; a random W with the same norm does not.

## Kill conditions

- G1 fails on the map side: the successor map is not doing latent learning here, and the
  hippocampus row is decoration.
- G3 fails: the theta separation, as modelled, does nothing measurable; the septum row is decoration.
