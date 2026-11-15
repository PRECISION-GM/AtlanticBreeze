# Native sensitivity and cost profiling, 4 March 2016 ENA case (reduced grids)

Status: native paired central finite differences (FD) only. No automatic-differentiation (AD) result, no
Jacobian, no descent. All numbers below come from the job logs and `observation_sensitivity.jls` files
listed under Provenance.

## Configuration (common to every run)
- Production interval `~/AtlanticBreeze-data/interval-v2`, forcing.csv SHA256
  ec1f2c1962fc25138f4fde0028a325404df3fec20536b65dc90ba17bf7770405 (checked by the script).
- Breeze AtmosphereModel, anelastic, 155 levels, WENO9 momentum and scalars, closure=nothing, P3 with
  prognostic aerosol (aer1), live ecCKD radiation every 30 s with the continued column to 65 km,
  time-dependent apparent solar position, Float64, dt = 1 s for every grid.
- Window 13:30–15:00 UTC (t = 7200–12600 s after launch, 5400 steps). Each grid spins up natively from the
  sounding to 7200 s with production forcing; that window state is then held fixed for all perturbations,
  so the derivatives exclude any dependence through the spinup.
- Loss: J = mean_L[((MA30 LWP_model − MA30 LWP_paper)/100 g m⁻²)²] + mean_D[((MA30 DWP_model − MA30 DWP_paper)/1 g m⁻²)²].
  The scales are chosen balancing weights, not measurement errors. MA30 is the centred 30-min
  piecewise-linear mean (≥95% coverage, no bridging gaps > 2 min). Time masks (61 centres each) are fixed
  from the baseline.
- LWP model: domain-mean cloud + rain + ice-liquid path. DWP model: a rain-path proxy below the first
  qᶜˡ > 1e-5 cell along the XZ transect j = ny/2, averaged over cloudy columns (no radar/lidar operator).
  "Frozen" holds the baseline cloud-base index and cloudy columns fixed, giving a conditional, frozen-mask
  derivative. "Moving" recomputes them and is nondifferentiable where they switch.
- Directions, per unit ε: heating +1 K/day and moistening +1 g/kg/day, both tapered linearly to 0 between
  1500 and 2500 m and applied at all hourly knots; SST +1 K at all knots, or at the 9000 s (14:00 UTC)
  knot only. These are directional derivatives, not time-series gradients.
- Paths the perturbations reach: dT/dt and dq/dt act through the model forcing. SST feeds a single surface
  temperature field used by both the surface fluxes and the radiation; it is refreshed every step from the
  interpolated, perturbed knots.

## Grids
| grid | domain | cells | role | baseline J0 |
|---|---|---|---|---|
| 16×16, dx 200 m | 3.2 km | 39,680 | reference | 2.3209 |
| 32×32, dx 100 m | 3.2 km | 158,720 | resolution (same domain) | 0.8010 |
| 32×32, dx 200 m | 6.4 km | 158,720 | domain size (same dx) | 4.7883 |

None of these represents the 12.8 km production domain.

## Results: dJ/dε, frozen masks [moving masks]
| grid | direction | 1 | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 |
|---|---|---|---|---|---|---|---|
| n16 | heating | −.3356 [−.239] | −.3098 [−.182] | −.2869 [−.135] | −.2842 [−.205] | −.2871 [−.245] | −.2596 [−.2596] |
| n16 | moistening | .7949 [.568] | .8054 [.585] | .7670 [.447] | .7085 [.400] | .6950 [.650] | .6208 [.186] |
| n16 | sst, all knots | −.6329 [−.716] | .0550 [−.024] | .6158 [.856] | .7904 [.922] | .7787 [1.299] | .1584 [1.941] |
| n16 | sst, 9000 s knot | −.4820 [−.552] | .3051 [.214] | .3304 [.411] | .7476 [.802] | .4794 [1.096] | −.0273 [.974] |
| n32 dx100 | heating | −.0777 [−.027] | −.0460 [−.006] | −.0368 [.007] | | | |
| n32 dx100 | moistening | .1859 [.079] | .1771 [.101] | .1267 [.043] | | | |
| n32 dx100 | sst | −.9518 [−.854] | −.8079 [−.732] | −.1628 [−.398] | | | |
| n32 dx200 | heating | −.4679 [−.248] | −.4767 [−.307] | −.5059 [−.367] | | | |
| n32 dx200 | moistening | 1.2889 [.744] | 1.2254 [.737] | 1.2630 [.927] | | | |
| n32 dx200 | sst | −1.3904 [−1.155] | −2.3237 [−1.503] | −1.8600 [−1.154] | | | |

- Reproducibility: in every job, re-running the baseline gives ΔJ = 0 and max ΔLWP = 0 (bitwise).
- The LWP-term slopes are small and stable (|d/dε| ≤ 0.015). The DWP term carries all the drift.
- Mask audit at ε = 0.003 (1456 transect column-times):
  - heating: 0 cloud-base switches and 0 cloudy/clear switches. J(+ε) = J0 − 1.4636e-3 (below J0) and
    J(−ε) = J0 + 9.383e-5 (above J0). The response is strongly asymmetric, with second difference −152.
    (Correction: an earlier version said both were below J0. The audit field `down` is J0 − J(−ε), and it is
    negative here.) The central slope −0.2596 is unaffected;
  - moistening: 3 base switches;
  - SST: 12–73 base switches.
  Mask switching therefore cannot explain the heating asymmetry. The trajectory diagnosis (job 16578)
  tests reproducibility at nonzero controls and the divergence floor at ε = 1e-6 and 1e-9.
- Conclusion so far: no direction has a validated plateau. The n16 heating value of ≈ −0.285 over
  ε = 0.1–0.01 did not persist at 0.003 (−0.260). SST changes sign with ε. Descent is not attempted.

## Cost (native; all runs on NVIDIA A100-SXM4-80GB, driver 550.90.12)
Fresh-process probe (`timing_probe.jl`, jobs 16570–16572). Each category is measured with the same
protocol on every grid.

| grid | package load | first model build* | first step† | rest of first radiation block | warm s/step (300 steps) | warm rebuild |
|---|---|---|---|---|---|---|
| n16 dx200 | 83.1 s | 477.1 s | 19.2 s | 1.37 s | 0.0470 | 3.13 s |
| n32 dx100 | 83.1 s | 478.1 s | 19.3 s | 1.38 s | 0.0466 | 3.15 s |
| n32 dx200 | 83.1 s | 475.3 s | 18.6 s | 1.28 s | 0.0448 | 3.23 s |

\* Includes model construction, native JIT of construction and initialization, and restoring the state; it
is not pure compile time.
† Includes the JIT of the step kernels and the radiation call on that iteration; it is not an isolated
radiation cost.

Window integrations inside the sensitivity jobs (synchronized, 5400 steps):
- warm: 219.1 s (n16), 225.7 s (n32 dx100) and 240.0 s (n32 dx200), i.e. ≈ 0.040–0.044 s/step;
- FD pair wall: 405–475 s;
- spinup 0–7200 s, including that process's JIT: 499 s, 529 s and 550 s.

Per-step cost is nearly independent of cell count from 40k to 159k cells, so at these sizes the step is
launch- or latency-bound rather than throughput-bound. Peak host RSS is ≈ 3.2 GB; peak GPU memory is
1.8 GB (n16) and 2.3 GB (n32).

In the sensitivity jobs, the "cold" labels on the n32 baselines are invalid: the same-process spinup had
already warmed the JIT.

AD and Reactant tracing, primal compile, reverse compile and reverse execution for this window are
unavailable (not run) and are not zero. The only Reactant data so far are kernel-level: for the P3 kernel at
n16, building the Reactant model took 1967 s, compiling the kernel 46–56 s and compiling its reverse
87–104 s.

## Provenance
- Code: branch `review`.
  - be738ca: script and forcing_perturbation hook.
  - 7f15f21: timing and CFL.
  - 1bd17da: mask/Taylor audit and timing_probe.
  - 46fc106: trajectory mode.
  - Runs use frozen detached worktrees `~/AtlanticBreeze-runs/<commit>`.
- Jobs and outputs (`~/AtlanticBreeze-outputs/<commit>/...`):
  - 16535: n16 thermo, ε 1–0.1. The n16 window state is `be738ca/obs-sens-thermo-n16/attempt-1/window_state.jls`.
  - 16545: n16 SST. 16553: n16 thermo, ε 0.03 and 0.01.
  - 16546 (n32 dx100) and 16547 (n32 dx200): each with its own window state.
  - 16568 / 16569: ε 0.003 thermo / SST 0.01 and 0.003.
  - 16570–16572: timing. 16578: trajectory diagnosis.
- Targets: `~/AtlanticBreeze-data/paper-gradient-targets` (digitized traces of Ghate & Cadeddu 2019, Fig. 2;
  not original retrievals).

## Trajectory diagnosis (job 16578, 46fc106, n16, same window state)
- Zero perturbation (heating 0): ΔJ = 0 and zero field divergence at every minute.
- Two independent heating +0.003 runs are identical in J and in every divergence diagnostic, so runs with
  a nonzero control are reproducible.
- Response ΔJ_frozen for each direction (the DWP term dominates in every case):

  | direction | ε | ΔJ_frozen |
  |---|---|---|
  | heating | +0.003 | −1.464e-3 |
  | heating | −0.003 | +9.38e-5 |
  | heating | 1e-6 | +1.90e-4 |
  | heating | 1e-9 | +2.62e-6 |
  | moistening | 1e-9 | +1.00e-5 |
  | SST | 1e-9 | −5.20e-5 |

- ΔJ/ε grows as ε shrinks (heating): 0.49 per unit at 0.003 (central slope 0.26 in magnitude), 190 at 1e-6, 2.6e3 at 1e-9. The response is not
  linear in ε over this range.
- Per-level RMS Δρqʳ (maximum over levels) after the first minute:

  | ε | 9 min | 29 min | 59 min | 90 min |
  |---|---|---|---|---|
  | 0.003 | 8.2e-11 | 7.7e-10 | 2.7e-8 | 7.1e-8 |
  | 1e-6 | 7.6e-13 | 6.3e-10 | 2.2e-8 | 2.8e-8 |
  | 1e-9 | 7.9e-13 | 7.7e-10 | 1.3e-8 | 1.9e-8 |

  By about 30 min the divergence no longer depends on ε, across 6.5 decades.
- Divergence of all four tracked fields (ρqᶜˡ, ρqʳ, w, θ) first appears at the second record (1 min).
- Interpretation: perturbations are amplified quickly and reproducibly up to a common, ε-independent
  amplitude well before the end of the 90-min window. That makes the central FD of J over this window
  non-convergent in ε. This shows strong amplification of tiny perturbations. It does not prove that no
  mathematical derivative exists, nor that a noise floor is irreducible, and it does not identify the
  mechanism (P3 cancellation, dynamical growth, or both). Next: per-step early trajectory, to find where
  linear scaling Δ ∝ ε first breaks and at which level.

## Early-trajectory diagnosis (job 16580, a0f22f2; analysis `scripts/early_divergence.jl`)
The first 600 s of the window, recorded every 1 s step, for heating ε = 1e-3, 1e-6, 1e-9 and 1e-12, all from
the same IC. The quantity is the maximum over levels of the per-level RMS difference from the baseline.
- θ, w and ρqᶜˡ respond linearly: Δ(1e-3)/Δ(1e-6) = 1000 throughout. For θ, ε = 1e-9 and 1e-12 are
  below its per-step Float64 resolution.
- ρqʳ does not respond linearly. From step 2, Δρqʳ ≈ 1.3–5e-16 for ε = 1e-6, 1e-9 and 1e-12 alike, at
  subcloud levels 8–22. That is a perturbation-independent jump of about 1e-6 relative to the local rain,
  far above ρqʳ round-off.
- These are the levels where catastrophic cancellation in the P3 rain share was found at kernel level.
  The match is consistent with that cancellation as the source, but does not prove it.
- The DWP loss is built from rain, so this floor enters J directly.
- Test: an isolated, algebraically identical stable form (Breeze worktree `stable-rain-share-test`,
  a6228574). It is validated against BigFloat first (job 16583), then used for one matched early-trajectory
  comparison. Production is unchanged.
