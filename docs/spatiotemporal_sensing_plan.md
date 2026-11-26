# Spatiotemporal adaptive sensing (STS): reference analysis and implementation plan

Status: **plan, nothing implemented** (2026-11-25). Written while the six TMP02 members train on GPUs 0-5. Nothing
in this plan modifies a file that TMP02 imports (see section 6.1).

Sources read for this plan:

* Reference: `~/adaptive_sensor_policy`, branch `spatiotemporal` (checked out, commit `d58be3fadb`):
  `envs/turtlebot_var_scan_rate.py`, `AdaptiveFPS/scripts/train_adaptive_fps_ppo.py`, `docs/spatiotemporal_ppo.md`,
  `AdaptiveFPS/config/spatiotemporal/ST0*.yaml`, `AdaptiveFPS/evaluation/*`.
* Current: `AdaptiveSensing/evaluation/scan_gate.py`, `AdaptiveSensing/training/{temporal_sensing,sensing_agent,
  recurrent_ppo}.py`, `AdaptiveSensing/scripts/train_temporal_ppo.py`, the TMP02 config and card,
  `source/st_nav/st_nav/tasks/nav2d/corridor_env.py`, `source/st_nav/st_nav/sensors/{lidar,ray_tables}.py`.

---

## 1. The reference implementation

Your understanding is correct, with one detail: the reference does not add the spatial dimension on top of the temporal
trainer. On the `spatiotemporal` branch the temporal action was **replaced**. The env takes one rate index per sector
(`MultiDiscrete`), and uniform whole-scan sensing is the special case "same rate in every sector". The temporal path,
the whole-scan action and temporal checkpoints were removed from that branch on 2026-11-14.

### 1.1 Where everything lives

| Concept | File | Symbol (line) |
|---|---|---|
| Sector definition | `envs/turtlebot_var_scan_rate.py` | `SECTORS`, `SECTOR_BOUNDS_DEG`, `BEAM_SECTOR`, `SECTOR_BEAMS` (78-84) |
| Rate set, obs layout | same | `SECTOR_FPS_CHOICES = (0.2, 0.5, 1.0, 2.5, 5.0)`, `OBS_DIM = 49`, `OBS_LAYOUT` (85-87) |
| Action space | same | `spaces.MultiDiscrete([len(fps_choices)] * 4)` (398) |
| Per-sector gate state, reset | same | `_init_native_scans` (549-587) |
| Acquisition, deadlines, rate changes | same | `_advance_with_native_scans` (672-726) |
| Step order, reward, FSE, info masks | same | `step` (883-999) |
| Augmented observation | same | `_get_augmented_obs` (481-486) |
| Policy network | `AdaptiveFPS/scripts/train_adaptive_fps_ppo.py` | `Agent` (289-368) |
| Masked PPO terms | same | `masked_policy_terms` (371-402) |
| Rollout storage of masks | same | 634-684 (`masks[step] = infos["sector_action_mask"]`) |
| Vector-env info plumbing | same | `adapt_infos_gym029`, `ARRAY_INFO_KEYS` (229-280) |
| Design document | `docs/spatiotemporal_ppo.md` | sections 1-10 (env and trainer), 12-14 (evaluator) |
| Tests | `tests/test_spatiotemporal_sensing.py`, `tests/test_adaptive_spatial_ppo.py` | Tests A-G and P1-P4 |

### 1.2 Spatial partitioning

The planar 40-beam scan is split into four 90° sectors, front / left / rear / right, as half-open intervals in the robot
frame: front [-45, 45), left [45, 135), rear [135, 225), right [225, 315). Beam angles are wrapped to [-45, 315) and
assigned with `np.searchsorted`, which gives 10 beams per sector. Partitioning is static and works on beam indices:
`BEAM_SECTOR[b]` is the sector of beam b, and `due[BEAM_SECTOR]` expands a per-sector mask to a per-beam mask.

### 1.3 Action space and policy distribution

* Action: `MultiDiscrete([5, 5, 5, 5])`, one index per sector into `fps_choices` (configs ST02/ST04-07 use all five
  rates; ST01 used `(1.0, 2.5)`).
* Policy: **four independent categoricals on a shared trunk** (a factorised policy, not a joint categorical):

  ```
  π(a_t | h_t) = Π_s π_s(a_{t,s} | h_t)
  logits = actor(h_t).view(-1, n_sectors, n_choices);  Categorical(logits=logits)   # batch shape (B, 4)
  ```

  log-probs and entropies are returned **per sector**, shape (B, 4). Masking is applied in the loss only.
* There is no joint categorical over the 5^4 = 625 combinations anywhere.

### 1.4 Per-sector timing (native LiDAR, `_advance_with_native_scans`)

Each sector j has its own state: `sector_hz`, `sector_period`, `sector_anchor`, `sector_n_next`, `sector_next_t`,
`sector_last_t`, `sector_last_acq_k` and `sector_n_acq`. Within the control step (t_start, t_end], every native scan at
time `now` is processed in order:

1. `due = ~(now < sector_next_t - tol)`. If no sector is due, the scan is skipped and nothing is raycast.
2. Otherwise the robot is integrated to `now`, one raycast is done, and **only the due sectors' beams** are copied into
   the held buffer: `last_sampled_scan[due[BEAM_SECTOR]] = scan[...]`.
3. The lattice advances at the old rate without float drift:
   `n_next = floor((now + tol − anchor)/period) + 1`, `next_t = anchor + n_next·period`.
4. **Decision application.** At a sector's *first* acquisition in the step (`due & ~tick_decided`), this step's
   requested rate is applied. If the rate is the same, the schedule is kept exactly. If it is different, the lattice
   re-anchors at the acquisition (`anchor = now`, `next = now + 1/hz`).
5. `tick_decided |= due` gives the per-sector causal mask. `sector_n_acq += due`, and `tick_beams` counts every
   acquired beam (a sector acquiring twice in one step counts twice).

* **Reset:** every sector starts at `max(fps_choices)`, anchored at the reset scan `native_t0`, with
  `sector_last_acq_k = 0` and `episode_frame_count = 1`.
* **Staleness:** `steps_since_last_obs[s] = k − sector_last_acq_k[s]` in control ticks, and the age in seconds is
  `k·dt − sector_last_t`.
* **Rate changes:** a rate request is applied only when the sector actually acquires. Ignored requests leave no trace
  (Test B checks this bit for bit).

### 1.5 Augmented observation (49 = 40 + 4 + 4 + 1)

```
fps_ratio[s]     = sector_hz[s] / max(fps_choices)
obs_age_ratio[s] = clip((k − sector_last_acq_k[s]) / max_obs_interval, 0, 1),   max_obs_interval = f_ctrl / min(fps)
frame_ratio      = clip(FSE / budget, 0, 1)                                      ONE global value, not per sector
FSE              = (40 + Σ_s n_acq[s]·10) / 40                                    full-scan equivalents, reset scan = 1
```

The masks are deliberately **not** in the observation. With the same action in every sector, all three features are
bit-identical to the temporal env's `_get_augmented_obs` (Test G).

### 1.6 Causal masking with partial acquisitions

`info["sector_action_mask"][s] = 1` if and only if sector s acquired during the step, which means component s of
`a_t` was applied. Under native timing the mask cannot be predicted, so it is stored at the action's own rollout
index t, taken from the same `env.step`. Per-sector log-probs are stored as (T, N, 4) for this reason.

### 1.7 Actor-critic, storage and loss (`masked_policy_terms`)

* Network (no CNN; 2D LiDAR): `Linear(49→64) Tanh → LSTM(64)`, shared, then actor `64-64-(4·n_choices)` (std 0.01)
  and critic `64-64-1`. The LSTM steps every tick and is reset on `done`.
* Storage: `actions`, `logprobs` and `masks` have shape (T, N, 4). Everything else is unchanged.
* Loss (doc section 7.3):

  ```
  ρ_t  = Σ_s m_{t,s} (log π_s^new − log π_s^old)        joint log-ratio over APPLIED sectors only
  r_t  = exp(ρ_t);   u_t = 1[Σ_s m_{t,s} ≥ 1]              one clipped ratio per tick
  Â_t  = (A_t − mean_{u=1} A) / (std_{u=1} A + 1e-8)       statistics over applied ticks of the minibatch
  L_pg = Σ_t u_t max(−Â_t r_t, −Â_t clip(r_t, 1±ε)) / max(Σ u, 1)
  H̄    = Σ_{t,s} m_{t,s} H_s / max(Σ m, 1)                 mean entropy per APPLIED sector decision
  L_v  = clipped value loss over EVERY tick; GAE over every tick (unchanged)
  L    = L_pg − ent_coef·H̄ + vf_coef·L_v
  ```

  Justification (doc 7.2): if `m_{t,s} = 0` the env discarded `a_{t,s}`, so
  `E[∇log π_s · A_t | h_t] = 0`. Dropping the term is unbiased and removes pure noise. Unapplied components contribute
  exactly 0 to ρ_t and to the gradient. P1 checks the zero gradient, and P2 checks that the actor rows of a
  never-applied sector stay untouched.
* KL and clipfrac are computed over applied ticks. A minibatch with no applied decision gives exactly zero for every
  policy term (denominators clamped to 1). Hyperparameters are the temporal ones: lr 3e-4, γ 0.99, λ 0.98,
  4 epochs × 4 minibatches of whole envs, clip 0.2, `ent_coef` 0.05, vf 0.5, grad clip 0.5.

### 1.8 Reward and sensing cost

```
r_t = nav_reward − frame_cost · beams_acquired_t / 40 + terminal(success / collision / timeout) − budget_penalty·[first t: FSE > budget]
```

One full scan costs `frame_cost`, and a 10-beam sector costs `frame_cost/4`. The budget (450) counts FSE and never
binds. ST01-ST05 swept `frame_cost` 0-0.05. **ST06/ST07 ("4xfc") swept 0.08-0.2** because, in the commit's own words,
models were "trained using 1/4 penalty for frame": a single sector's saving was too small to move the low-value sectors.

### 1.9 Episodes and metrics

* Init: every sector at the maximum rate, with the reset scan in every sector. The reset scan counts as FSE 1 and is
  not charged. Termination is unchanged.
* Training logs: `charts/fse`, `charts/mean_chosen_fps_<sector>` (rate of applied decisions only),
  `losses/entropy_<sector>`, `charts/applied_tick_fraction`, `charts/applied_decisions_per_tick`.
* Evaluation: per-sector `<s>_hz`, `<s>_n_decisions` and `<s>_n_switches`, `mean_fse` (successful episodes), plus a
  fixed constant-configuration factorial (`--spatial-all`, ST00: 63 configurations) as baselines.
* Reported result (ST01, 10 episodes, smoke test): front 1.79, left 1.49, rear 1.26 and right 1.63 Hz. Front was the
  fastest and **rear the slowest**.

---

## 2. Proposed extension for Isaac Lab / Mirrorcle 3D LiDAR

### 2.1 Partition: verified 46 / 47 / 46 columns

The range image is a uniform cell-centre grid over −14..+14° azimuth (`grid_pattern`, `cell_center`), with
28/139 = 0.2014° per column and 34 rays per column. Column 0 is the leftmost (+14°). Every column has the same number
of rays, so equal ray counts mean equal column counts. Computed with the real ray table and the R4 mount (pitch 6.75°):

| Section | Columns | Rays | Weight w_s = R_s / 4726 | Sensor azimuth | Base-frame azimuth (after pitch) |
|---|---|---|---|---|---|
| L (left) | 0-45 (46) | 1,564 | 0.33094 | +14.00 … +4.73° | +14.19 … +4.80° (9.39°) |
| C (centre) | 46-92 (47) | 1,598 | 0.33813 | +4.73 … −4.73° | +4.73 … −4.73° (9.46°) |
| R (right) | 93-138 (46) | 1,564 | 0.33094 | −4.73 … −14.00° | −4.80 … −14.19° (9.39°) |

* 46/47/46 is the only split that is symmetric about the forward axis. The other splits, 47/45/47 and 46/46/47,
  differ by up to 68 rays per section.
* The mount pitch spreads the outer columns slightly in the base frame: the edge columns span 13.80-14.19° over the
  34 rows. This makes the 46-column outer sections almost exactly as wide in angle as the 47-column centre (9.39° vs
  9.46°). Section boundaries are slightly curved in the base frame (±0.07°), but the partition is by column index, so
  it is exact, disjoint and complete.
* Floor returns are balanced: 1,315, 1,363 and 1,315 rays per section point at the floor within 12 m.
* The section layout must be derived from `grid_col` of the env's actual ray table, not hard-coded. The code then
  stays correct if a measured Mirrorcle ray table (`lidar3d.ray_table`) replaces the grid.

### 2.2 Timing: the TMP02 mechanism, once per section

TMP02's timing differs from the reference: scans sit on control instants, and the action selected at t_k gates the
native scan **of instant t_k**. This is kept, and applied independently to each section s ∈ {L, C, R}:

```
at control instant k (physics tick τ_k):
  o_k = (held image H_k, φ_k, α_k, β_k)      built BEFORE gating the scan of τ_k
  a_k = (a_{k,L}, a_{k,C}, a_{k,R}) ~ π(·|h_k)
  new_k = (native scan time changed)                          shared by all sections
  for every section s:
      c_s += 1
      m_{k,s} = new_k ∧ (τ_k ≥ D_s)                           section s accepts the native scan
      if m_{k,s}:  H[:, :, cols(s)] ← I_k[:, :, cols(s)]      ONLY its columns
                   F_s = τ_k;  D_s += (⌊(τ_k − D_s)/P_s⌋ + 1)·P_s   (deadline advanced by whole periods)
                   P' = P(fps[a_{k,s}]);  if P' ≠ P_s: P_s = P', D_s = τ_k + P'   (set_target_fps re-anchor)
                   c_s = 0;  n_s += 1
      else:        a_{k,s} discarded
  NAV05({nav features_k, H_k})              complete 34×139 image, sections of different ages
  env.step                                  20 physics steps, native 4 Hz scheduler unchanged
```

* Every section keeps its own period P_s, deadline D_s, last acquisition F_s, age counter c_s and acquisition counter
  n_s. Rates and deadlines are integer physics ticks: 25, 50, 100, 200 and 500 ticks for 4, 2, 1, 0.5 and 0.2 Hz, so
  all comparisons are exact.
* Reset: every section is at 4 Hz with the reset scan accepted in every section, `n_s = 1`, `c_s = 0`, not charged.
  Terminal tick and reset attribution are as in TMP02.
* Native 4 Hz LiDAR and 5 Hz control are unchanged. One tick in five has no new native scan, so all three masks are 0
  there, exactly as in TMP02.
* **No technical limitation requires a different mechanism.** All sections see the same native scans, so the only
  change is that acceptance is a per-section predicate.

### 2.3 Augmented observation (nine variables)

Variable-major order, as the reference `OBS_LAYOUT`, so that S = 1 reproduces the TMP02 layout `[φ, α, β]` exactly:

```
x_k = [ φ_L, φ_C, φ_R,  α_L, α_C, α_R,  β_L, β_C, β_R ] ∈ [0, 1]^9
φ_s = f_s / f_max                         f_max = 4 Hz
α_s = clip(c_s / K, 0, 1)                 K = f_ctrl / f_min = 5 / 0.2 = 25 ticks
β_s = clip(n_s / B, 0, 1)                 B = 500 (TMP02 budget), n_s includes the reset scan
```

The reference has **one global** `frame_ratio` (FSE / budget). You asked for three per-section consumption ratios.
This loses nothing, because the global value is linear in them: FSE/B = Σ_s w_s β_s (below the clip, which never
binds at B = 500). With uniform actions, each β_s equals TMP02's frame_ratio. Reset observation: φ = 1, α = 0,
β = 1/500 for every section.

---

## 3. Network and PPO

### 3.1 Recommended architecture: TMP02's agent, minimal changes

```
held image (1, 34, 139) -> NAV05-architecture CNN -> 3072 -> Linear 128 ReLU      unchanged
   ++ x_k (9)                                                     (131 -> 137 inputs)
   -> Linear 64 Tanh -> LSTM 64                                   unchanged, shared by actor and critic
   -> actor  64 Tanh 64 Tanh -> Linear 3 × 5 = 15 logits (std 0.01), viewed as (B, 3, 5)
   -> critic 64 Tanh 64 Tanh -> 1                                 unchanged
```

* This adds +384 weights in the first trunk layer and +650 in the actor output, about 0.2 % of TMP02's 475,398
  parameters. With S = 1 the agent is exactly TMP02's `SensingAgent`. The new class should **subclass**
  `SensingAgent` and override only the actor output and `get_action_and_value`. It must not edit `SensingAgent` itself.
* Initial policy: uniform in each section. The entropy is ln 5 = 1.609 per section, the same scale as TMP02's
  entropy, so `ent_coef` = 0.05 means the same thing **per decision**.
* Spatial awareness: the CNN has no padding and stride 2, and its flatten → Linear keeps spatial positions, so
  section-specific features can be learned. The section boundaries at columns 46 and 93 fall near feature columns ~5
  and ~11 of the 16. The ages enter only through x_k.
* Optional ablation (not the default): add a second image channel with each column's normalised age α_s, so the CNN
  sees where the stale columns are. Defer it unless the default agent fails to separate sections.

### 3.2 Independent categoricals, not a joint categorical

The reference uses independent categoricals, and so should we:

* **Masking needs factorisation.** With π = Π π_s, an unapplied component simply drops out of the log-ratio. A joint
  categorical over 5³ = 125 combinations would need, at every tick, the marginal log π(a_applied) = logsumexp over
  the unapplied components. That is possible but more complex, and harder to verify.
* Coordination between sections at the same tick (for example "keep at least one side fast") still comes from the
  shared LSTM state. A factorised head cannot express correlations within a single tick, and the reference showed
  that this is not limiting.

### 3.3 Ratios, entropy, advantages, masks (and the "no borrowed gradient" requirement)

Use the reference's `masked_policy_terms` for 3 sections with the TMP02 chunked accumulation:

```
m_{t,s} = 1[section s accepted at tick t]      stored at the action's index t (same sense.step)
ρ_t = Σ_s m_{t,s}(log π_s^new(a_{t,s}|h_t) − log π_s^old(a_{t,s}|h_t));   r_t = e^{ρ_t};   u_t = 1[Σ_s m_{t,s} ≥ 1]
Â_t over {u_t = 1} of the whole minibatch (before chunking, as TMP02)
L_pg = Σ u_t max(−Â_t r_t, −Â_t clip(r_t, 1±ε)) / Σ u_t
H̄    = Σ m_{t,s} H_s(h_t) / Σ m_{t,s};    KL and clipfrac over u_t = 1;    L_v, GAE: every tick
```

**Guarantee:** ∂L/∂z_{t,s} = 0 whenever m_{t,s} = 0, where z_{t,s} are section s's logits at tick t. ρ_t contains no
term for s, and H̄ excludes s. So a section's decision gets no policy or entropy gradient because *another* section
acquired. The shared trunk and LSTM still learn from the applied sections' terms, which is intended.

Two subtleties:

1. **Joint clipping couples applied sections.** If section A's ratio pushes r_t outside [1−ε, 1+ε], section B's
   gradient on that tick is also cut. This *removes* gradient; it never adds borrowed gradient. It is the standard
   trust region for a factorised action (cleanRL `MultiDiscrete`), and with S = 1 it reduces exactly to the TMP02
   loss. Alternative: per-section clipping, Σ_s m_{t,s} max(−Â r_{t,s}, −Â clip(r_{t,s})) / Σ m. I recommend the
   joint form (the reference, and directly comparable with TMP02), with per-section clipping as a config switch
   (`ratio_mode: joint | per_section`) for an ablation.
2. **Shared advantage.** All sections applied at tick t share A_t, because there is one reward stream. Credit between
   sections comes from the LSTM state and from averaging over ticks, as in the reference.

The policy loss weights each applied tick 1/Σu, while the entropy weights each applied decision 1/Σm. Their relative
strength therefore depends on applied decisions per tick (between 1 and 3). Log `rollout/applied_decisions_per_tick`
(reference) to keep `ent_coef` interpretable.

---

## 4. Reward and sensing consumption

### 4.1 Full-scan equivalents with exact ray weights

```
FSE_k   = Σ_s w_s n_{k,s},           w = (1564, 1598, 1564) / 4726 = (0.33094, 0.33813, 0.33094),   FSE_0 = 1
ΔFSE_k  = Σ_s w_s m_{k,s}            consumption of tick k; a complete acquisition (all three) = 1 exactly
r_k     = r_k^NAV05 − λ ΔFSE_k − ρ_b 1[first k with FSE_k > B]
```

* This is the reference's `beams/40` with the actual ray counts. Sum the weights in float64 and store them in float32.
  Σ w_s = 1 holds exactly at float64 precision, so a full acquisition costs exactly λ.
* The NAV05 reward (progress, proximity, time, ±30 terminal, timeout 0) is unchanged.
* The budget (500, budget penalty 15) is counted in FSE. At most 281 FSE are possible per 70 s episode, so it never
  binds, as in TMP02.

### 4.2 Comparability with TMP02

| Quantity | TMP02 | STS | Relation |
|---|---|---|---|
| Consumption per episode | `frames` (incl. reset) | FSE | equal when all sections accept together |
| Charged consumption | `mean_frame_consumption` | FSE − 1 | same convention (reset free, terminal tick charged) |
| Cost per full scan | λ | λ | same λ values are directly comparable |
| Effective rate | applied fps | Σ_s w_s f_s (ray-weighted rate in force) | same unit (Hz-equivalent) |
| Budget | 500 frames | 500 FSE | same |
| Fixed baselines | TMP01 (5 uniform rates) | STS01 factorial (5³ = 125; the 5 diagonal configurations = TMP01) | TMP01 ⊂ STS01 |
| Policy class | one rate | three rates | uniform STS actions **reproduce TMP02 bitwise** (validated, section 6.3), so TMP02 ⊂ STS |

Sensing efficiency is reported as success rate and NAV return against mean FSE: one Pareto plane holds TMP01, TMP02,
STS01 and STS02.

### 4.3 Is a change to the sensing reward needed?

No change to its form. **Frame-cost scale** is the open question:

* One skipped section acquisition saves λ/3. That is a weaker per-decision incentive than in TMP02, but not as weak as
  the reference's λ/4, where ST06/ST07 had to use 4× costs.
* Recommendation: STS02 uses the TMP02 sweep `[0, 0.1, 0.15, 0.2, 0.3, 0.4]` for a cost-matched comparison. Decide
  whether to extend upward (×3: 0.3-1.2) after (a) the STS01 factorial shows which sections are cheap to slow, and
  (b) TMP02's results show where its policies sit. Above ~0.6 the TMP02 calibration found crash-prone rates
  outranking 4 Hz, so extended costs need care.

---

## 5. Simulation feasibility and cost

### 5.1 Partial acquisitions in the simulator

* The simulator does **not** acquire sections independently, and it does not need to. In
  `CorridorNavEnv._get_observations`, `if due.any(): self.corr.lidar.read(...)` makes `MultiMeshRayCaster` cast
  **all 4,726 rays for all envs** whenever any env is due. Noise is keyed per (env, tick, ray).
* So STS, like TMP02, generates the full native scan and **forwards sections selectively**. Section consumption is a
  *simulated sensing cost*, the physical motivation being a MEMS mirror scanning a sub-window. It is not a
  computational saving here.
* Real compute savings would need a custom ray caster that casts per-env ray subsets. Ray casting is 4.5 ms of the
  ~140 ms tick (TMP02 profiling), so the best case is about 3 %. That is not worth it, and it would make the cost model
  harder to validate.
* Limitation: the time a MEMS sub-window scan takes is not modelled. Acquisitions are instantaneous at control
  instants, as in TMP02.

### 5.2 Required modifications

| Component | Change | TMP02 file touched? |
|---|---|---|
| Env (`corridor_env.py`, scene, fast path) | none | no |
| Scan gate | new `SectionScanGate`: (N, S) state, shared `new`, per-section `fwd`, column-masked image update | no (new file; reuses `period_ticks`) |
| Sensing wrapper | new `SpatiotemporalSensing` (same structure as `TemporalSensing`) | no (imports `load_nav_policy`, `nav_run_cfgs`, `nav_contract_mismatches`) |
| Agent | subclass of `SensingAgent`: 9 aux, (B, 3, 5) logits, per-section log-prob / entropy | no |
| PPO update | new `section_minibatch_update` (masked joint ratio, per-decision entropy, chunked) | no (imports `compute_gae`) |
| Trainer | new `train_spatiotemporal_ppo.py` (copy of the temporal trainer, STS paths, (T, N, 3) storage, per-section logs, same resume) | no |

### 5.3 Storage, memory, overhead

* Rollout storage adds `actions`, `logprobs` and `masks` at (T, N, 3) instead of (T, N), plus 6 aux columns:
  262,144 × 15 × 4 B ≈ **16 MB** on top of TMP02's 4.96 GB image buffer. GPU peak is unchanged
  (~38.8 GB first update with chunk 128, 15.7 GB after), which fits on GPUs 6 and 7.
* Per tick: the gate adds O(N·3) element-wise ops and one `torch.where` over (N, 34, 139) (TMP02 already does one),
  and the actor head grows from 5 to 15 outputs. Expect < 1 ms on a ~140 ms tick, **< 1 %**. The rollout stays bound
  by env launch overhead. The update grows negligibly (logits only). Expected ~43-48 s per iteration, ~9-10 h for 200
  M steps per member, as TMP02.
* The fast path (`fast_path: [all]`) is preserved unchanged, because the env is untouched.

### 5.4 Effect of asynchronous sections on frozen NAV05

NAV05 was trained only on **synchronous** 4 Hz images. STS feeds it composites whose sections can differ in age by up
to 5 s. That is out of distribution in ways TMP01 never tested:

* **Rotation tearing:** at w up to 2 rad/s, the scene shifts by a whole section (9.4°) in ~0.08 s. A stale section then
  shows obstacles at wrong bearings, with seams at columns 46 and 93.
* **Translation and moving obstacles:** v ≤ 0.22 m/s and obstacles at 0.04-0.08 m/s. Over 5 s that is up to 1.1 m of
  robot motion and 0.4 m of obstacle motion inside a stale section.
* **Weaker spatial prior than the reference.** There, the rear sector of a 360° LiDAR was the natural one to slow
  down. Here all three sections face forward within 28°, so the gain from spatial adaptivity is less certain.

The STS01 fixed-section factorial (section 6) measures all of this **before** any training. It also tells the sensing
policy which configurations NAV05 tolerates, for example whether the sides can drop to 1-2 Hz while the centre stays
at 4 Hz. NAV05 stays frozen, as required. Retraining navigation on asynchronous images is outside this plan.

### 5.5 Reuse

* Reused as is: the env, fast path, NAV05 loading and the contract check, `period_ticks`, `compute_gae`, the
  `SensingAgent` CNN/LSTM/critic, the trainer's run-folder / checkpoint / resume / TensorBoard machinery (copied),
  `tools/experiments.py` (already defines `STS` → `AdaptiveSensing/{configs,runs,eval}/spatiotemporal/`), the
  `FakeEnv` pattern of `tests/unit/test_temporal_sensing.py`, and the evaluation protocol `ref_chase_c10` and
  recording format of `evaluate_fixed_rate.py`.
* New: the section layout, section gate, wrapper, agent subclass, masked update, trainer, fixed-section evaluator,
  learned-policy evaluator and tests.

---

## 6. Implementation and validation plan

### 6.1 Isolation rules

* **No edits** to anything TMP02 imports: `AdaptiveSensing/evaluation/scan_gate.py`,
  `AdaptiveSensing/training/{temporal_sensing,sensing_agent,recurrent_ppo}.py`,
  `AdaptiveSensing/scripts/train_temporal_ppo.py`, `tools/experiments.py`, `source/st_nav/**`. Python has already
  loaded these modules in the running processes, but `--resume` would load the edited files from disk.
* GPUs 6 and 7 only, until TMP02 finishes.
* All STS outputs go under `AdaptiveSensing/{configs,runs,eval}/spatiotemporal/<STS ID>/`.

### 6.2 Experiments

| ID | Analogue | Purpose |
|---|---|---|
| `STS01_chase-ob1ob2_fixed-sections-nav05` | TMP01 / reference ST00 | constant per-section rates, 5³ = 125 configurations × 100 `ref_chase_c10` episodes, frozen NAV05; NAV05 robustness to asynchronous sections and the baseline Pareto set |
| `STS02_chase-ob1ob2_adaptive-nav05` | TMP02 / reference ST05, ST07 | learned three-section sensing policy, same scene, DR, NAV05, scale and frame-cost sweep as TMP02 |

### 6.3 New files

```
AdaptiveSensing/evaluation/sections.py                SectionLayout: sections from the env's grid_col (contiguous column
                                                      ranges, 46/47/46 default), col->section index, ray counts, weights w_s,
                                                      checks (disjoint, complete, S = 1 -> whole image). Kit-free.
AdaptiveSensing/evaluation/section_scan_gate.py       SectionScanGate (N, S): reset / step / set_rate / hz_in_force / fse;
                                                      composite image update with the column mask. Kit-free.
AdaptiveSensing/training/spatiotemporal_sensing.py    SpatiotemporalSensing: obs (image, 9 aux), step(action (N, 3)) ->
                                                      reward, masks (N, 3), per-section episode counters; reuses NAV05 helpers.
AdaptiveSensing/training/spatiotemporal_agent.py      SectionSensingAgent(SensingAgent): (B, 3, 5) logits, per-section
                                                      log-prob / entropy, deterministic argmax per section.
AdaptiveSensing/training/spatiotemporal_ppo.py        section_minibatch_update (section 3.3; ratio_mode joint | per_section;
                                                      chunked accumulation with minibatch-level denominators); compute_gae imported.
AdaptiveSensing/scripts/train_spatiotemporal_ppo.py   trainer (accepts STS IDs only; storage (T, N, 3); per-section logs; resume)
AdaptiveSensing/scripts/evaluate_fixed_sections.py    STS01: --sections fL fC fR | --factorial 0.2 0.5 1 2 4 (configurations
                                                      looped in one process, one env setup); ref_chase_c10; evaluate_fixed_rate.py format
                                                      + per-section columns
AdaptiveSensing/scripts/evaluate_spatiotemporal.py    STS02 checkpoints (argmax per section, LSTM reset per episode), same protocol
AdaptiveSensing/configs/spatiotemporal/STS02_chase-ob1ob2_adaptive-nav05.yaml
AdaptiveSensing/eval/spatiotemporal/STS01_.../README.md, STS02_.../README.md     experiment cards
tests/unit/test_sections.py, test_section_scan_gate.py, test_spatiotemporal_sensing.py,
tests/unit/test_spatiotemporal_agent.py, test_spatiotemporal_ppo.py
tests/sim/sts_validation.py                           simulator checks (6.4, S1-S3)
docs/EXPERIMENTS.md                                   two registry rows (documentation only)
```

### 6.4 Steps, each with its validation gate

1. **Layout and gate (Kit-free).** `sections.py` and `section_scan_gate.py`.
   * U1: the ray-table-derived layout gives 46/47/46 with rays 1564/1598/1564, and Σw = 1.
   * U2: on random action and native-time sequences, each section equals a literal per-section re-implementation of
     the real-robot gate (`real_eval_adaptive.py`), the same literal rule TMP02 is tested against.
   * U3: an accepting section changes only its columns, and the others stay bitwise equal.
   * U4: **uniform equivalence.** Identical actions in all sections give bitwise the `ScanGate` result: held image,
     deadlines, rates, and any(mask) = TMP02 mask.
2. **Wrapper (`FakeEnv`).**
   * U5: ignored-action invariance. Flipping every component with m = 0 leaves the image, ratios, rewards, masks and
     NAV05 inputs bitwise equal.
   * U6: FSE and reward decomposition, budget penalty once, terminal tick, reset attribution, reset observation
     (1, 0, 1/500).
   * U7: with uniform actions, `SpatiotemporalSensing` equals `TemporalSensing` bitwise (rewards, x_k aggregated,
     NAV05 inputs).
   * U8: the timing example from the TMP02 card, per section.
3. **Simulator (GPU 6).** `tests/sim/sts_validation.py`.
   * S1: `--uniform <hz>` for 4/1/0.5/0.2 Hz is identical to TMP01/TMP02 fixed-rate results, episode by episode.
     2 Hz is compared with the common re-run, because of the documented run-to-run variation.
   * S2: random actions, 100 envs × 1200 ticks, 0 violations of the per-section rule, masks, FSE and ratios.
   * S3: ignored-action invariance on a short horizon, with a same-process repeat as the control for simulator
     non-determinism.
4. **STS01 fixed-section factorial (GPUs 6-7).**
   * Validation: the five diagonal configurations equal TMP01.
   * Output: success / collision / FSE for 125 configurations, and the Pareto set.
   * Read-out: which sections NAV05 tolerates slowing, and how large the asynchrony damage is. This is the go/no-go
     for STS02 and the input to the cost choice.
5. **Agent and update (Kit-free).**
   * U9: shapes (B, 3); per-section log-prob = log-softmax; initial entropy ln 5; LSTM done-reset; replay reproduces
     the rollout log-probs.
   * U10: exact zero gradient on every unapplied component, logits *and* entropy; the actor rows of a never-applied
     section stay zero; joint ratio = product of applied ratios; all-zero minibatch gives finite zeros; the per-decision
     entropy example gives (0.60 + 0.30)/2 = 0.45.
   * U11: **S = 1 equals TMP02's `minibatch_update`** (losses, KL, gradients); chunked = unchunked.
   * Deliberately break each property and confirm the tests catch it (the reference's practice).
6. **Trainer (GPU 6).**
   * Smoke run (64 envs, 2 iterations).
   * 3-iteration benchmark at 1024 envs, chunk 128: iteration time and peak memory against TMP02.
   * Resume test (as TMP02's `resume_test/`).
   * The checkpoint stores `section_layout` (column ranges, ray counts, weights), `fps_choices`, `aux_layout` and
     `ratio_mode`. Evaluation refuses mismatches.
7. **STS02 training.** Config = TMP02's agent and env sections + `sections: [46, 47, 46]`. Sweep the TMP02 costs, two
   members at a time on GPUs 6-7, then all after TMP02 ends.
   * TensorBoard adds `charts/fse`, `charts/mean_applied_fps_<s>`, `rollout/applied_tick_fraction`,
     `rollout/applied_decisions_per_tick`, `rollout/section_applied_frac_<s>`, `losses/entropy_<s>`,
     `charts/mean_obs_age_s_<s>` and `charts/ray_weighted_fps`.
8. **Evaluation.** STS02 checkpoints on `ref_chase_c10` with the same seeds, then one Pareto figure with TMP01, TMP02,
   STS01 and STS02.

### 6.5 Mathematical formulation (summary)

```
Sections       S = {L, C, R}, columns C_L = [0, 46), C_C = [46, 93), C_R = [93, 139);  R_s = 34|C_s|;  w_s = R_s / 4726
Observation    o_k = (H_k, x_k),  H_k ∈ [0,1]^{1×34×139},  x_k = (φ_{k,s}, α_{k,s}, β_{k,s})_{s∈S}  (variable-major)
               H_k[:, :, C_s] = I_{F_{k,s}}[:, :, C_s]     image of section s at its last acquisition tick F_{k,s}
               φ = f/4,  α = clip(c/25, 0, 1),  β = clip(n/500, 0, 1)
Action         a_k ∈ {0..4}^3,  π_θ(a_k | h_k) = Π_s Cat(a_{k,s}; softmax(z_{k,s})),  h_k = LSTM(CNN(H_k) ⊕ x_k, h_{k−1})
Transition     section gate (2.2); NAV05 acts on H_{k+1}; env dynamics unchanged
Reward         r_k = r_k^NAV05 − λ Σ_s w_s m_{k,s} − ρ_b 1[first FSE_k > B]
Loss           section 3.3
```

### 6.6 Open decisions for you

1. Per-section β_s (your nine-variable spec, recommended) or the reference's single global FSE ratio (7 variables).
2. `ratio_mode`: joint (reference, recommended default) or per-section clipping as an ablation.
3. The frame-cost range for STS02: TMP02 values only, or also a 3× extension (decide after STS01 and early TMP02
   results).
4. STS01 scope: the full 125 configurations (recommended; one in-process loop) or {0.5, 1, 2, 4}³ = 64.
