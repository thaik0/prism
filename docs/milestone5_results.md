# Milestone 5 — Rigorous Evaluation Results

## Result in one sentence

The frozen 36-run sweep is engineering-valid and exactly reproducible, but it
does **not** support continued dynamic value from the current fast predictor:
Predictive Greedy (Prism) matched Validation-Final Frozen (Prism) in 35 runs and
matched Recent-State-Only (Prism ablation) in total cost in 34 runs.

## Why Milestone 4 required qualification

The accepted seed-1729 Milestone 4 run reported Predictive Greedy test access
cost `57,764`, promotion cost `0`, combined cost `57,764`, and hit rate
`0.540704`. Recent-Demand Greedy reported access cost `58,277`, promotion cost
`12,163.7827`, combined cost `70,440.7827`, and hit rate `0.535638`. Predictive
made zero test promotions, so most of its combined-cost advantage was avoided
migration rather than a large access advantage. Its factor calibration rows
were:

```text
[0.18229646, 0.00000000, 1.42444579]
[0.32132138, 0.30600196, 1.51758971]
[0.08509736, 0.00000000, 1.22688181]
[0.15981663, 0.00000000, 1.81588968]
```

The middle column is the activation/intensity coefficient; three of four were
zero. Low migration can be prudent, but it is not by itself evidence of
successful anticipation. Milestone 5 therefore separates validation-developed
static placement, recent learned-factor state, fast-predictor contribution, and
residual popularity.

## Frozen design

The committed manifest is `configs/milestone5_experiments.json`, SHA-256
`f09f10c6164d7641854d4d13c8927b2a44568d724d3718578ad6003a56287b56`.
It was committed and pushed before aggregate results were inspected. Seeds are
exactly `1729`, `2718`, and `31415`; IDs are `<variant>__seed_<seed>` in manifest
variant order and then seed order.

| Variant | Only changed value |
|---|---|
| `baseline` | capacity `0.25`; promotion `2` saved reads |
| `capacity_10` | capacity `0.10` |
| `capacity_40` | capacity `0.40` |
| `promotion_0` | promotion `0` saved reads |
| `promotion_1` | promotion `1` saved read |
| `promotion_4` | promotion `4` saved reads |
| `noise_low` | noise access weight `0.175` |
| `noise_high` | noise access weight `0.7` |
| `burst_short` | duration bounds `[1, 2]` windows |
| `burst_long` | duration bounds `[4, 10]` windows |
| `context_strong` | precursor scale `0.6875`; spontaneous probability `0.04` |
| `context_weak` | precursor scale `0.41250000000000003`; spontaneous probability `0.16` |

The accepted baseline values are noise `0.35`, duration `[2, 5]`, precursor
scale `0.55`, and spontaneous probability `0.08`. Fast/slow read costs remain
`1.0/10.0`. Trace-specific resolved capacities for seeds `1729/2718/31415` are
`154728/120762/155259` bytes at 25%, `61891/48305/62103` at 10%, and
`247565/193220/248415` at 40%. Baseline promotion costs per byte are
`0.0017976630/0.0029244517/0.0018614271`; the 1- and 4-read variants are exactly
one half and twice those values, and the zero-cost variant is zero.

Every run evaluates exactly these policies, in this order:

1. LRU
2. LFU
3. Recent-Demand Greedy
4. Predictive Greedy (Prism)
5. Training-Popularity Static (Prism)
6. Validation-Final Frozen (Prism)
7. Recent-State-Only (Prism ablation)
8. Activation/Intensity-Only (Prism ablation)
9. Residual-Baseline-Only (Prism ablation)
10. Oracle Greedy
11. Oracle Exact

The two controls have independent state. Training popularity uses training
windows only and is placed once before validation. Validation-final copies
Predictive residency at the test boundary and never changes it. The three
ablations mechanically remove terms from the original projection without
refitting coefficients. All policies receive identical events and storage
settings; hidden truth affects post-replay diagnostics only, and future demand
affects oracle policies only.

## Execution and validity

Both full roots contain 36 completed, zero failed, and zero pending runs. Resume
hash-validated and reused all 36 completed runs without changing any file. A
recursive comparison of the independently generated roots had exit status zero,
so source, intermediate, simulation, index, JSON, NPZ, and Markdown artifacts
were byte-identical.

Scientific gates were retained as outcomes rather than used as filters:

| Gate | Passed | Failed |
|---|---:|---:|
| Workload demonstrations | 36 | 0 |
| Workload intensity signal | 36 | 0 |
| Structure recovery | 36 | 0 |
| Predictor gate 1 | 36 | 0 |
| Predictor gate 2 | 22 | 14 |
| Predictor gate 3 | 29 | 7 |
| Simulation gate 1 | 28 | 8 |
| Simulation gate 2 | 35 | 1 |
| Simulation gate 3 | 31 | 5 |
| Simulation gate 4 | 36 | 0 |
| Simulation gate 5 | 33 | 3 |

Predictor failures vary by seed and configuration; they do not invalidate fitted
models. Simulation gate 1 failed for `baseline` seed `2718`, `capacity_40` seed
`2718`, all three `promotion_4` seeds, `noise_low` seed `2718`, `noise_high`
seed `2718`, and `context_strong` seed `2718`. Simulation gate 2 failed only for
`capacity_40__seed_2718`; gate 3 failed there, for all `promotion_4` seeds, and
for `noise_low__seed_2718`. Gate 5 failed for all `burst_short` seeds. Every
exact solve still proved its per-window optimum; different myopic trajectories
can make Oracle Exact's complete-run cost exceed Oracle Greedy's.

## Predictive Greedy results

Costs are seed `1729/2718/31415`. Sample standard deviation uses `ddof=1`.
`Burst` and `first two` are mean transition combined costs. `Coverage` is mean
pre-transition realized-demand coverage. Test promotion cost is zero in every
row: 35 runs never promoted, and promotions in `promotion_0` were
free.

| Variant | Seed costs | Mean | Median | SD | Min–max | Hit rate | Burst | First two | Coverage | Mean target changes / promotions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 57764/63188/51837 | 57596.3 | 57764 | 5677.4 | 51837–63188 | .5594 | 35604.0 | 51138.7 | .6331 | 0 / 0 |
| `capacity_10` | 69977/63188/74103 | 69089.3 | 69977 | 5511.4 | 63188–74103 | .4498 | 42753.0 | 61518.7 | .5076 | 0 / 0 |
| `capacity_40` | 57764/63188/51837 | 57596.3 | 57764 | 5677.4 | 51837–63188 | .5594 | 35604.0 | 51138.7 | .6331 | 0 / 0 |
| `promotion_0` | 40943/31868/42765 | 38525.3 | 40943 | 5837.0 | 31868–42765 | .7422 | 23775.0 | 34080.7 | .7998 | 18.7 / 28.0 |
| `promotion_1` | 40943/33398/42144 | 38828.3 | 40943 | 4741.0 | 33398–42144 | .7393 | 23910.0 | 34338.7 | .7963 | 0 / 0 |
| `promotion_4` | 99434/79118/100689 | 93080.3 | 99434 | 12108.0 | 79118–100689 | .2196 | 57693.0 | 82764.7 | .2683 | 0 / 0 |
| `noise_low` | 54483/62579/50704 | 55922.0 | 54483 | 6066.9 | 50704–62579 | .5754 | 35813.0 | 51632.0 | .6494 | 0 / 0 |
| `noise_high` | 60237/65420/51901 | 59186.0 | 60237 | 6820.5 | 51901–65420 | .5392 | 36284.0 | 52672.0 | .6138 | 0 / 0 |
| `burst_short` | 60701/65300/52342 | 59447.7 | 60701 | 6569.3 | 52342–65300 | .5470 | 52009.0 | 58850.0 | .6272 | 0 / 0 |
| `burst_long` | 57966/58241/49299 | 55168.7 | 57966 | 5085.1 | 49299–58241 | .5864 | 22866.7 | 36914.7 | .6703 | 0 / 0 |
| `context_strong` | 59233/62446/51280 | 57653.0 | 59233 | 5748.2 | 51280–62446 | .5522 | 36469.0 | 51704.3 | .6262 | 0 / 0 |
| `context_weak` | 53711/54095/52444 | 53416.7 | 53711 | 864.0 | 52444–54095 | .5884 | 32953.3 | 47254.3 | .6767 | 0 / 0 |

These are simulated costs, not measured latency. Variation in workload size
means raw cost should be compared within paired variant/seed runs, not treated as
a normalized cross-variant performance score.

## Static controls

Each cell is mean combined cost / hit rate / burst-start cost. Paired values are
Predictive minus the named control at seeds `1729/2718/31415`; negative favors
Predictive. `W/E/L` counts Predictive wins, numerical equals, and losses.

| Variant | Predictive | Training static | Frozen | vs training; W/E/L | vs frozen; W/E/L |
|---|---:|---:|---:|---:|---:|
| `baseline` | 57596/.5594/35604 | 60413/.5326/37362 | 57596/.5594/35604 | -2025/-1107/-5319; 3/0/0 | 0/0/0; 0/3/0 |
| `capacity_10` | 69089/.4498/42753 | 69458/.4463/42984 | 69089/.4498/42753 | 0/-1107/0; 1/2/0 | 0/0/0; 0/3/0 |
| `capacity_40` | 57596/.5594/35604 | 60413/.5326/37362 | 57596/.5594/35604 | -2025/-1107/-5319; 3/0/0 | 0/0/0; 0/3/0 |
| `promotion_0` | 38525/.7422/23775 | 38318/.7442/23601 | 38606/.7415/23802 | 0/0/621; 0/2/1 | 0/0/-243; 1/2/0 |
| `promotion_1` | 38828/.7393/23910 | 41273/.7162/25353 | 38828/.7393/23910 | 0/-4743/-2592; 2/1/0 | 0/0/0; 0/3/0 |
| `promotion_4` | 93080/.2196/57693 | 93080/.2196/57693 | 93080/.2196/57693 | 0/0/0; 0/3/0 | 0/0/0; 0/3/0 |
| `noise_low` | 55922/.5754/35813 | 60938/.5274/39008 | 55922/.5754/35813 | -9675/-1008/-4365; 3/0/0 | 0/0/0; 0/3/0 |
| `noise_high` | 59186/.5392/36284 | 61871/.5132/37898 | 59186/.5392/36284 | -2808/-945/-4302; 3/0/0 | 0/0/0; 0/3/0 |
| `burst_short` | 59448/.5470/52009 | 63036/.5128/55165 | 59448/.5470/52009 | -6651/0/-4113; 2/1/0 | 0/0/0; 0/3/0 |
| `burst_long` | 55169/.5864/22867 | 59540/.5466/24694 | 55169/.5864/22867 | -1809/-10683/-621; 3/0/0 | 0/0/0; 0/3/0 |
| `context_strong` | 57653/.5522/36469 | 59546/.5343/37759 | 57653/.5522/36469 | 0/-1179/-4500; 2/1/0 | 0/0/0; 0/3/0 |
| `context_weak` | 53417/.5884/32953 | 59717/.5265/36898 | 53417/.5884/32953 | -6579/-8010/-4311; 3/0/0 | 0/0/0; 0/3/0 |

All three policies have zero test promotion cost except dynamic Predictive under
`promotion_0`, where the cost per byte is zero. Predictive beat training static
in a majority of seeds for 9 of 12 variants, tied it at `promotion_4`, produced
only one win at `capacity_10`, and lost once at `promotion_0`. This supports
value beyond training-only popularity, but frozen equivalence shows that the
value was normally acquired during validation rather than through test updates.

## Forecast ablations

Each policy cell is mean total / burst-start cost / realized pre-transition
coverage. Paired values again retain all three seed differences and `W/E/L`.

| Variant | Predictive | Recent state | Activation/intensity | Residual | vs recent; W/E/L | vs activation; W/E/L | vs residual; W/E/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 57596/35604/.6331 | 57596/35604/.6331 | 58640/36273/.6239 | 63815/39573/.5744 | 0/0/0; 0/3/0 | -2025/-1107/0; 2/1/0 | -6210/-7128/-5319; 3/0/0 |
| `capacity_10` | 69089/42753/.5076 | 69089/42753/.5076 | 69458/42984/.5060 | 71465/44271/.4823 | 0/0/0; 0/3/0 | 0/-1107/0; 1/2/0 | 0/-7128/0; 1/2/0 |
| `capacity_40` | 57596/35604/.6331 | 57596/35604/.6331 | 58640/36273/.6239 | 63815/39573/.5744 | 0/0/0; 0/3/0 | -2025/-1107/0; 2/1/0 | -6210/-7128/-5319; 3/0/0 |
| `promotion_0` | 38525/23775/.7998 | 38540/23784/.7999 | 38606/23802/.7999 | 38606/23802/.7999 | 0/0/-45; 1/2/0 | 0/0/-243; 1/2/0 | 0/0/-243; 1/2/0 |
| `promotion_1` | 38828/23910/.7963 | 38828/23910/.7963 | 39692/24453/.7905 | 42719/26226/.7689 | 0/0/0; 0/3/0 | 0/0/-2592; 1/2/0 | 0/-5994/-5679; 2/1/0 |
| `promotion_4` | 93080/57693/.2683 | 93080/57693/.2683 | 93080/57693/.2683 | 98915/61362/.1991 | 0/0/0; 0/3/0 | 0/0/0; 0/3/0 | -4248/-12168/-1089; 3/0/0 |
| `noise_low` | 55922/35813/.6494 | 55922/35813/.6494 | 58973/37715/.6221 | 63656/40784/.5768 | 0/0/0; 0/3/0 | -4788/0/-4365; 2/1/0 | -11142/-7002/-5058; 3/0/0 |
| `noise_high` | 59186/36284/.6138 | 59186/36284/.6138 | 60620/37178/.6050 | 64361/39536/.5669 | 0/0/0; 0/3/0 | 0/0/-4302; 1/2/0 | -4095/-7128/-4302; 3/0/0 |
| `burst_short` | 59448/52009/.6272 | 60555/52996/.6174 | 61926/54211/.6076 | 66207/58045/.5649 | -3321/0/0; 1/2/0 | -3321/0/-4113; 2/1/0 | -9162/-7002/-4113; 3/0/0 |
| `burst_long` | 55169/22867/.6703 | 55169/22867/.6703 | 58562/24277/.6266 | 64211/26689/.5755 | 0/0/0; 0/3/0 | 0/-9558/-621; 2/1/0 | -4923/-16983/-5220; 3/0/0 |
| `context_strong` | 57653/36469/.6262 | 57653/36469/.6262 | 57653/36469/.6262 | 62855/39814/.5786 | 0/0/0; 0/3/0 | 0/0/0; 0/3/0 | -4203/-6903/-4500; 3/0/0 |
| `context_weak` | 53417/32953/.6767 | 53417/32953/.6767 | 59420/36712/.6040 | 63176/39100/.5633 | 0/0/0; 0/3/0 | -6579/-7119/-4311; 3/0/0 | -10935/-14031/-4311; 3/0/0 |

The full and Recent-State-Only costs are equal in 34 runs. The only differences
are `promotion_0__seed_31415` (`-45`) and `burst_short__seed_1729` (`-3321`).
The latter still had no test action: the fast term changed the placement reached
during validation, not continued test behavior. Strong, baseline, and weak
context all show zero full-versus-recent differences for every seed, so the
fast-predictor contribution does not strengthen with context reliability.
Factor-plus-residual information is useful relative to residual alone: the full
policy wins all seeds in 9 variants and one or two seeds in the other three.

Complete residency traces quantify the mechanical differences. Predictive was
identical to recent state in 34 runs (415 total record-window disagreements), to
activation/intensity-only in 19 runs (3,708 disagreements), and to residual-only
in 5 runs (14,908 disagreements). The two full-versus-recent differences are
the 15 record-window disagreements in `promotion_0__seed_31415` and 400 in
`burst_short__seed_1729`.

## Dynamic and pre-transition findings

Only `promotion_0__seed_31415` had `dynamic_test_action=true`. It changed target
sets in 56 of 200 windows (`28%`), made 84 promotions totaling `685,468` bytes,
and differed from frozen in 36 windows and 108 record-windows. It differed from
Recent-State-Only in 5 windows and 15 record-windows. The run recorded 7 useful
burst-start promotions and 10 useful first-two-window promotions.

That dynamic run cost `243` less than frozen, but mean realized pre-transition
coverage was slightly lower (`0.749855` versus `0.750047`); supported-record
coverage was also lower (`0.529778` versus `0.536231`). Thus it demonstrates one
useful behavioral cost change, not robustly improved anticipation. All other 35
runs had zero target changes, promotions, or bytes promoted and exactly matched
the frozen residency trace.

Baseline pre-transition means show where opportunity remains:

| Policy | Supported-record | Membership-weighted | Supported-byte | Realized-demand | Burst cost | First-two cost |
|---|---:|---:|---:|---:|---:|---:|
| Predictive Greedy (Prism) | .4236 | .5768 | .2849 | .6331 | 35604.0 | 51138.7 |
| Training-Popularity Static (Prism) | .4025 | .5510 | .2713 | .6132 | 37362.0 | 53691.7 |
| Validation-Final Frozen (Prism) | .4236 | .5768 | .2849 | .6331 | 35604.0 | 51138.7 |
| Recent-State-Only (Prism ablation) | .4236 | .5768 | .2849 | .6331 | 35604.0 | 51138.7 |
| Activation/Intensity-Only (Prism ablation) | .4071 | .5665 | .2767 | .6239 | 36273.0 | 52065.7 |
| Residual-Baseline-Only (Prism ablation) | .3602 | .5173 | .2427 | .5744 | 39573.0 | 56793.7 |
| Recent-Demand Greedy | .3746 | .4898 | .3215 | .5588 | 43033.3 | 60431.6 |
| Oracle Greedy | .4481 | .6101 | .3738 | .7203 | 28565.2 | 40409.1 |
| Oracle Exact | .4482 | .6115 | .3750 | .7214 | 28517.2 | 40342.9 |

Across the three baseline seeds, Predictive made zero useful pre-transition
promotions; Recent-Demand made 458 start-window and 558 first-two useful
promotions, while the oracles made about 1,164. These counts describe different
policies and are not normalized causal effects.

## Experiment-family findings

- **Capacity:** 10% worsened Predictive mean cost to `69,089.3`, coverage to
  `.5076`, and hit rate to `.4498`. The 25% and 40% results were identical because
  the Predictive positive-benefit set did not fill the extra capacity; mean
  occupancy fractions at 40% were only `.3707/.2082/.4425` across seeds. Mean
  oracle regret was `4,564.3`, `12,430.8`, and `13,906.6` at 10/25/40%.
- **Promotion cost:** Predictive mean cost moved from `38,525.3` at zero, to
  `38,828.3` at one, `57,596.3` at two, and `93,080.3` at four saved reads. It
  acted dynamically only in one zero-cost seed. At four reads Recent-Demand
  (`80,966.1`) beat Predictive in all three seeds, showing that conservative
  validation placement can become poor when the storage objective changes.
- **Noise:** low/baseline/high Predictive means were
  `55,922.0/57,596.3/59,186.0`; realized coverages were
  `.6494/.6331/.6138`. Projection RMSE was `1.2723/1.2796/1.2559`, so policy cost
  did not follow projection RMSE monotonically. Predictive beat Recent-Demand in
  two seeds and lost one at each level; the result is mixed.
- **Duration:** short/baseline/long means were
  `59,447.7/57,596.3/55,168.7`; first-two costs were
  `58,850.0/51,138.7/36,914.7`, and projection RMSE was
  `1.4647/1.2796/1.1984`. No duration run made a test promotion. All short runs
  failed the exact-controller trajectory gate despite optimal per-window solves.
- **Context:** strong/baseline/weak means were
  `57,653.0/57,596.3/53,416.7`; projection RMSE was
  `1.2155/1.2796/1.2787`. Predictive and Recent-State-Only were identical at all
  three levels, so the lower weak-context cost cannot be attributed to the fast
  predictor and should not be interpreted as monotonic context benefit.

## Migration-cost diagnostics

The behavioral family above is primary because controllers react to migration
price. Ranking changes materially: Predictive is behind only the oracles at one
saved read, but Recent-Demand beats it at four. At zero, Training-Popularity
Static has the lowest mean among non-oracle policies (`38,318.3`) and Predictive
is next (`38,525.3`); its 28 mean test promotions arise entirely from one seed.

Holding baseline trajectories fixed, only seed `2718` has a meaningful
Predictive-versus-Recent-Demand crossover: `0.0032636561321765342` cost units per
promoted byte. Seeds `1729` and `31415` produce negative/nonmeaningful
crossovers because Predictive dominates both access and promoted bytes. All
Predictive comparisons with frozen, training static, and recent state have a
zero promoted-byte denominator and are undefined. These accounting crossovers
are local diagnostics, not substitutes for behavioral runs: changing promotion
cost can change placement decisions and trajectories.

## Oracle regret and descriptive error terms

Means across seeds are shown below. Regret is Predictive minus Oracle Greedy.
`Access` and `migration` are the corresponding cost differences; `missed` is
access cost attributable to Oracle-Greedy records missing from Predictive at
burst boundaries. `Greedy-exact` is Oracle Greedy minus Oracle Exact.

| Variant | Regret | Access | Migration | Projection RMSE | Missed access cost | Greedy-exact |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 12430.8 | 23679 | -11248.2 | 1.2796 | 20931 | 78.2 |
| `capacity_10` | 4564.3 | 11640 | -7075.7 | 1.2796 | 13107 | 322.6 |
| `capacity_40` | 13906.6 | 26043 | -12136.4 | 1.2796 | 23115 | 0.0 |
| `promotion_0` | 14226.0 | 14226 | 0.0 | 1.2796 | 13686 | 72.0 |
| `promotion_1` | 3804.6 | 12312 | -8507.4 | 1.2796 | 11817 | 147.4 |
| `promotion_4` | 28390.5 | 38547 | -10156.5 | 1.2796 | 33363 | 0.0 |
| `noise_low` | 11480.0 | 22665 | -11185.0 | 1.2723 | 20640 | 78.5 |
| `noise_high` | 13344.2 | 24465 | -11120.8 | 1.2559 | 21537 | 81.9 |
| `burst_short` | 12932.4 | 26724 | -13791.6 | 1.4647 | 44844 | -22.3 |
| `burst_long` | 10380.2 | 20295 | -9914.8 | 1.1984 | 10368 | 64.7 |
| `context_strong` | 12760.4 | 23610 | -10849.6 | 1.2155 | 22911 | 67.9 |
| `context_weak` | 8333.5 | 20154 | -11820.5 | 1.2787 | 17754 | 28.0 |

Regret is positive in every seed and variant. Predictive saves migration relative
to Oracle Greedy but loses more access cost, while missed-oracle accesses and
target disagreement show unrealized placement opportunity. The terms are
descriptive and are not claimed to form an exact additive causal decomposition.
The oracles are one-window, not full-horizon, optima.

## Deterministic hypothesis summaries

The exact rule is: `supported` requires the expected direction for all three
seeds in at least one relevant non-baseline variant without uniform majority
contradiction; `not_supported` means no relevant variant has a seed majority;
`mixed` means evidence materially varies; and `insufficient_data` means a
required variant lacks all three seeds.

| Hypothesis | Status | Evidence and counterexample |
|---|---|---|
| Dynamic value | `not_supported` | Frozen equality in 35 runs; one `promotion_0` seed improved by `243` |
| Fast-predictor contribution | `not_supported` | Full minus recent-state is zero for all context-family seeds |
| Beyond static popularity | `supported` | All-seed wins at baseline, 40%, low/high noise, long burst, and weak context; zero-cost seed `31415` loses |
| Factor and residual contribution | `supported` | Full beats residual in all seeds for 9 variants; 10% and zero-cost provide only one win |
| Capacity pressure | `supported` | All-seed static-control wins at 25%/40%; only one win at 10% and unused capacity at 40% |
| Noise tolerance | `mixed` | Two wins and one loss versus Recent-Demand at each noise level |
| Burst duration | `supported` | Full beats training static in all long-burst seeds and two short-burst seeds, but remains test-static |
| Oracle regret | `supported` | Predictive cost exceeds Oracle Greedy in all 36 runs |

These labels summarize the frozen sweep; they are not project pass/fail gates.

## Reproduction

```bash
PYTHONPATH=src python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --output-dir /tmp/prism_milestone5_a

PYTHONPATH=src python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --output-dir /tmp/prism_milestone5_a \
  --resume

PYTHONPATH=src python3 -m prism.experiments.cli \
  --manifest configs/milestone5_experiments.json \
  --output-dir /tmp/prism_milestone5_b

diff -ru /tmp/prism_milestone5_a /tmp/prism_milestone5_b
```

The verified full regression command produced `206 passed`; `compileall` and
`pip check` also succeeded. The normal test suite uses reduced fixtures and does
not execute the 36-run sweep.

## Limitations and explicit non-goals

Three seeds give descriptive mean, median, sample standard deviation, minimum,
and maximum only. There are no confidence intervals or significance tests.
Scientific gates fail in some deliberately varied configurations. Costs are
simulated units and omit model CPU time, queues, contention, asynchronous
migration, filesystem effects, and wall-clock storage latency. Undefined
break-even values are preserved as null. Raw comparisons across variants are
confounded by different generated event counts; paired within-variant comparisons
are the causal focus.

No predictor, feature set, factorization, hyperparameter, split, projection
refit, controller objective, policy, seed, or variant was changed after aggregate
inspection. This milestone adds no TinyLFU, random replacement, adaptive factor
count, full factorial sweep, pandas, plot, notebook, database, parallelism, real
storage, C++, pybind11, GPU, LLM, or later-phase implementation.
