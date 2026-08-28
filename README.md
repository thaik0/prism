# Prism
**Stack:** Python, C++17, pybind11, NumPy, scikit-learn, AWS

Prism is an ongoing, experimental storage-tiering system that studies whether learned access patterns can help decide which data should stay in fast memory and which data can remain on slower storage.

The idea was predictive: learn demand patterns well enough to move data into memory before a burst of accesses arrives. The experiments produced a more nuanced result. Forecasts became more accurate, but those improvements did not reliably translate into better placement decisions after accounting for limited memory and data-movement cost.

The final system therefore uses learned demand estimates for stable, cost-aware placement rather than claiming successful predictive prefetching.

## Architecture
```
       Access history
             |
             v
    Learn demand structure
             |
             v
      Estimate demand
             |
             v
     Project to records
             |
             v
   Cost-aware controller <---- sizes, capacity,
             |                 residency, move cost
             v
        RAM / disk
```
ML estimates demand, while the controller chooses placement. ML never will directly move the data.

## What I found

I evaluated Prism on controlled synthetic workloads where the hidden demand structure was known.

The learned representation recovered the underlying overlapping working sets with 0.950 mean cosine similarity and 0.785 support recall, compared with 0.25 analytic-chance recall.

Synthetic hidden truth is separate from model-visible events. Compared policies
receive identical frozen traces. ML never chooses storage actions directly.
But better forecasts did not reliably create better storage actions. In a frozen 36-run placement experiment, the predictive policy matched the final stable Prism policy in 35 of 36 runs. A simpler recent-state-only ablation also matched its cost in 34 runs. I then tested additional predictive settings designed to make future demand shifts more actionable. None produced placement changes that passed the pre-specified evaluation criteria.

My main conclusion is that prediction quality and decision quality are not the same thing.

## Native storage engine
To test the placement logic against an actual storage implementation rather than only a Python simulator, I built a C++17 two-tier store with a bounded in-memory tier, file-backed storage, CRC-verified reads, deterministic placement transitions, and capacity enforcement.

I also built a parity test that runs the same storage operations through the Python model and native engine. Across 130,349 operations from four policy paths, the implementations produced zero state mismatches and zero capacity violations. 

## LLM-serving experiment
As an additional application test, I integrated Prism with a pinned LLM-serving simulator and used its placement decisions for simulated KV-cache storage.

In that experiment, static placement, frozen learned placement, predictive placement, LFU, and native LRU produced identical time-to-first-token and recomputation results.

Even an oracle policy did not improve latency: it transferred an additional 56 MiB and increased mean time-to-first-token by 1.401 ms.

That result reinforced the earlier finding: knowing future demand more accurately is useful only when the resulting placement change is worth its movement cost.
