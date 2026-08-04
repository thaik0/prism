# Prism v1.0.0

Prism is a completed experimental storage-tiering project. It learns recurring
latent demand, projects demand to records or reusable prefix blocks, and gives
those estimates to a deterministic byte- and movement-cost-aware controller.

The final thesis is:

> **Prism learns latent-demand structure and uses it for stable, cost-aware
> storage tiering.**

## Included

- controlled seeded workloads with separate simulator-only hidden truth;
- deterministic fuzzy working-set recovery and fixed activation/intensity
  predictors;
- record-demand projection, deterministic placement, static controls, forecast
  ablations, and oracle diagnostics;
- frozen 36-run evaluation and 27-run actionability diagnosis;
- standalone C++17 immutable RAM/file-backed storage engine;
- exact Python/C++ operation-level parity for four certified policy paths;
- pinned LLMServingSim reusable prefix-KV integration with simulator-native
  timing and cache behavior;
- final technical report, lessons learned, documentation index, and verified
  reproducibility commands.

## Principal finding

Forecast metrics improved, but continued fast prediction did not reliably create
dynamic placement value. Predictive Greedy (Prism) matched frozen placement in
35 of 36 Milestone 5 runs, and every precommitted sparse multi-window Milestone
5.5 candidate failed all actionability gates. Stable learned placement did retain
value relative to training-only popularity in many tested cells.

The pinned LLMServingSim experiment did not reverse that conclusion: five
non-Oracle policies had identical TTFT and recomputation, while Oracle added
transfer traffic and was slightly slower.

## Important limitations

Prism is not production-ready. Controlled placement costs are simulated; the
native engine is synchronous and untimed; the Python boundary holds the GIL; the
LLM result covers one pinned simulator configuration; and no real
heterogeneous-hardware advantage was demonstrated. Milestone 9 was intentionally
canceled rather than adding deployment complexity without a supported
actionability claim.

Reproduction starts in [README.md](README.md) and
[docs/reproducibility.md](docs/reproducibility.md). The complete evidence and
claim boundaries are in [docs/final_report.md](docs/final_report.md).
