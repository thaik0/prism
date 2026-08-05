# Lessons Learned Building Prism

## Why the original hypothesis seemed plausible

I started Prism from an intuition that still seems reasonable: if a cache can
recognize the context preceding a burst, it should be able to move the right data
before a reactive policy pays the first misses. The storage controller did not
need to be learned. It only needed a useful estimate of future demand, plus
record sizes, residency, capacity, and movement cost.

That division of labor was appealing because it gave ML one narrow job and kept
the final decision auditable. It also made latent working sets a natural
representation. Instead of predicting every record independently, I could learn
recurring overlapping groups slowly and update only their near-term activation
and intensity quickly.

What I underestimated was how many transformations separated a better factor
forecast from a different storage action.

## What I expected prediction to change

I expected the fast model to move a small number of high-value records just
before abrupt activations. The result I had in mind was not constant churn or a
large replacement of the cache. It was a targeted change at the transition: pay
for a promotion once, avoid several slow reads, and let the deterministic
controller reject weak guesses.

Milestone 4 initially looked like that idea was working. Predictive Greedy
(Prism) beat the reactive baselines on combined simulated cost. But the policy
made no test promotion. It had reached a good placement during validation and
kept it. That was the first point where the system result forced me to ask a
more precise question: was the predictor helping dynamically, or had the pipeline
learned a stable popularity structure and stopped there?

## Why forecast quality was not enough

The forecasting results were not imaginary. NMF recovered the planted fuzzy
working sets, the context model improved activation Brier score, and the
conditional-intensity model improved RMSE on untouched test data. Those were the
metrics I had declared in advance, and they passed.

But a storage action depends on rank and threshold, not just average predictive
error. Two forecast vectors can differ enough to improve Brier score or RMSE and
still order records almost identically. Even a changed ordering is irrelevant if
the changed records are already resident, cannot fit, or do not repay promotion.

I learned to treat model quality as evidence about one boundary, not as a proxy
for the whole system. The right systems question was never simply “is the
forecast better?” It was “does the improvement survive projection, change a
feasible target, and save more than it costs?”

## Where the signal disappeared

Milestone 5.5 gave the clearest answer. The factor forecasts moved in every
candidate cell, but 87--90% of projected record-score magnitude came from the
stable residual baseline. Consecutive candidate sets had Jaccard similarity
above 0.98. By the time the controller saw the scores, the part of the forecast
that moved was riding on top of a much larger stable component.

The controller then did exactly what I had designed it to do. It rejected
nonresident records whose expected savings did not repay movement, and it
discarded lower-density candidates when bytes were scarce. That behavior was not
a controller bug. It exposed that the forecast signal had too little economic
leverage.

The oracle disagreement mattered here. The learned and oracle targets were not
already the same, so the null result was not caused by a world with no better
placement. There was opportunity, but the learned forecast did not identify it
in a form the controller could use.

## Why migration economics mattered

I originally thought of promotion cost as a penalty that would prevent noisy
overreaction. It did that, but it also changed the meaning of a “good” forecast.
A record does not deserve promotion because demand is likely; it deserves
promotion only if the expected saved reads exceed the cost of moving its bytes.

Milestone 5 made the sensitivity concrete. At zero movement cost, one seed acted
dynamically and saved 243 simulated units relative to frozen placement. At the
baseline price, nearly every run stayed fixed. At a four-saved-read promotion
price, the conservative predictive placement became worse than Recent-Demand
Greedy in every seed. Policy ordering depended on economics, not forecast error
alone.

This changed how I think about ML for systems. A false positive is not uniformly
bad, and a true positive is not uniformly useful. Error has to be weighted by
object size, current residency, competing capacity, timing, and the repayment
window.

## What the sparse-regime experiments taught me

The sparse experiments were a disciplined attempt to give prediction more room.
They created fewer overlapping activations and much longer dormant intervals.
The regimes separated exactly as intended, so I could not dismiss the result as
a workload-generation failure.

Longer quiet periods still did not produce action. Horizons of two and four
windows accumulated more demand error, residual demand still dominated the
projection, and all four eligible cells matched frozen and recent-state
placement exactly. Two very-sparse one-window diagnostic runs did contain a
repaid promotion, but that cell had been declared ineligible before the sweep.
Moving the goalposts afterward would have hidden the result rather than learned
from it.

The lesson was uncomfortable but useful: making a workload look more favorable
to anticipation does not guarantee that the existing representation exposes the
right transient signal.

## What I would design differently

I would put actionability diagnostics into the first placement milestone instead
of adding them after a promising aggregate cost result. A policy report should
always say how often targets changed, which forecast component caused the
change, which candidates movement cost rejected, and whether each promotion was
used and repaid.

I would also model stable and transient demand as separate outputs at the
record-projection boundary. Prism's residual baseline was valuable—it helped the
learned placement beat training-only popularity—but it obscured the contribution
of fast signal in the combined score. A future design should be able to ask how
much fast-tier capacity, if any, is reserved for transient opportunity without
pretending that stable demand is unimportant.

Finally, I would choose an external trace only after measuring recurrence,
capacity pressure, and cold-start share. In the pinned LLMServingSim run, 94.22%
of test logical references were to blocks unseen in training, while the complete
reusable catalog fit in the simulator's post-weight capacity. That was a valid
transfer test, but it offered little of the pressure or recurrence Prism needed
to show differentiated placement.

## What I learned about building ML systems

The most valuable engineering choices were the ones that made it possible to
believe a negative result. Hidden truth lived in separate artifacts. Models fit
only chronological training populations. Policies replayed identical frozen
events. Scientific failures were retained instead of filtered. The actionability
thresholds were frozen before the sweep. Complete experiment roots could be
regenerated byte for byte.

The static controls mattered as much as the learned models. Validation-Final
Frozen (Prism) revealed that the apparent predictive policy was usually static
during test. Recent-State-Only (Prism ablation) showed that the fast
activation/intensity term rarely changed cost. Residual-Baseline-Only (Prism
ablation) showed that latent factor structure still contributed beyond stable
per-record demand. Without those controls, I could have told a confident but
incorrect story from the headline cost table.

I also learned to keep evidence classes separate. The C++ engine proves that
placement operations preserve bytes, capacity, and atomicity. The Python/C++
ledger proves semantic parity. LLMServingSim supplies simulator-native timing.
None of those facts by itself proves hardware performance or predictive value.

## What remains genuinely open

The original question is still interesting: can anticipatory placement repay
itself in a workload with real recurring transitions and constrained tiers? Prism
does not answer “never.” It answers “not with this forecast-to-projection path in
the precommitted regimes we tested.”

A credible next investigation would need a stronger action-level premise before
it needed a more sophisticated model. It should show that transient demand is
observable early, that the signal remains distinct at the record or block level,
that capacity pressure creates a meaningful choice, and that asynchronous
movement can finish and repay itself. Those conditions should be measured before
committing to deployment.

That is why canceling Milestone 9 is the honest closeout. A real heterogeneous
deployment would add concurrency, transfer, and hardware complexity without
repairing the missing causal link. Prism is complete as an experimental account
of where learned structure helped and where prediction stopped becoming action.
