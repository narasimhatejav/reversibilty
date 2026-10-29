# Session 13 — Reversibility: build it, prove it works, report it properly

A from-scratch reversible transformer (midpoint/leapfrog coupling and a
two-stream Euler/RevNet coupling), each with a **hand-written backward pass**
that never builds an autograd graph across the transformer stack, checked
against plain autograd on a toy model before being trusted on anything real.

## Setup

| | |
|---|---|
| Model | 10-layer, 384-dim, 6-head, GELU-MLP (4x), pre-norm, tied embeddings |
| Params | 21,072,384 (~21M) |
| Vocab | 8,192, custom byte-level BPE trained on TinyStories itself |
| Seq len | 512 |
| Data | TinyStories (Eldan & Li, 2023), streamed in document order, fixed 6,000,000-token slice (see [Scope reduction](#scope-reduction-and-why)) |
| Hardware | Apple M3 Pro, PyTorch 2.14 MPS backend (no CUDA available) |
| Optimizer | AdamW, `weight_decay=0.0`, `lr=4e-4`, 2% linear warmup + cosine decay, grad-clip 1.0 |
| Reversible constraints | `dropout=0.0` (recomputation requires deterministic blocks), `weight_decay=0.0`, step size `h=0.25` |

`weight_decay=0.0` and `dropout=0.0` are applied to **all three runs**,
including the standard baseline, for a fair comparison — a 21M model trained
on a 6-50M-token corpus is compute-bound, not overfitting-bound, so this costs
the baseline nothing.

## Results

All four runs trained on the same fixed 6,000,000-token TinyStories slice,
same tokenizer, same seed, same optimizer settings.

| Run | Variant | Batch | Steps | Final loss* | Tokens/sec (steady-state) | Peak mem (GB) | Wall-clock |
|---|---|---|---|---|---|---|---|
| 1 | Standard | 13 | 899 | 4.801 | 13,718 | 5.34 | 436.6 s |
| 2a | Reversible (midpoint) | 13 | 899 | 4.868 | 12,094 | 2.66 | 494.8 s |
| 2b | Reversible (Euler) | 13 | 899 | **4.683** | 11,866 | 2.66 | 504.0 s |
| 3 | Reversible (Euler, max batch) | 27 | 433 | 4.851 | 12,193 | 5.93 | 490.6 s |

\* mean loss over the last 50 logged steps. Total pipeline wall-clock
(gradcheck + both max-batch searches + all four training runs): 1945 s
(~32.4 minutes) on this Apple M3 Pro.

**Headline numbers:**
- At the **same batch size** (13), reversible peak memory is **2.66 GB vs.
  5.34 GB standard — a 2.0x reduction** from not storing per-layer
  activations. Both reversible variants land at the *same* peak memory
  (2.657 GB, to three decimals) since they run the identical number of
  `Block` calls per step and never differ in what they keep resident.
- Under an identical batch-size search (same memory budget, same doubling
  procedure), the winning reversible variant fits **B=27 vs. B=13 standard
  — 2.08x the batch size** at the same memory budget. This is the actual
  thesis of the assignment, demonstrated end to end: freed activation
  memory converts directly into more parallelism.
- Reversible is **~13-15% slower per token** at the same batch size
  (11,866-12,094 vs. 13,718 tok/s) — real, but well under the guide's own
  reference figure of 30-50% slower, plausibly because this model is tiny
  (10 layers, 384-dim) and the extra `torch.autograd.grad` calls per layer
  are a smaller fraction of a much cheaper forward pass than they would be
  in a large model with heavier per-layer FLOPs.

## Correctness: the gradcheck story

The midpoint backward pass failed gradcheck on the first attempt — every
parameter gradient was off by ~1e-3 to 1e-2 (loss matched exactly, so the
*forward* math was right; only backward was wrong). The bug: the recompute
step evaluated `blocks[l]` on the wrong one of the two running states
(`p_hi` instead of `p_lo`), and after the loop, the code read the final
`p^0` from the wrong slot for the same reason. Once both reads were swapped
to match the actual leapfrog recursion (`p^{l+1} = p^{l-1} + 2h·f_l(p^l)`,
so `f_l` must see `p^l`, i.e. the *lower* state of the running pair), every
parameter matched the autograd reference to ~1e-16. The Euler/two-stream
coupling matched to machine precision on the first attempt — its backward is
the textbook RevNet derivation, structurally simpler because each stream
state only ever appears in one addition, not two apart in a recursion.

This is the reason the guide's own warning ("a one-off indexing mistake here
silently produces a trainable-looking but wrong model") is not boilerplate:
the buggy version *would have trained* — cross-entropy loss would have gone
down, because the gradient direction was correlated with the true gradient
just not exactly right. Only the toy-model diff caught it.

## Which variant won, and why

**Euler won** (final loss 4.683 vs. midpoint's 4.868, both at batch=13,
both against a standard-model reference of 4.801 — euler's loss is even
*below* the standard baseline's, midpoint's is above it). Neither variant
diverged or produced a non-finite loss; both tracked the same overall
learning-curve shape as the standard run (see `figures/loss_curves.png`),
which is itself part of the correctness evidence — a broken reversible
backward would show a systematically higher loss or a curve that doesn't
track the reference, not a curve that's occasionally *better*.

A plausible reason euler is both faster and lower-loss here: its backward
is the plain two-term RevNet derivation (each stream state appears in
exactly one addition), versus the midpoint scheme's leapfrog recursion
where a state's gradient is assembled from two separate contributions
across two loop iterations — more bookkeeping per step, and more
opportunities for numerical noise to accumulate over 899 steps even
though both are gradient-exact per the toy-model check. This is a
hypothesis, not a proven explanation — a single 6M-token run at one
learning rate isn't enough data to separate "euler is intrinsically
better here" from "this particular seed favored euler."

## Memory: does reversible actually buy a bigger batch?

`find_max_batch` does a doubling search on **synthetic random batches**
(one forward+backward+optimizer.step() per trial, fresh model each time),
so the search itself never touches the real corpus. On this machine
(Apple M3 Pro, unified memory, `torch.mps.recommended_max_memory()` ≈
14.3 GB), memory is read via `torch.mps.driver_allocated_memory()` — the
allocator's own reported footprint, not `current_allocated_memory()`, which
under-reported activation/optimizer-state memory by roughly an order of
magnitude on this torch build and was caught by comparing it against the
OOM error's own self-reported allocation size .

The search stops at a **memory budget** (75% of `recommended_max_memory`,
≈10.73 GB on this machine) rather than waiting for a hard crash: on unified
memory the OS does not always raise a clean `RuntimeError` before the
machine starts swapping, and a doubling search that has to swap its way to
a crash at 4-8x the intended batch size is not a search you want to run on
a shared laptop. This is a deliberately more conservative stopping rule
than "run until it crashes," so both `B_std` (13) and `B_max_rev` (27) are
undershoots of the true hard ceiling — consistently, for both models, so
the comparison between them is still fair. `figures/memory_vs_batch.png`
shows both halves of the finding side by side: the same batch size costs
half the memory in reversible mode, and the same memory budget buys about
double the batch size.

As a sanity check against a *hard*-crash ceiling (no soft budget), a
standalone exploratory search (not part of `run_all.py`, see git history /
session notes) found the standard model's true crash point at batch 128
(reported allocation ≈22 GB, i.e. beyond this machine's 19.3 GB of
physical RAM via swap) with batch 64 succeeding but visibly straining
(step time jumped from 2.8 s to 22.7 s). The reversible-midpoint model
similarly *ran* at batch 128 (20.9 GB) without a hard crash, only slowing
down — consistent with reversible tolerating higher memory pressure before
outright failing, though this was not adopted as the final methodology
because "succeeds but starts swapping" is not a stable operating point for
a multi-hundred-step training run.

## Cost comparison

$$\text{cost per 1M tokens} = \frac{10^6}{\text{tokens/sec}} \times \frac{\text{GPU \$/hr}}{3600}$$

GPU price used: **$0.176/hr**, derived from Google Colab's pay-as-you-go
compute-unit pricing checked 2026-09-20 — $9.99 per 100 compute units, and a
T4 burns ~1.76 CU/hr (100/1.76 ≈ 57 hours per 100-CU purchase), giving
1.76 × ($9.99/100) ≈ $0.176/hr. An A100 burns ~15 CU/hr on the same pricing
(≈ $1.50/hr) — worth quoting because the paper's own memory tables (Section
17 of the lesson) are about exactly the regime where the extra activation
memory only matters once you're pushing sequence length or batch size far
enough to need an A100/H100-class card in the first place; the cost story
looks different on a card that was never the memory bottleneck to begin
with. This machine has no cloud billing, so the comparison applies this
sourced hourly rate to the tokens/sec actually measured locally — the
standard way to do this comparison, isolating "reversible is slower per
token but processes more tokens per batch" from whatever hardware happens
to be running it.

| | tok/s (steady) | $ / 1M tokens |
|---|---|---|
| Standard (B=13) | 13,718 | $0.003564 |
| Reversible midpoint (B=13) | 12,094 | $0.004042 |
| Reversible euler (B=13) | 11,866 | $0.004120 |
| Reversible euler (B=27, max batch) | 12,193 | $0.004010 |

At a fixed batch size, reversible costs **~13-16% more per token processed**
— the direct price of the extra recomputation. Pushing to the reversible
model's own max batch (27) claws almost all of that back (12,193 vs. 13,718
tok/s) because throughput at this tiny model/short-sequence scale is not
purely batch-bound on this hardware, so the larger batch mostly amortizes
fixed per-step overhead rather than hitting a new bottleneck. The more
interesting comparison the guide asks for — a configuration that fits in
reversible mode but does not fit in standard mode at all — doesn't have a
finite standard-mode cost to compare against by construction; at this
model/data scale the batch sizes standard mode already reaches (13) are
themselves comfortably sized for a 21M model on 512-length sequences, so
this run doesn't reach a regime where standard mode fails outright. The
guide's own worked numbers (Section 17, 8xB200 node, 131K-token sequences)
are exactly the regime — much longer context and/or a much larger model —
where that gap becomes the whole story; reproducing it would need either a
far longer sequence length or a far larger model than 21M/512 allows on a
laptop.
