# The Sine Wave and the Epicycle

Two Colab notebooks on one question: **when can a machine learning system find a rule, rather than fit
one?**

Both are self-contained, CPU-runnable, and written so that every claim in them is a number you can
re-run rather than a statement you have to take on trust.

---

## Notebook 1 — `Extrapolation_Rule_Learning_Benchmark.ipynb`

A benchmark of 19 learners across 9 rule-learning splits, measuring interpolation against extrapolation,
parameter count against description length, and priors against dense sampling. It replicates Figure 2 of
Hasson, Nastase & Goldstein's *Direct Fit to Nature* (Neuron, 2020), then puts frugal and symbolic
learners on the same axes, and finishes by asking open-weight language models to induce rules they
cannot have memorised.

**The story and the full results are in the essay, not here:**
[The Sine Wave and the Epicycle](https://aslansatarydizaji.substack.com/p/the-sine-wave-and-the-epicycle)

The one-line version, because the second notebook is a reply to it: *the best method is the smallest
hypothesis space that still contains the rule* — not the smallest model, and not the largest.

---

## Notebook 2 — `Rule_Induction_as_Planning.ipynb`

If search over a hypothesis space is what wins, build that and measure which part of it is doing the
work. This notebook trains a **0.6M-parameter transformer from scratch** to propose programs in a small
DSL, and searches that space with four planners under an exact execution check. It then grows the DSL by
DreamCoder-style library learning and asks whether abstraction buys anything.

### The design

Rule induction is written as an MDP: the state is a partial program, the actions are grammar-legal
tokens, and the terminal reward is whether the executor reproduces all eight examples. Four planners
search that tree — greedy rollout (the language-model condition), beam, best-first, and PUCT.

Two design choices carry most of the result:

- **The network proposes a sketch; the executor solves the constants.** `+ * ? a ?` rather than
  `+ * N 8 E a N 5 5 E`. This factorisation is what makes held-out constants reachable at all.
- **Verification, not confidence.** A program is returned only if it reproduces every example exactly.
  Because the output is then *executed*, far-range accuracy is 1.0 by construction — extrapolation stops
  being a statistical hope and becomes a property of the representation.

Evaluation is on four axes of novelty, all held out by construction: unseen rule *semantics*, unseen
*constants* (training uses 1–19, testing 20–99), one extra level of composition *depth*, and the six
rules from the first notebook's LLM probe, verbatim.

### What it found

**Search is what makes it right; the network is a speed-up.** The greedy condition — emit a program,
don't check it — recovers the rule on 4–10% of tasks. Every planner with an execution check lands
between 28% and 48%, and on in-distribution tasks there is *no gap at all* between "fits the eight
examples" and "is the true rule".

**Representation beats prior.** A 2×2 over {sketch, digits} × {learned prior, uniform prior} gives a
main effect of **+0.15 for the representation** and **+0.08 for the learned prior**. The two compose on
held-out semantics (0.25 alone, 0.20 alone, 0.50 together) and the prior is worth exactly nothing on
held-out constants, where the network has no opinion worth having.

**Learning buys speed, not correctness.** Neural beam reaches a higher solve rate than uniform
enumeration while expanding a median of **203 nodes against 2059**.

**The head-to-head against the first notebook's LLM probe is a loss, and that is the useful part.** The
system solves all three canonical rules ($a+b$, $a\cdot b$, $a^2$) where Qwen2.5-0.5B scored 0.11 — but
returns nothing on the three freshly-generated rules or the two non-polynomial ones inside a 2000-node
budget. Meanwhile the six-term polynomial dictionary from notebook 1, with no learning at all, solves
every polynomial rule instantly. Generality is paid for in search, and at Colab budgets the bill comes
due. H2 again, now aimed at my own design.

### Library learning (§8–10)

Five iterations of wake → abstract → re-estimate, growing the DSL from 0 to 15 invented primitives:

| | iteration 0 | iteration 5 |
|---|---:|---:|
| median nodes to solution | 331 | **154** |
| depth searched | 0.80 | 1.12 |
| depth of the rules found (base language) | 0.80 | **1.65** |
| tasks **fitted** | 0.34 | **0.41** |
| tasks **correctly induced** | 0.28 | **0.26** |
| Part I rules solved | 3/8 | 3/8 |

The mechanism works: the planner ends up reaching rules 1.65 levels deep while descending only 1.12, at
less than half the node cost. But the last two rows are the finding. **A bigger language is a better
fitter and a slightly worse inducer** — the gap between fitting eight observations and recovering the
rule more than doubles (0.07 → 0.16), because abstraction increases the density of *short* programs and
short programs are what the search returns first. Compression and identification are not the same
objective.

And nothing flipped on the Part I rules, for a reason visible in the printed library: **not one of the
fifteen invented primitives contains a multiplication.** They are all division, modulo and additive
idioms, because those are the shapes that recur in the programs the system could already find. §9.3
hands it the two primitives it actually needed and the score moves **3/8 → 5/8** at the same budget,
which localises the failure to discovery rather than machinery:

> Library learning abstracts over the solutions you already have, so it cannot invent the abstraction
> you need in order to solve what you currently cannot.

That is a structural limit on the phrase "produces new knowledge". The system compresses its own
competence and reaches modestly further next iteration. It does not bootstrap across a gap. §10.3 lists
what to do about it, starting with a difficulty-graded curriculum for the wake phase — one changed line,
and the highest-value experiment left.

---

## Running them

Both notebooks run top to bottom in Colab with no setup and no GPU.

| notebook | CPU runtime | GPU |
|---|---|---|
| `Extrapolation_Rule_Learning_Benchmark.ipynb` | ~40 min at `QUICK = False` | not needed |
| `Rule_Induction_as_Planning.ipynb` | ~45 min at `QUICK = False` | optional, scales the training budget up |

Set `QUICK = True` in the setup cell of either for a fast pass (10–20 min) with the same qualitative
results. Notebook 1's §I downloads small open-weight models from Hugging Face and is the only section
that benefits from a GPU. All results are written to `results/*.csv`.

Dependencies are whatever Colab ships — `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`,
`torch` — plus `gplearn` (notebook 1, `QUICK = False`) and `transformers` (notebook 1, §I), both
pip-installed by the notebooks themselves.

## Contents

```
Extrapolation_Rule_Learning_Benchmark.ipynb   19 learners x 9 splits; the essay's source
Rule_Induction_as_Planning.ipynb              neural-guided program synthesis + library learning
README.md
LICENSE
```

## Citing

If you use either notebook, please link to this repository and to the essay. Both are released under
the licence in `LICENSE`.
