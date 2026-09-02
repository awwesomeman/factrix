# factrix — project conventions for agents

## Period grid, not calendar (first-order principle)

factrix never reads the calendar. `date` is an ordering/alignment key only;
every horizon, window, lag, stride and sample floor is a count of **periods on
the panel's own distinct-date grid** — never calendar time, never row position
within an asset. The evaluation grid may be unevenly spaced in periods; nothing
may assume a constant stride or a bar frequency.

Wording rule for code, docstrings, warnings, docs, issues and PRs: say
**periods**. Never days / trading days / weeks / month-ends in a contract-level
statement; concrete cadences appear only in explicitly labelled examples.
Source of truth: [Period grid, not calendar](docs/development/architecture.md#period-grid-not-calendar).

## Report what ran, not what was asked for

A reported value is the quantity the code **actually used**, never the one it
resolved, requested or would have used. A bandwidth resolved against one sample
and then clipped by the kernel is reported at the clipped value; the same holds
for any lag, window, floor or count that a callee may narrow.

Rules for metadata and the warnings that describe it:

- A message naming a metadata key must print the value **under that key**. If
  the count the screen reads is not the key the message names, name the key it
  reads — a reader who looks up the named key has to reproduce the comparison
  the message states.
- Warning text may not contradict the metadata emitted beside it. When a screen
  is deliberately calibrated on a wider quantity than the one applied, say both.
- Where a branch cannot satisfy the contract — a short circuit runs no kernel —
  document the exception explicitly. Do not reuse a sentinel that already means
  something else.

This defect class recurs. #1034, #1035, #1036 and #1039 were one contract, one
key apart; treat that as illustration rather than a closed list.

## A test that cannot fail proves nothing

Before claiming a test guards anything, **run it against the unfixed state and
watch it fail**. A guard that passes on the defect it was written for is worse
than no guard, because it reads as coverage. Ways this has shipped here:

- Asserting the absence of a string that was never present — the real text
  carried markup between the words, so the literal never appeared.
- Splitting one composite invariant into separate `in` checks. Assert the whole
  token; two independent checks stop verifying that the parts belong together.

A repo-wide text assertion also needs a false-positive sweep over every file it
will run against, or it becomes a nuisance the next person deletes.

Pinning tests that pass on an unchanged codebase are legitimate — they fix a
decision against silent drift — but they are guards, not regression proof, and
must be labelled as such. Source of truth:
[Testing rules](docs/development/contributing.md#testing-rules).

## Measure the claim; do not assert it

Quantitative statements in code comments, docstrings, issues, PRs and reviews
are backed by a sweep that was actually run, with the cell counts stated. This
applies hardest to "nothing changes" claims: a p-value, size or effective-df
invariant is established by enumerating the reachable cells, not by reasoning
from the formula alone. Prefer the argument that survives a change in
calibration constants over one that depends on their current values.

Do not take a passing run on trust either: read the `N passed / N failed`
summary line itself. An exit code from a piped `pytest | grep | tail` is the
exit code of the last command in the pipe, not of pytest.

## Check the premise, not the presentation

When a change, a design or a review argues for itself, find the factual claim
the argument rests on and **measure that claim**. A design reverse-engineered
around the case in front of it reads as coherent, and its reasoning is usually
sound *given its premise* — so reading the reasoning more carefully is not a
substitute for testing what it assumes.

Ask what single fact would have to hold for this to be necessary, then check
that fact directly, in the smallest script that settles it. Apply it to your
own arguments first: the claim you are least likely to test is the one you
needed to be true.
