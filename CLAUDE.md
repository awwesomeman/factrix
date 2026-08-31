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
