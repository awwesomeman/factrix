---
title: factrix.inference
---

::: factrix.inference

## Direct `compute` result

Every series-mean member has the same direct-call shape:

```python
from datetime import date, timedelta

import numpy as np
import polars as pl
from factrix.inference import NEWEY_WEST

data = pl.DataFrame(
    {
        "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(60)],
        "value": np.linspace(-0.1, 0.2, 60),
    }
)
result = NEWEY_WEST.compute(
    data,
    value_col="value",
    overlap_periods=5,
    alternative="greater",
)
assert result.alternative == "greater"
```

Direct calls require a non-null Polars `Date` or `Datetime` column named
`date`, exactly one observation per date, a numeric column named by `value_col`,
and a positive integer `overlap_periods`. Missing or non-numeric value columns
raise `UserInputError`. If the source contains multiple observations in one
period, aggregate them explicitly before calling `compute`; factrix does not
guess whether `mean`, `last`, or another reduction represents the series.

`InferenceResult.p_value` is always paired with
`InferenceResult.alternative`, which is the validated `"two-sided"`,
`"greater"`, or `"less"` tail passed to the statistical kernel. Read these
two fields together; metadata contains method-specific diagnostics such as
bandwidth, effective degrees of freedom, or bootstrap seed.
