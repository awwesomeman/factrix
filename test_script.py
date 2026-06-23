import factrix as fx
from factrix.preprocess import compute_forward_return
from factrix.metrics import caar, event_quality, mfe, corrado_rank

try:
    print("--- 1. Generating single-asset event data ---")
    # Generate single asset data with sparse events
    raw = fx.datasets.make_event_panel(n_assets=1, n_dates=500, seed=2024, p_event=0.05)
    data = compute_forward_return(raw, forward_periods=5)

    print("\n--- 2. Inspecting data ---")
    info = fx.inspect_data(data, factor_cols=["factor"])
    print(f"Structure: {info.properties.structure}")
    print(f"Usable metrics: {[m.name for m in info.metrics.usable]}")

    print("\n--- 3. Evaluating single asset event ---")
    # Assuming caar and others are usable. Let's run evaluate.
    metrics_to_run = {}
    for m in info.metrics.usable:
        # Construct instance. Might need to guess parameters or use defaults.
        # Let's dynamically import the class from fx.metrics and instantiate it.
        cls = getattr(fx.metrics, m.name, None)
        if cls is not None:
            try:
                metrics_to_run[m.name] = cls()
            except Exception as e:
                print(f"Could not instantiate {m.name}: {e}")

    results = fx.evaluate(
        data,
        metrics=metrics_to_run,
        factor_cols=["factor"],
        forward_periods=5,
    )
    res = results["factor"]
    print("Metrics evaluated successfully.")
    for name, m_res in res.metrics.items():
        print(f"  {name}: value={m_res.value}, p_value={m_res.p_value}")

    print("\n--- 4. Testing by_slice ---")
    # Add a dummy grouping column
    import polars as pl
    data = data.with_columns(
        pl.when(pl.col("date").dt.year() < 2012).then(pl.lit("A")).otherwise(pl.lit("B")).alias("group")
    )
    slice_results = fx.by_slice(data, metrics_to_run["caar"], by="group", factor_col="factor")
    for group_name, res in slice_results.items():
        print(f"Slice {group_name}: CAAR value={res.metrics['caar'].value}")

    print("\n--- 5. Testing slice_pairwise_test & slice_joint_test ---")
    print("Trying slice_pairwise_test...")
    pairwise_res = fx.slice_pairwise_test(slice_results, metric="caar")
    print(pairwise_res)

    print("Trying slice_joint_test...")
    joint_res = fx.slice_joint_test(slice_results, metric="caar")
    print(joint_res)

    print("\n--- 6. Testing multi_factor ---")
    # Just to test compare
    print("Trying compare()...")
    comp_df = fx.compare(list(slice_results.values()), metrics=["caar"])
    print(comp_df)

except Exception as e:
    import traceback
    traceback.print_exc()
