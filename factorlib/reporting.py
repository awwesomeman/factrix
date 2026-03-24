"""Research script convenience: formatted factor evaluation report."""


def print_report(results: dict, name: str) -> None:
    """Print a formatted report of factor scoring results.

    Args:
        results: Output dict from FactorScorer.compute().
        name: Display name for the factor.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for dim_name, dim_weight in results.get("dimension_weights", {}).items():
        dim = results.get("dimensions", {}).get(dim_name, {})
        metrics = dim.get("metrics", {})
        if not metrics:
            continue

        dim_score = results.get(f"{dim_name}_score", 0)
        print(f"\n  --- {dim_name.capitalize()} (w={dim_weight:.0%}, score={dim_score:.1f}) ---")

        for m, detail in metrics.items():
            s = detail["score"]
            t = detail["t_stat"]
            w = detail["adaptive_w"]
            bar = "\u2588" * int(s / 5) + "\u2591" * (20 - int(s / 5))
            t_str = f"t={t:+.1f}" if t is not None else "     "
            print(f"    {m:25s} {s:6.1f}  {t_str}  w={w:.2f}  {bar}")

    print(f"\n{'\u2500'*60}")
    for p in results.get("penalties", []):
        print(f"  ! {p}")
    label = "Total (penalized)" if results.get("penalties") else "Total"
    print(f"  {label}: {results['total']:.1f}")
    print(f"{'='*60}\n")
