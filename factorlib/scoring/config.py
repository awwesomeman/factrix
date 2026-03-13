"""Scoring configuration — dimension weights and metric parameters."""

# ---------------------------------------------------------------------------
# Individual Stock Factor Config (Selection-facet, serializable string keys)
# ---------------------------------------------------------------------------

SCORING_CONFIG = {
    "Alpha": {
        "weight": 0.30,
        "metrics": {
            "Rank_IC": {"weight": 0.4},
            "IC_IR": {"weight": 0.2},
            "Long_Only_Alpha": {"weight": 0.4, "q_top": 0.2},
        },
    },
    "Robustness": {
        "weight": 0.35,
        "metrics": {
            "Internal_OOS_Decay": {"weight": 0.6, "oos_ratio": 0.2},
            "IC_Stability": {"weight": 0.4},
        },
    },
    "Risk": {
        "weight": 0.25,
        "metrics": {
            "Turnover": {"weight": 0.5, "min_threshold": 20},
            "MDD": {"weight": 0.5},
        },
    },
    "Novelty": {
        "weight": 0.10,
        "metrics": {
            "Orthogonality": {"weight": 1.0},
        },
    },
}
