"""Per-factor-type scoring configuration (4-dimension model).

Each factor type defines:
- routing: dimension weight split (signal/performance/robustness/efficiency)
- per-dimension metric list with optional params

Metric params (q_top, oos_ratio, min_threshold) are per-metric config,
not importance weights — adaptive_weight handles weighting via t-stat.

# WHY: 從五維度（alpha/persistence/efficiency/risk/orthogonality）重構為四維度：
# 1. Predictability：純截面預測力（IC_IR 已包含 Rank_IC 的資訊，移除冗餘）
# 2. Profitability：組合層面 P&L 品質（Long_Alpha + MDD 同屬一條 NAV 曲線的正反面）
# 3. Robustness：穩健性（OOS_Decay + IC_Stability + Hit_Rate，後者從 Alpha 移入）
# 4. Tradability：可交易性（Turnover，未來可加 Capacity）
# 砍掉 Orthogonality — 無 factor zoo 時為死權重，等有了再作為 post-filter 加回

# TODO: 待實證調校 — routing 權重為合理預設值，後續根據實證資料優化
"""

# WHY: adaptive_weight sigmoid 參數預設值；tau=2.0 對應傳統 t>2 顯著性門檻，
# k=2.0 控制 sigmoid 斜率。Harvey et al. (2016) 建議 K 未知時提高至 tau=3.0。
DEFAULT_ADAPTIVE_TAU: float = 2.0
DEFAULT_ADAPTIVE_K: float = 2.0

# WHY: 被 VETO 的因子仍保留此比例的原始分數供橫向排名，
# 而非直接歸零，避免微小 VETO 導致排名斷崖
DEFAULT_VETO_PENALTY: float = 0.2

DIMENSIONS = ("predictability", "profitability", "robustness", "tradability")

FACTOR_CONFIGS = {
    "individual_stock": {
        "routing": {
            "predictability": 0.40, "profitability": 0.25,
            "robustness": 0.25, "tradability": 0.10,
        },
        "predictability": {
            "IC_IR": {},
            "Monotonicity": {"n_groups": 5},
        },
        "profitability": {
            "Long_Alpha": {"q_top": 0.2},
            "MDD": {},
        },
        "robustness": {
            "OOS_Decay": {"oos_ratio": 0.2},
            "IC_Stability": {},
            "Hit_Rate": {},
        },
        "tradability": {
            "Turnover": {"min_threshold": 20},
        },
    },
    "group_region": {
        "routing": {
            "predictability": 0.35, "profitability": 0.25,
            "robustness": 0.25, "tradability": 0.15,
        },
        "predictability": {
            "IC_IR": {},
        },
        "profitability": {
            "MDD": {},
        },
        "robustness": {
            "Hit_Rate": {},
            "Cross_Consistency": {},
        },
        "tradability": {
            "Turnover": {"min_threshold": 20},
        },
    },
    "global_macro": {
        "routing": {
            "predictability": 0.35, "profitability": 0.30,
            "robustness": 0.20, "tradability": 0.15,
        },
        "predictability": {
            "IC_IR": {},
        },
        "profitability": {
            "MDD": {},
            "Profit_Factor": {},
        },
        "robustness": {
            "Hit_Rate": {},
        },
        "tradability": {
            "Turnover": {"min_threshold": 20},
        },
    },
    "event_signal": {
        # WHY: 事件訊號無 z-score factor 欄位，Turnover/MDD/OOS_Decay 不適用。
        # 改用純事件指標評估，efficiency 預留但暫無指標。
        "routing": {
            "predictability": 0.50, "profitability": 0.20,
            "robustness": 0.25, "tradability": 0.0,
        },
        "predictability": {
            "Event_CAAR": {"min_threshold": 20},
            "Event_KS": {},
            "Event_CAR_Dispersion": {},
        },
        "profitability": {
            "Profit_Factor": {},
            "Event_Skewness": {},
        },
        "robustness": {
            "Event_Decay": {},
            "Event_Stability": {},
            "Event_Hit_Rate": {},
        },
        "tradability": {},
    },
}
