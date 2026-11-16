import optuna, pandas as pd

# 1. Inspect Runner-Up #2 (s2718-315) from existing SQLite DB
study = optuna.load_study(
    study_name="nse_swing_v14.0_pool_lm250_sc100_b15be74b_ae7a36_s2718",
    storage="sqlite:///output/POOL_LM250_SC100/nse_swing_study.db"
)
for tid in (461, 315):
    t = study.trials[tid]
    print(f"\n=== Trial s2718-{tid} (Score: {t.value:.4f}, Closed Trades: {t.user_attrs.get('n_closed')}) ===")
    print(t.params)

# 2. Audit winner_trades.csv for size-neutral Profit Factor and TMPV
df = pd.read_csv("output/POOL_LM250_SC100/winner_trades.csv")
wins = df[df["ret"] > 0.0025]["ret"].sum()
losses = df[df["ret"] < -0.0025]["ret"].abs().sum()
print(f"\nSize-Neutral Profit Factor (using % returns): {wins / losses:.2f}")
print("\nAny TMPV trades around Oct 2025 demerger:")
print(df[df["coin"] == "TMPV"][["coin", "entry_date", "exit_date", "ret", "reason"]])