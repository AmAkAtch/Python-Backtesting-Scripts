"""
MULTI-FOLD CONSISTENCY CHECK -- standalone companion to main.py
=================================================================
Does NOT run Bayesian optimization and does NOT modify anything main.py
saves. It loads whatever champion is already saved (best_params_v4.json,
or current_is_champion_v4.json if no fully OOS-validated champion exists
yet) and asks one question: does this exact strategy hold up across
several separate multi-year windows, or did it only work in one lucky
stretch?

HOW TO USE
  python walk_forward_multifold.py
  (Optional) edit N_FOLDS below -- default splits the whole dataset into
  4 non-overlapping windows of a few years each.

WHAT IT PRINTS
  1. HEADLINE -- one continuous run across the WHOLE dataset, Rs.0 start.
     This is the plain "final ROI over ~N years" number for a quick gut
     check. It is NOT a validation result (see main.py's docstring on why
     a full-data number is expected to look good).
  2. FOLD TABLE -- N_FOLDS non-overlapping windows, each evaluated
     independently from Rs.0 (so a bad early fold can't hide behind gains
     from a good later one). Each fold reports its own ROI/IRR/Sharpe/
     MaxDD/win rate and whether it individually beat the benchmark.
  3. SUMMARY -- how many folds beat the benchmark, and the mean/min/max
     fold ROI, so you can see consistency at a glance.

This is a REPORT, not a gate -- nothing here saves, rejects, or feeds back
into main.py's search. It reuses main.py's data loading and evaluate_params
directly (same import), so results are guaranteed apples-to-apples with
whatever main.py printed for the same champion.
"""

from main import (
    fetch_dynamic_universe, prepare_matrix_data, clear_ma_cache,
    build_sip_trigger_mask, evaluate_params, load_previous_winner,
    BEST_PARAMS_FILE, INTERMEDIATE_FILE,
)

N_FOLDS = 4   # non-overlapping windows spanning the full dataset


def load_champion():
    _, _, params = load_previous_winner(BEST_PARAMS_FILE)
    source = BEST_PARAMS_FILE
    if params is None:
        _, _, params = load_previous_winner(INTERMEDIATE_FILE)
        source = INTERMEDIATE_FILE
    if params is None:
        raise SystemExit(
            "No saved champion found (best_params_v4.json or "
            "current_is_champion_v4.json) -- run main.py first."
        )
    print(f"Loaded champion from {source}")
    return params


def run_window(label, params, opens, closes, atr, adx, sip_trigger, index_closes,
               is_div_stock, eligible_mask, start_day, end_day):
    """Evaluates `params` on [start_day, end_day) starting from Rs.0.
    Prints one summary line and returns the metrics dict (or None if the
    window had too little data to produce a meaningful result)."""
    score, m = evaluate_params(
        params, opens, closes, atr, adx, sip_trigger, index_closes,
        is_div_stock, eligible_mask,
        start_day=start_day, end_day=end_day, is_oos=False, starting_wealth=0.0
    )
    years = (end_day - start_day) / 252.0
    if not m or m.get('trades', 0) == 0:
        print(f"  {label:<26} {years:4.1f}y | no trades in this window")
        return None

    wt, lt = m.get('winning_trades', 0), m.get('losing_trades', 0)
    wr = wt / (wt + lt) * 100 if (wt + lt) > 0 else 0.0
    beat_bench = "YES" if m['roi'] > m['bench_roi'] else "no"
    print(f"  {label:<26} {years:4.1f}y | ROI {m['roi']*100:7.1f}% | "
          f"IRR {m['annual_return']*100:6.1f}% | Sharpe {m['sharpe']:5.2f} | "
          f"MaxDD {abs(m['max_dd'])*100:5.1f}% | WinRate {wr:5.1f}% | "
          f"Trades {m['trades']:4d} | Bench {m['bench_roi']*100:6.1f}% | Beat bench: {beat_bench}")
    return m


def main():
    params = load_champion()

    tickers = fetch_dynamic_universe()
    (opens, closes, atr, adx, months, index_closes,
     is_div_stock, stock_names, eligible_mask) = prepare_matrix_data(tickers)
    clear_ma_cache()
    sip_trigger = build_sip_trigger_mask(months, fixed_offset=0)

    n_days = closes.shape[0]
    print(f"\nMatrix: {n_days} days x {closes.shape[1]} stocks (~{n_days/252:.1f} years total)")

    # ---- HEADLINE: one continuous run across the whole dataset ----
    print("\n" + "=" * 78)
    print("HEADLINE -- continuous run, Rs.0 start, full dataset")
    print("=" * 78)
    headline_m = run_window("Full period", params, opens, closes, atr, adx,
                            sip_trigger, index_closes, is_div_stock, eligible_mask,
                            1, n_days - 1)

    # ---- FOLDS: non-overlapping windows, each independent from Rs.0 ----
    print("\n" + "=" * 78)
    print(f"FOLD-BY-FOLD CONSISTENCY ({N_FOLDS} non-overlapping windows)")
    print("=" * 78)
    edges = [max(1, int(round(i * (n_days - 1) / N_FOLDS))) for i in range(N_FOLDS + 1)]
    fold_metrics = []
    for i in range(N_FOLDS):
        s, e = edges[i], edges[i + 1]
        m = run_window(f"Fold {i+1} (days {s}-{e})", params, opens, closes, atr, adx,
                       sip_trigger, index_closes, is_div_stock, eligible_mask, s, e)
        if m:
            fold_metrics.append(m)

    # ---- SUMMARY ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if fold_metrics:
        rois = [m['roi'] * 100 for m in fold_metrics]
        beat_count = sum(1 for m in fold_metrics if m['roi'] > m['bench_roi'])
        print(f"  Folds with usable data:        {len(fold_metrics)}/{N_FOLDS}")
        print(f"  Folds that beat the benchmark: {beat_count}/{len(fold_metrics)}")
        print(f"  Fold ROI  mean / min / max:    {sum(rois)/len(rois):.1f}% / {min(rois):.1f}% / {max(rois):.1f}%")
    else:
        print("  No fold had enough data to evaluate -- try lowering N_FOLDS.")

    if headline_m:
        print(f"\n  HEADLINE full-period ROI: {headline_m['roi']*100:.1f}%  "
              f"(IRR {headline_m['annual_return']*100:.1f}%/yr over ~{n_days/252:.1f} years)")

    print("\nThis is a consistency REPORT, not a pass/fail gate. Read the fold table to see")
    print("whether performance holds up across different multi-year stretches, or was")
    print("concentrated in just one of them. Read the headline for plain magnitude only.")


if __name__ == "__main__":
    main()
