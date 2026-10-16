import os
import glob
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. CONFIGURATION: SHADOW T1 -> REAL T2 + SLIPPAGE + OPTIMIZER
# ==============================================================================
CSV_FILENAME       = "NAS100ft.r_M5_202505210520_202610161725.csv"
INITIAL_CAPITAL    = 50000.0
RISK_PER_TRADE     = 300.0
MT5_TO_NY_SHIFT    = -7           # Vantage MT5 Server -> America/New_York (-7h)

# --- CME MNQ FUTURES MICROSTRUCTURE & SLIPPAGE ---
POINT_VALUE        = 2.0          # MNQ = $2.00 per index point
TICK_SIZE          = 0.25         # CME MNQ minimum tick = 0.25 pts
COMM_PER_UNIT      = 1.40         # $1.40 round-turn commission per contract
IS_FUTURES         = True

ENTRY_SLIP_TICKS   = 1.0          # 1 tick (0.25 pt) market-order entry slippage on T2 close
SL_SLIP_TICKS      = 2.0          # 2 ticks (0.50 pt) stop-market slippage on T2 stop loss
TP_SLIP_TICKS      = 0.0          # 0 ticks on resting limit Take-Profit orders

# --- STRATEGY CORE PARAMETERS ---
FIB_LEVEL          = 0.618        # Shadow T1 pullback level (0.618 retracement)
T2_FIXED_RR        = 2.0          # Real T2 fixed target (2.0R)
MONTE_CARLO_RUNS   = 2500         # Bootstrap iterations for Drawdown Cone

OUTPUT_TRADES_CSV  = "rorb_shadow_t2_trades.csv"
OUTPUT_REPORT_HTM  = "rorb_quant_suite.html"

# ==============================================================================
# 2. BULLETPROOF LOCAL SSD DATA LOADER (AUTO-LOCATES SCRIPT FOLDER)
# ==============================================================================
def load_mt5_data(target_file: str) -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(script_dir, target_file),
        os.path.join(script_dir, "data", target_file),
        *glob.glob(os.path.join(script_dir, "NAS100*.csv")),
        *glob.glob(os.path.join(script_dir, "*.csv")),
        target_file
    ]
    csv_path = next(
        (p for p in search_paths if os.path.exists(p) and os.path.getsize(p) > 500 and "rorb_" not in os.path.basename(p)),
        None
    )
    if not csv_path:
        raise FileNotFoundError(f"Could not find '{target_file}' inside '{script_dir}'.")

    print(f"[1/5] Loading MT5 dataset from SSD: {csv_path}")
    df_raw = pd.read_csv(csv_path, sep="\t")
    if df_raw.shape[1] == 1:
        df_raw = pd.read_csv(csv_path, sep=None, engine="python")

    df_raw.columns = [c.replace("<", "").replace(">", "").strip().lower() for c in df_raw.columns]

    if "date" in df_raw.columns and "time" in df_raw.columns:
        dt_series = pd.to_datetime(
            df_raw["date"].astype(str) + " " + df_raw["time"].astype(str),
            format="%Y.%m.%d %H:%M:%S",
            errors="coerce"
        )
    else:
        dt_series = pd.to_datetime(df_raw.iloc[:, 0])

    df_raw["ny_time"] = dt_series + pd.Timedelta(hours=MT5_TO_NY_SHIFT)
    df = df_raw.set_index("ny_time").sort_index()
    df = df[["open", "high", "low", "close"]].astype(float).dropna()
    df = df[~df.index.duplicated(keep="first")]

    print(f"[OK] Loaded {len(df):,} clean 5m bars | NY Range: {df.index[0]} -> {df.index[-1]}")
    return df

# ==============================================================================
# 3. SHADOW T1 -> REAL T2 ENGINE WITH SLIPPAGE & MAE/MFE TRACKING
# ==============================================================================
def round_tick(val: float) -> float:
    return np.round(val / TICK_SIZE) * TICK_SIZE

def calc_qty(sl_dist: float) -> float:
    safe_dist = max(sl_dist, TICK_SIZE)
    raw_qty = RISK_PER_TRADE / (safe_dist * POINT_VALUE)
    return max(1.0, np.round(raw_qty)) if IS_FUTURES else np.round(raw_qty, 2)

def run_shadow_t2_engine(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Executing Shadow T1 -> Real T2 Engine (with Slippage & MAE/MFE tracking)...")
    open_arr   = df["open"].to_numpy(dtype=np.float64)
    high_arr   = df["high"].to_numpy(dtype=np.float64)
    low_arr    = df["low"].to_numpy(dtype=np.float64)
    close_arr  = df["close"].to_numpy(dtype=np.float64)
    day_id     = (df.index.year * 10000 + df.index.month * 100 + df.index.day).to_numpy(dtype=np.int64)
    min_of_day = (df.index.hour * 60 + df.index.minute).to_numpy(dtype=np.int64)

    entry_slip_pts = ENTRY_SLIP_TICKS * TICK_SIZE
    sl_slip_pts    = SL_SLIP_TICKS * TICK_SIZE
    tp_slip_pts    = TP_SLIP_TICKS * TICK_SIZE

    n = len(df)
    records = []

    # States:
    # 0 = Building 09:30-09:45 OR
    # 1 = Waiting for initial 5m breakout close
    # 2 = Shadow T1 0.618 Limit Pending
    # 3 = Shadow T1 Active (Monitoring for stop-out at ORL/ORH; if Shadow T1 hits 1R-3R TP or EOD, session done)
    # 4 = Shadow T1 Stopped by Wick -> Waiting for 5m Reversal Close beyond ORL/ORH before 12:00 NY
    # 5 = Real Trade 2 Active (Floats until SL/TP or next day 09:30 NY reset)
    # 6 = Session Done
    state, cur_day = 6, -1
    orh, orl, orw, or_valid = 0.0, 0.0, 0.0, False

    t1_dir, t1_entry, t1_sl, t1_tp, t1_risk, recent_ext = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    t2_dir, t2_theo_entry, t2_fill_entry, t2_sl, t2_tp, t2_risk, t2_qty = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    t2_fill_idx, t2_Entry_min = 0, 0
    t2_max_fav_pts, t2_max_adv_pts = 0.0, 0.0

    def record_t2_exit(exit_idx, theo_exit_px, actual_exit_px, exit_reason):
        gross_no_slip = (theo_exit_px - t2_theo_entry) * t2_dir * t2_qty * POINT_VALUE
        net_no_slip   = gross_no_slip - (COMM_PER_UNIT * t2_qty)

        gross_slip    = (actual_exit_px - t2_fill_entry) * t2_dir * t2_qty * POINT_VALUE
        net_slip      = gross_slip - (COMM_PER_UNIT * t2_qty)
        slip_cost_usd = net_no_slip - net_slip

        realized_r    = net_slip / RISK_PER_TRADE
        mae_r         = -(t2_max_adv_pts / t2_risk)
        mfe_r         = (t2_max_fav_pts / t2_risk)
        dur_mins      = (df.index[exit_idx] - df.index[t2_fill_idx]).total_seconds() / 60.0

        records.append({
            "entry_time":    df.index[t2_fill_idx],
            "exit_time":     df.index[exit_idx],
            "trade_date":    df.index[t2_fill_idx].date(),
            "weekday":       df.index[t2_fill_idx].day_name(),
            "weekday_num":   df.index[t2_fill_idx].weekday(),
            "entry_min":     t2_Entry_min,
            "entry_bucket":  f"{t2_Entry_min // 60:02d}:{(t2_Entry_min % 60 // 15) * 15:02d}",
            "direction":     "LONG" if t2_dir == 1 else "SHORT",
            "or_width_pts":  round(orw, 2),
            "t2_risk_pts":   round(t2_risk, 2),
            "qty":           t2_qty,
            "theo_entry_px": round(t2_theo_entry, 2),
            "fill_entry_px": round(t2_fill_entry, 2),
            "exit_px":       round(actual_exit_px, 2),
            "sl_px":         round(t2_sl, 2),
            "tp_px":         round(t2_tp, 2),
            "exit_reason":   exit_reason,
            "pnl_no_slip":   round(net_no_slip, 2),
            "slip_cost_usd": round(slip_cost_usd, 2),
            "pnl_usd":       round(net_slip, 2),
            "r_multiple":    round(realized_r, 3),
            "mae_r":         round(mae_r, 3),
            "mfe_r":         round(mfe_r, 3),
            "duration_min":  round(max(5.0, dur_mins), 1)
        })

    def trigger_t2(idx, close_px, high_px, low_px, m_od):
        nonlocal t2_dir, t2_theo_entry, t2_fill_entry, t2_sl, t2_tp, t2_risk, t2_qty
        nonlocal t2_fill_idx, t2_Entry_min, t2_max_fav_pts, t2_max_adv_pts, state
        t2_dir         = -t1_dir
        t2_theo_entry  = round_tick(close_px)
        t2_fill_entry  = round_tick(t2_theo_entry + (t2_dir * entry_slip_pts))
        t2_sl          = round_tick(high_px if t2_dir == -1 else low_px)
        # Risk is anchored from theoretical signal close to candle extreme; TP is 2R from signal
        t2_risk        = max((t2_sl - t2_theo_entry) if t2_dir == -1 else (t2_theo_entry - t2_sl), TICK_SIZE)
        t2_tp          = round_tick(t2_theo_entry + (t2_dir * T2_FIXED_RR * t2_risk))
        t2_qty         = calc_qty(t2_risk)
        t2_fill_idx    = idx
        t2_Entry_min   = int(m_od)
        t2_max_fav_pts = 0.0
        t2_max_adv_pts = 0.0
        state          = 5

    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        d, m = day_id[i], min_of_day[i]

        # 09:30-09:45 NY Opening Range
        if 570 <= m < 585:
            if state != 0 or d != cur_day:
                if state == 5:
                    actual_exit = round_tick(o - (t2_dir * entry_slip_pts))
                    record_t2_exit(i, o, actual_exit, "NEXT_OPEN_RESET")
                cur_day, state = d, 0
                orh, orl, or_valid = h, l, False
            else:
                if h > orh: orh = h
                if l < orl: orl = l
            if orh > orl: or_valid = True
            continue

        if state == 0 and m >= 585:
            if or_valid:
                orw, state = orh - orl, 1
            else:
                state = 6

        in_window = (d == cur_day) and (585 <= m <= 715)
        cutoff    = (m >= 715)

        # STATE 1: Initial Breakout Detection
        if state == 1:
            if not in_window or cutoff:
                state = 6
            elif c > orh:
                t1_dir     = 1
                t1_entry   = round_tick(orh - (FIB_LEVEL * orw))
                t1_sl      = round_tick(orl)
                t1_risk    = max(t1_entry - t1_sl, TICK_SIZE)
                recent_ext = h
                t1_tp      = round_tick(t1_entry + min(3.0 * t1_risk, max(1.0 * t1_risk, recent_ext - t1_entry)))
                state      = 2
            elif c < orl:
                t1_dir     = -1
                t1_entry   = round_tick(orl + (FIB_LEVEL * orw))
                t1_sl      = round_tick(orh)
                t1_risk    = max(t1_sl - t1_entry, TICK_SIZE)
                recent_ext = l
                t1_tp      = round_tick(t1_entry - min(3.0 * t1_risk, max(1.0 * t1_risk, t1_entry - recent_ext)))
                state      = 2

        # STATE 2: Shadow T1 Limit Pending
        elif state == 2:
            if not in_window:
                state = 6
            else:
                filled = (l <= t1_entry) if t1_dir == 1 else (h >= t1_entry)
                if filled:
                    hit_sl = (l <= t1_sl) if t1_dir == 1 else (h >= t1_sl)
                    if hit_sl:
                        rev_close = (c < orl) if t1_dir == 1 else (c > orh)
                        if rev_close:
                            trigger_t2(i, c, h, l, m)
                        elif not cutoff:
                            state = 4
                        else:
                            state = 6
                    else:
                        state = 3
                else:
                    if cutoff:
                        state = 6
                    else:
                        if t1_dir == 1:
                            if h > recent_ext: recent_ext = h
                            t1_tp = round_tick(t1_entry + min(3.0 * t1_risk, max(1.0 * t1_risk, recent_ext - t1_entry)))
                        else:
                            if l < recent_ext: recent_ext = l
                            t1_tp = round_tick(t1_entry - min(3.0 * t1_risk, max(1.0 * t1_risk, t1_entry - recent_ext)))

        # STATE 3: Shadow T1 Active (Waiting for Shadow T1 to fail at ORL/ORH)
        elif state == 3:
            hit_sl = (l <= t1_sl) if t1_dir == 1 else (h >= t1_sl)
            hit_tp = (h >= t1_tp) if t1_dir == 1 else (l <= t1_tp)
            if hit_sl:
                rev_close = (c < orl) if t1_dir == 1 else (c > orh)
                if in_window and rev_close:
                    trigger_t2(i, c, h, l, m)
                elif in_window and not cutoff:
                    state = 4
                else:
                    state = 6
            elif hit_tp:
                # Shadow T1 succeeded -> no reversal setup today
                state = 6

        # STATE 4: Shadow T1 Wicked Out -> Waiting for 5m Close Beyond OR
        elif state == 4:
            if not in_window:
                state = 6
            else:
                rev_close = (c < orl) if t1_dir == 1 else (c > orh)
                if rev_close:
                    trigger_t2(i, c, h, l, m)
                elif cutoff:
                    state = 6

        # STATE 5: Real Trade 2 Active (Track MAE/MFE & Check SL/TP)
        elif state == 5:
            # Update intra-trade excursion from filled entry
            if t2_dir == 1:
                fav = max(0.0, h - t2_fill_entry)
                adv = max(0.0, t2_fill_entry - l)
            else:
                fav = max(0.0, t2_fill_entry - l)
                adv = max(0.0, h - t2_fill_entry)
            if fav > t2_max_fav_pts: t2_max_fav_pts = fav
            if adv > t2_max_adv_pts: t2_max_adv_pts = adv

            hit_sl = (l <= t2_sl) if t2_dir == 1 else (h >= t2_sl)
            hit_tp = (h >= t2_tp) if t2_dir == 1 else (l <= t2_tp)

            if hit_sl:
                t2_max_adv_pts = max(t2_max_adv_pts, abs(t2_fill_entry - t2_sl) + sl_slip_pts)
                actual_sl_px   = round_tick(t2_sl - (t2_dir * sl_slip_pts))
                record_t2_exit(i, t2_sl, actual_sl_px, "SL")
                state = 6
            elif hit_tp:
                t2_max_fav_pts = max(t2_max_fav_pts, abs(t2_tp - t2_fill_entry))
                actual_tp_px   = round_tick(t2_tp - (t2_dir * tp_slip_pts))
                record_t2_exit(i, t2_tp, actual_tp_px, "TP")
                state = 6

    return pd.DataFrame(records)

# ==============================================================================
# 4. RULE OPTIMIZER & TEARSHEET GENERATOR
# ==============================================================================
def summarize_segment(df_sub: pd.DataFrame, name: str, pnl_col: str = "pnl_usd") -> dict:
    if len(df_sub) == 0:
        return {"Configuration": name, "Trades": 0}
    pnl   = df_sub[pnl_col]
    wins  = pnl[pnl > 0]
    loss  = pnl[pnl <= 0]
    wr    = (len(wins) / len(pnl)) * 100.0
    pf    = wins.sum() / abs(loss.sum()) if len(loss) > 0 and abs(loss.sum()) > 0 else np.inf
    daily = df_sub.groupby("trade_date")[pnl_col].sum()
    sh    = (daily.mean() / daily.std()) * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0
    eq    = INITIAL_CAPITAL + pnl.cumsum()
    dd    = (eq - np.maximum(INITIAL_CAPITAL, eq.cummax())).min()
    return {
        "Configuration":  name,
        "Trades":         len(df_sub),
        "Win Rate (%)":   f"{wr:.2f}%",
        "Profit Factor":  f"{pf:.2f}",
        "Net PnL ($)":    f"${pnl.sum():,.2f}",
        "Slippage ($)":   f"-${df_sub['slip_cost_usd'].sum():,.2f}" if pnl_col == "pnl_usd" else "$0.00",
        "Expectancy ($)": f"${pnl.mean():,.2f}",
        "Expectancy (R)": f"{(pnl.mean() / RISK_PER_TRADE):.3f} R",
        "Max DD ($)":     f"${dd:,.2f}",
        "Sharpe (Ann.)":  f"{sh:.2f}"
    }

def optimize_strategy_rules(tr_df: pd.DataFrame):
    print("[3/5] Running Multi-Dimensional Rule Optimization (Entry Window, OR Width, Weekday)...")
    # 1. Entry Cutoff Scan (10:30, 10:45, 11:00, 11:15, 11:30, 11:45, 12:00 NY)
    best_cutoff, best_cutoff_pf = 715, 0.0
    for m_cut in [630, 645, 660, 675, 690, 705, 715]:
        sub = tr_df[tr_df["entry_min"] <= m_cut]
        if len(sub) >= 25:
            w = sub[sub["pnl_usd"] > 0]["pnl_usd"].sum()
            l = abs(sub[sub["pnl_usd"] <= 0]["pnl_usd"].sum())
            pf = w / l if l > 0 else 0
            if pf > best_cutoff_pf:
                best_cutoff_pf, best_cutoff = pf, m_cut

    # 2. Opening Range Width Filter Scan (Remove extreme top/bottom range outliers)
    q10, q85 = tr_df["or_width_pts"].quantile(0.10), tr_df["or_width_pts"].quantile(0.85)

    # 3. Identify Worst Weekday (if negative expectancy)
    day_perf = tr_df.groupby("weekday")["pnl_usd"].sum()
    worst_days = day_perf[day_perf < 0].index.tolist()

    # Apply Combined Filter
    opt_mask = (tr_df["entry_min"] <= best_cutoff) & (tr_df["or_width_pts"] <= q85)
    if len(worst_days) > 0:
        opt_mask = opt_mask & (~tr_df["weekday"].isin(worst_days))

    opt_df = tr_df[opt_mask].copy()
    if len(opt_df) < 15:
        opt_df = tr_df[tr_df["or_width_pts"] <= q85].copy()

    rule_desc = (
        f"Entry <= {best_cutoff // 60:02d}:{best_cutoff % 60:02d} NY | "
        f"OR Width <= {q85:.1f} pts" +
        (f" | Skip {', '.join(worst_days)}" if worst_days else "")
    )
    return opt_df, rule_desc

# ==============================================================================
# 5. BUILD ALL 5 VISUALIZATION MODULES + FULL CALENDAR HEATMAP
# ==============================================================================
def build_quant_dashboard(tr_df: pd.DataFrame, opt_df: pd.DataFrame, rule_desc: str):
    print(f"[4/5] Building 6-Panel Institutional Visualization Suite -> '{OUTPUT_REPORT_HTM}'...")

    tr_df = tr_df.copy()
    tr_df["cum_no_slip"] = tr_df["pnl_no_slip"].cumsum()
    tr_df["cum_slip"]    = tr_df["pnl_usd"].cumsum()
    eq_slip              = INITIAL_CAPITAL + tr_df["cum_slip"]
    tr_df["dd_slip"]     = eq_slip - np.maximum(INITIAL_CAPITAL, eq_slip.cummax())

    opt_df = opt_df.copy()
    opt_df["cum_opt"]    = opt_df["pnl_usd"].cumsum()
    eq_opt               = INITIAL_CAPITAL + opt_df["cum_opt"]
    opt_df["dd_opt"]     = eq_opt - np.maximum(INITIAL_CAPITAL, eq_opt.cummax())

    # --- MODULE A: FULL DAILY CALENDAR VIEW HEATMAP (GitHub-Style Week x Weekday Grid) ---
    start_dt = pd.to_datetime(tr_df["trade_date"].min())
    end_dt   = pd.to_datetime(tr_df["trade_date"].max())
    # Align start to Monday
    cal_start = start_dt - pd.Timedelta(days=start_dt.weekday())
    all_days  = pd.date_range(cal_start, end_dt, freq="B")  # Business days Mon-Fri

    daily_map = tr_df.groupby(pd.to_datetime(tr_df["trade_date"]))["pnl_usd"].sum()
    cal_df    = pd.DataFrame({"date": all_days})
    cal_df["pnl"]     = cal_df["date"].map(daily_map).fillna(0.0)
    cal_df["weekday"] = cal_df["date"].dt.weekday
    cal_df["week_id"] = ((cal_df["date"] - cal_start).dt.days // 7)

    n_weeks   = cal_df["week_id"].max() + 1
    z_cal     = np.full((5, n_weeks), np.nan)
    text_cal  = np.full((5, n_weeks), "", dtype=object)
    week_lbls = []

    for w in range(n_weeks):
        w_dates = cal_df[cal_df["week_id"] == w]["date"]
        week_lbls.append(w_dates.min().strftime("%Y-%m-%d") if len(w_dates) > 0 else f"W{w}")

    for _, row in cal_df.iterrows():
        wd, wk, val = int(row["weekday"]), int(row["week_id"]), float(row["pnl"])
        if wd < 5:
            z_cal[wd, wk]    = val
            text_cal[wd, wk] = f"{row['date'].strftime('%Y-%m-%d')} ({row['date'].strftime('%a')})<br>PnL: ${val:,.2f}"

    # --- MODULE B: ENTRY-TIME WINDOW EFFICIENCY (15-Min Buckets) ---
    bucket_grp = tr_df.groupby("entry_bucket").agg(
        net_pnl=("pnl_usd", "sum"),
        trades=("pnl_usd", "count"),
        win_rate=("pnl_usd", lambda s: (s > 0).mean() * 100.0)
    ).sort_index()
    b_colors = ["#00E676" if v >= 0 else "#FF5252" for v in bucket_grp["net_pnl"]]

    # --- MODULE C: MAE / MFE SCATTER PLOT ---
    wins_df = tr_df[tr_df["pnl_usd"] > 0]
    loss_df = tr_df[tr_df["pnl_usd"] <= 0]

    # --- MODULE D: MONTE CARLO DRAWDOWN CONE & EQUITY FAN ---
    np.random.seed(42)
    pnl_array = tr_df["pnl_usd"].to_numpy()
    n_trades  = len(pnl_array)
    mc_matrix = np.random.choice(pnl_array, size=(MONTE_CARLO_RUNS, n_trades), replace=True)
    mc_cum    = np.cumsum(mc_matrix, axis=1)
    mc_peaks  = np.maximum(0.0, np.maximum.accumulate(mc_cum, axis=1))
    mc_dds    = mc_cum - mc_peaks  # Underwater drawdown paths

    dd_p50 = np.percentile(mc_dds, 50, axis=0)
    dd_p05 = np.percentile(mc_dds, 5, axis=0)   # 95th percentile worst DD
    dd_p01 = np.percentile(mc_dds, 1, axis=0)   # 99th percentile tail DD

    # --- CONSTRUCT 6-ROW INTERACTIVE DASHBOARD ---
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.055,
        row_heights=[0.22, 0.14, 0.16, 0.16, 0.16, 0.16],
        subplot_titles=(
            "1. Shadow T1 -> Real T2 Cumulative PnL ($): Zero-Slippage vs. Slippage-Adjusted vs. Rule-Optimized",
            "2. Underwater Drawdown ($): Baseline T2 (w/ Slippage) vs. Rule-Optimized T2",
            "3. Full Daily Calendar PnL Heatmap (Mon-Fri Grid Across All Weeks)",
            "4. Entry-Time Window Efficiency (15-Minute NY Open Buckets: 09:45 - 12:00)",
            "5. Intra-Trade MAE vs. MFE Excursion Scatter (in R-Multiples) & Trade Duration Bubble Profile",
            f"6. Monte Carlo Underwater Drawdown Cone ({MONTE_CARLO_RUNS:,} Resampled Paths: Median, 95% & 99% Tail Risk)"
        )
    )

    # Row 1: Equity Curves
    fig.add_trace(go.Scatter(
        x=tr_df["exit_time"], y=tr_df["cum_no_slip"], mode="lines",
        name="Shadow T2 (Zero Slippage)", line=dict(color="#90A4AE", width=1.5, dash="dot")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tr_df["exit_time"], y=tr_df["cum_slip"], mode="lines",
        name="Shadow T2 (With CME Slippage)", line=dict(color="#FFA726", width=2.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=opt_df["exit_time"], y=opt_df["cum_opt"], mode="lines",
        name=f"Optimized T2 ({rule_desc})", line=dict(color="#00E676", width=3.0)
    ), row=1, col=1)

    # Row 2: Underwater Drawdown
    fig.add_trace(go.Scatter(
        x=tr_df["exit_time"], y=tr_df["dd_slip"], mode="lines", fill="tozeroy",
        name="Baseline T2 DD ($)", line=dict(color="#FF5252", width=1.5),
        fillcolor="rgba(255, 82, 82, 0.22)"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=opt_df["exit_time"], y=opt_df["dd_opt"], mode="lines",
        name="Optimized T2 DD ($)", line=dict(color="#00E5FF", width=2.0)
    ), row=2, col=1)

    # Row 3: Full Calendar Heatmap (Week x Weekday)
    fig.add_trace(go.Heatmap(
        z=z_cal,
        x=week_lbls,
        y=["Mon", "Tue", "Wed", "Thu", "Fri"],
        text=text_cal,
        hoverinfo="text",
        colorscale=[[0.0, "#f23645"], [0.5, "#1e222d"], [1.0, "#089981"]],
        zmid=0.0,
        name="Daily Calendar PnL",
        showscale=True,
        colorbar=dict(title="PnL ($)", len=0.14, y=0.58)
    ), row=3, col=1)

    # Row 4: 15-Minute Entry Window Efficiency
    fig.add_trace(go.Bar(
        x=bucket_grp.index,
        y=bucket_grp["net_pnl"],
        marker_color=b_colors,
        name="Net PnL by Entry Window ($)",
        text=[f"${v:,.0f}<br>({n} tr | {wr:.0f}% WR)" for v, n, wr in zip(bucket_grp["net_pnl"], bucket_grp["trades"], bucket_grp["win_rate"])],
        textposition="outside"
    ), row=4, col=1)

    # Row 5: MAE vs. MFE Scatter (Bubble size = Trade Duration in mins)
    fig.add_trace(go.Scatter(
        x=wins_df["mae_r"], y=wins_df["mfe_r"], mode="markers",
        name="Winning T2 Trades (Size = Duration)",
        marker=dict(
            color="#00E676",
            size=np.clip(np.sqrt(wins_df["duration_min"]) * 2.2, 8, 28),
            opacity=0.8, line=dict(width=1, color="#FFFFFF")
        ),
        text=[f"Date: {d}<br>PnL: ${p:,.2f}<br>MAE: {mae:.2f}R | MFE: {mfe:.2f}R<br>Duration: {dur:.0f}m"
              for d, p, mae, mfe, dur in zip(wins_df["trade_date"], wins_df["pnl_usd"], wins_df["mae_r"], wins_df["mfe_r"], wins_df["duration_min"])],
        hoverinfo="text"
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=loss_df["mae_r"], y=loss_df["mfe_r"], mode="markers",
        name="Losing T2 Trades (Size = Duration)",
        marker=dict(
            color="#FF5252",
            size=np.clip(np.sqrt(loss_df["duration_min"]) * 2.2, 8, 28),
            symbol="x", opacity=0.85
        ),
        text=[f"Date: {d}<br>PnL: ${p:,.2f}<br>MAE: {mae:.2f}R | MFE: {mfe:.2f}R<br>Duration: {dur:.0f}m"
              for d, p, mae, mfe, dur in zip(loss_df["trade_date"], loss_df["pnl_usd"], loss_df["mae_r"], loss_df["mfe_r"], loss_df["duration_min"])],
        hoverinfo="text"
    ), row=5, col=1)

    # Row 6: Monte Carlo Drawdown Cone
    trade_nums = np.arange(1, n_trades + 1)
    fig.add_trace(go.Scatter(
        x=trade_nums, y=dd_p50, mode="lines", name="MC Median Drawdown (50%)",
        line=dict(color="#29B6F6", width=2)
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=trade_nums, y=dd_p05, mode="lines", name="MC 95th Percentile Worst DD",
        line=dict(color="#FFA726", width=2, dash="dash")
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=trade_nums, y=dd_p01, mode="lines", fill="tonexty", name="MC 99th Percentile Tail DD",
        line=dict(color="#f23645", width=2, dash="dot"), fillcolor="rgba(242, 54, 69, 0.18)"
    ), row=6, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=2350,
        title_text=f"RORB QUANTITATIVE SUITE — SHADOW T1 -> REAL T2 (SLIPPAGE + RULE OPTIMIZATION)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Intra-Trade Max Adverse Excursion (MAE in R)", row=5, col=1)
    fig.update_yaxes(title_text="Max Favorable (MFE R)", row=5, col=1)
    fig.update_xaxes(title_text="Trade Sequence Number (1 -> N)", row=6, col=1)
    fig.update_yaxes(title_text="Underwater DD ($)", row=6, col=1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_html   = os.path.join(script_dir, OUTPUT_REPORT_HTM)
    fig.write_html(out_html, auto_open=False)
    webbrowser.open(f"file://{out_html}")
    print(f"[5/5] Dashboard saved & opened: {out_html}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    df_5m = load_mt5_data(CSV_FILENAME)
    t2_trades = run_shadow_t2_engine(df_5m)

    if len(t2_trades) == 0:
        raise RuntimeError("No Trade 2 setups were triggered in the dataset.")

    opt_trades, opt_rule_str = optimize_strategy_rules(t2_trades)

    # Print Comparative Tearsheet
    tearsheet = pd.DataFrame([
        summarize_segment(t2_trades, "1. Shadow T2 Baseline (Zero Slippage)", pnl_col="pnl_no_slip"),
        summarize_segment(t2_trades, "2. Shadow T2 Realistic (With CME Slippage)", pnl_col="pnl_usd"),
        summarize_segment(opt_trades, f"3. Optimized Shadow T2 ({opt_rule_str})", pnl_col="pnl_usd")
    ])

    print("\n" + "=" * 125)
    print(f" SHADOW T1 -> REAL T2 TEARSHEET | ENTRY SLIP: {ENTRY_SLIP_TICKS} tick | SL SLIP: {SL_SLIP_TICKS} ticks | RISK: ${RISK_PER_TRADE:.0f}/Trade")
    print("=" * 125)
    print(tearsheet.to_string(index=False))
    print("=" * 125)

    # Breakdown by Entry Bucket
    print("\n--- ENTRY WINDOW BREAKDOWN (WITH SLIPPAGE) ---")
    b_table = t2_trades.groupby("entry_bucket").agg(
        Trades=("pnl_usd", "count"),
        Win_Rate=("pnl_usd", lambda s: f"{(s > 0).mean()*100:.1f}%"),
        Net_PnL=("pnl_usd", lambda s: f"${s.sum():,.2f}"),
        Avg_MAE_R=("mae_r", lambda s: f"{s.mean():.2f} R"),
        Avg_MFE_R=("mfe_r", lambda s: f"{s.mean():.2f} R")
    )
    print(b_table.to_string())

    # Save CSV inside the script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv    = os.path.join(script_dir, OUTPUT_TRADES_CSV)
    t2_trades.to_csv(out_csv, index=False)
    print(f"\n[OK] Exported {len(t2_trades)} detailed Shadow T2 trades (with MAE/MFE/Slippage) to: {out_csv}")

    # Launch Visual Dashboard
    build_quant_dashboard(t2_trades, opt_trades, opt_rule_str)
