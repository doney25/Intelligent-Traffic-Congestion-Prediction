from __future__ import annotations
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html as st_html

from core import signals as sg
from core.data import hourly_profile, load_data
from core.models import (backtest, build_forecasters, exceed_probability, interval, metrics_table,
                         residual_bands)
from core.network import (BPR_ALPHA, BPR_BETA, DEFAULT_THRESHOLDS, INT, INTERSECTIONS, LEVELS, ROUTES, SEG,
                          SEGMENTS, SENSOR_SEGMENT, freeflow_min, level_from_ratio, speed_kmh)
from core.viz import ACCENT, INK, folium_map, forecast_figure, style, timelapse_figure

st.set_page_config(page_title="Traffic Intelligence System", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.0rem; max-width: 1440px;}
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #f3f8f6 0%, #e8f3f2 48%, #eef2f8 100%); background-attachment: fixed;}
[data-testid="stImage"] img {border-radius: 12px; width: 100%;}
[data-testid="stMetric"] {border: 1px solid #dfe5e8; border-radius: 8px; padding: .55rem .9rem;}
h1 {font-size: 1.75rem !important; margin-bottom: 0 !important;}
</style>""", unsafe_allow_html=True)

DOT = {"Flowing": "🟢", "Slow": "🟡", "Jammed": "🔴"}
HOUR = pd.Timedelta(hours=1)


# ============================================================ cached resources
@st.cache_resource(show_spinner="Loading data and models (first run trains a small baseline, ~20 s)...")
def get_ctx():
    data = load_data()
    fcs, notes = build_forecasters(data)
    bt = backtest(data, fcs)
    metrics = metrics_table(bt)
    bands = residual_bands(bt)
    cand = metrics[~metrics.Model.eq("Same hour last week")].groupby("Model")["MAE"].mean()
    return SimpleNamespace(data=data, fcs=fcs, notes=notes, bt=bt, metrics=metrics, bands=bands,
                           best=cand.idxmin() if len(cand) else None)


@st.cache_data(show_spinner=False)
def sensor_forecast(model: str, asof_iso: str):
    """{k: (mean, lo, hi)} for k = 1, 2 hours ahead of `asof`."""
    ctx = get_ctx()
    t = pd.Timestamp(asof_iso)
    p = ctx.fcs[model].predict_pair(ctx.data, [t])
    out = {}
    for k, pk in zip((1, 2), p):
        mean = float(pk[0])
        lo, hi = interval(ctx.bands, model, k, (t + k * HOUR).hour, mean)
        out[k] = (mean, lo, hi)
    return out


ctx = get_ctx()
data = ctx.data

# ============================================================ sidebar: as-of time
valid = data.valid_asof
vmin, vmax = valid[0].date(), valid[-1].date()


def _default_asof():
    fri = valid[(valid.dayofweek == 4) & (valid.hour == 15)]
    return fri[-1] if len(fri) else valid[-1]


def _set_asof(ts):
    st.session_state["asof_date"], st.session_state["asof_hour"] = ts.date(), int(ts.hour)


def _shift(hours, lo, hi):
    ts = pd.Timestamp(st.session_state["asof_date"]) + pd.Timedelta(hours=st.session_state["asof_hour"] + hours)
    _set_asof(min(max(ts, lo), hi))


if "asof_date" not in st.session_state:
    _set_asof(_default_asof())

with st.sidebar:
    st.header("Time and model")
    st.caption("The dataset ends in Sept 2018, so the app replays history as if it were live. "
               "Step forward an hour to see the map update.")
    st.date_input("Date", min_value=vmin, max_value=vmax, key="asof_date")
    st.slider("Hour of day", 0, 23, key="asof_hour")
    c1, c2 = st.columns(2)
    c1.button("-1 hour", on_click=_shift, args=(-1, valid[0], valid[-1]))
    c2.button("+1 hour", on_click=_shift, args=(1, valid[0], valid[-1]))
    st.button("Jump to rush hour (Fri 15:00)", on_click=lambda ts: _set_asof(ts), args=(_default_asof(),))
    st.button("Jump to latest data", on_click=lambda ts: _set_asof(ts), args=(valid[-1],))

    asof = pd.Timestamp(st.session_state["asof_date"]) + pd.Timedelta(hours=int(st.session_state["asof_hour"]))
    if asof not in valid:
        snapped = valid[valid <= asof]
        asof = snapped[-1] if len(snapped) else valid[0]
        st.warning(f"No complete 24 h of sensor history at that hour. Using {asof:%d %b %Y %H:%M}.")

    names = list(ctx.fcs)
    if not names:
        st.error("No forecasting model could be loaded. See the Models tab.")
        st.stop()
    model = st.selectbox("Forecast model", names, index=names.index(ctx.best) if ctx.best in names else 0)
    st.caption(f"Lowest back-test error: **{ctx.best}**" if ctx.best else "")

    with st.expander("Congestion thresholds"):
        slow_at = st.slider("Slow from (share of capacity)", 0.20, 0.80, DEFAULT_THRESHOLDS[0], 0.01)
        jam_at = st.slider("Jammed from (share of capacity)", 0.40, 1.00, DEFAULT_THRESHOLDS[1], 0.01)
        jam_at = max(jam_at, slow_at + 0.05)
    TH = (slow_at, jam_at)

    with st.expander("Signal settings"):
        fixed_cycle = st.slider("Current fixed cycle (s)", 60, 150, 90, 5)
        method = st.selectbox("Optimiser", sg.METHODS)

fc = sensor_forecast(model, str(asof))
sensor_now = float(data.full[asof])


# ============================================================ helpers
def sensor_est(s: pd.Timestamp):
    """(mean, lo, hi) of the sensor at time s: observed up to as-of, forecast after."""
    if s <= asof:
        v = data.full.get(s, np.nan)
        return v, v, v
    return fc[int((s - asof) / HOUR)]


def seg_state(seg, k: int) -> dict:
    m, lo, hi = sensor_est(asof + k * HOUR - seg.lag * HOUR)
    vol = m * seg.mult
    ratio = vol / seg.capacity
    return dict(seg=seg, volume=vol, lo=lo * seg.mult, hi=hi * seg.mult, ratio=ratio,
                level=level_from_ratio(ratio, TH), speed=speed_kmh(seg, vol))


def top_intersections(k: int, n: int = 5):
    vol = sensor_est(asof + k * HOUR)[0]
    rows = []
    for i in INTERSECTIONS:
        q = sg.flows_for(i, vol)
        rows.append((i, q, float(sg.flow_ratios(q, i.lanes).sum())))
    return sorted(rows, key=lambda r: r[1].sum(), reverse=True)[:n]


def day_profile(day_type: str) -> pd.Series:
    return hourly_profile(data.full, day_type)


# ============================================================ header
# Keep traffic_intelligence_banner.png in the same folder as this app.py file.
banner_path = Path(__file__).with_name("traffic_intelligence_banner.png")
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)

st.title("Traffic Intelligence System")
st.caption(f"I-94 Minneapolis - St Paul corridor  |  Viewing **{asof:%A %d %b %Y, %H:%M}**  |  "
           f"Model: **{model}**  |  Sensor now: **{sensor_now:,.0f} veh/h**")
tabs = st.tabs(["Live map", "Forecast", "Signals", "Peak hours", "Reliability", "Before / after", "Models and notes"])

# ============================================================ 1. live map
with tabs[0]:
    k = st.radio("Show", [0, 1, 2], format_func=lambda x: ["Now (observed)", "In 1 hour (forecast)",
                                                           "In 2 hours (forecast)"][x], horizontal=True)
    states = pd.DataFrame([seg_state(s, k) for s in SEGMENTS])
    counts = states["level"].value_counts()
    a, b, c, d = st.columns(4)
    for col, lev in zip((a, b, c), LEVELS):
        col.metric(f"{DOT[lev]} {lev}", int(counts.get(lev, 0)))
    d.metric("Corridor sensor", f"{sensor_est(asof + k * HOUR)[0]:,.0f} veh/h")
    top_ids = {r[0].id for r in top_intersections(k)}
    st_html(folium_map(states, top_ids), height=560)
    st.caption("Roads are coloured by volume as a share of capacity. Hover a road for volume and speed. "
               "Dark dots are the five busiest signalised intersections.")
    show = states.assign(Road=states.seg.map(lambda s: s.label), Status=states.level.map(lambda l: f"{DOT[l]} {l}"),
                         **{"Volume (veh/h)": states.volume.round(0).astype(int),
                            "Load": (states.ratio * 100).round(0).astype(int).astype(str) + "%",
                            "Speed (km/h)": states.speed.round(0).astype(int)})
    st.dataframe(show.sort_values("ratio", ascending=False)[["Road", "Status", "Volume (veh/h)", "Load", "Speed (km/h)"]],
                 hide_index=True)

# ============================================================ 2. forecast
with tabs[1]:
    st.subheader("1-hour and 2-hour forecast")
    labels = {s.label: s for s in SEGMENTS}
    pick = st.selectbox("Road corridor", list(labels), index=list(labels.values()).index(SEG[SENSOR_SEGMENT]))
    seg = labels[pick]
    hist_times = pd.date_range(asof - 21 * HOUR, asof, freq="h")
    hist = pd.Series([data.full.get(t - seg.lag * HOUR, np.nan) * seg.mult for t in hist_times], index=hist_times)
    seg_fc, actual, cols = {}, {}, st.columns(2)
    for k_, col in zip((1, 2), cols):
        st1 = seg_state(seg, k_)
        seg_fc[k_] = (st1["volume"], st1["lo"], st1["hi"])
        actual[k_] = data.full.get(asof + k_ * HOUR - seg.lag * HOUR, np.nan) * seg.mult
        if seg.lag >= k_:
            p_jam = float(st1["ratio"] >= TH[1])
        else:
            p_jam = exceed_probability(ctx.bt, model, k_ - seg.lag, (asof + (k_ - seg.lag) * HOUR).hour,
                                       st1["volume"] / seg.mult, TH[1] * seg.capacity / seg.mult)
        col.metric(f"In {k_} hour{'s' if k_ > 1 else ''}",
                   f"{st1['volume']:,.0f} veh/h", f"{st1['volume'] - hist.iloc[-1]:+,.0f} vs now")
        col.write(f"{DOT[st1['level']]} **{st1['level']}**  |  90% range {st1['lo']:,.0f} to {st1['hi']:,.0f} veh/h  |  "
                  f"chance of jam: **{p_jam:.0%}**")
    st.plotly_chart(forecast_figure(hist.dropna(), asof, seg_fc, actual,
                                    (TH[0] * seg.capacity, TH[1] * seg.capacity), f"{seg.label}: observed and forecast"))
    if seg.lag >= 1:
        st.caption(f"This segment follows the sensor by {seg.lag} h in the network model, so part of its forecast "
                   "comes from flow that has already been observed and has no model uncertainty.")
    st.caption("The band is the central 90% of the model's errors over the test period (Oct 2017 - Sep 2018) "
               "for the same hour of day. Red diamonds show what the dataset recorded, when available.")
    rows = []
    for s in SEGMENTS:
        r = {"Road": s.label}
        for k_ in (0, 1, 2):
            x = seg_state(s, k_)
            r[["Now", "+1 h", "+2 h"][k_]] = f"{DOT[x['level']]} {x['volume']:,.0f}"
        rows.append(r)
    st.markdown("**All corridors** (veh/h)")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# ============================================================ 3. signals
with tabs[2]:
    st.subheader("Signal timing recommendations")
    kk = st.radio("Plan for", [1, 2], format_func=lambda x: f"{x} hour ahead", horizontal=True, key="sig_k")
    top = top_intersections(kk)
    rows = []
    for i, q, Y in top:
        plan = sg.optimise(q, i.lanes, method, fixed_cycle)
        rows.append({"Intersection": i.name, "Entering flow (veh/h)": int(q.sum()), "Load": round(Y, 2),
                     "Status": sg.status(Y), "Cycle (s)": int(round(plan["cycle"])),
                     **{f"{a[0]} green (s)": int(round(g)) for a, g in zip(sg.APPROACHES, plan["green"])}})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Top 5 by predicted entering flow. Greens are per approach (North, South, East, West), one phase each. "
               "Load is the sum of flow ratios (about 0.9 or more means the junction is close to capacity).")
    names_top = [r[0].name for r in top]
    chosen = st.selectbox("Compare current and recommended plan for", names_top)
    i, q, Y = next(r for r in top if r[0].name == chosen)
    new, old = sg.optimise(q, i.lanes, method, fixed_cycle), sg.fixed_plan(fixed_cycle)
    en, eo = sg.evaluate(q, i.lanes, new), sg.evaluate(q, i.lanes, old)
    m1, m2, m3 = st.columns(3)
    m1.metric("Average delay per vehicle", f"{en['avg_delay']:.0f} s", f"{en['avg_delay'] - eo['avg_delay']:+.0f} s vs fixed",
              delta_color="inverse")
    m2.metric("Cycle length", f"{new['cycle']:.0f} s", f"{new['cycle'] - old['cycle']:+.0f} s vs fixed", delta_color="off")
    m3.metric("Busiest approach saturation", f"{en['max_x']:.2f}", f"{en['max_x'] - eo['max_x']:+.2f} vs fixed",
              delta_color="inverse")
    fig = go.Figure()
    fig.add_bar(x=list(sg.APPROACHES), y=old["green"], name=f"Current fixed ({old['cycle']:.0f} s cycle)", marker_color="#9aa9ae")
    fig.add_bar(x=list(sg.APPROACHES), y=new["green"], name=f"Recommended ({new['cycle']:.0f} s cycle)", marker_color=ACCENT)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Green time (s)")
    st.plotly_chart(style(fig, 330))
    st.caption("Planning-level estimate using Webster's method and the HCM delay equation. Approach demand and lanes are "
               "assumptions (the dataset has no intersection counts). Pedestrian phases and coordination are not modelled.")

# ============================================================ 4. peak hours
with tabs[3]:
    st.subheader("How congestion spreads through the rush hours")
    mode = st.radio("Day", ["Typical weekday", "Typical weekend", "A specific date"], horizontal=True)
    vol_by_hour = None
    if mode == "A specific date":
        dmin, dmax = data.full.dropna().index[[0, -1]]
        day = st.date_input("Date", value=asof.date(), min_value=dmin.date(), max_value=dmax.date(), key="tl_date")
        s = data.full.loc[str(day)]
        if s.notna().sum() >= 18:
            vol_by_hour = s.interpolate(limit_direction="both")
            vol_by_hour.index = vol_by_hour.index.hour
            title = f"Congestion on {pd.Timestamp(day):%A %d %b %Y}"
        else:
            st.warning("The sensor has too few readings that day. Showing a typical weekday instead.")
    if vol_by_hour is None:
        vol_by_hour = day_profile("Weekend" if mode == "Typical weekend" else "Weekday")
        title = f"Congestion over a {mode.lower()}"
    lv = {}
    for h in range(24):
        row = {}
        for s_ in SEGMENTS:
            v = vol_by_hour[(h - s_.lag) % 24] * s_.mult
            row[s_.id] = level_from_ratio(v / s_.capacity, TH)
        lv[h] = row
    lv = pd.DataFrame(lv).T
    st.plotly_chart(timelapse_figure(lv, title))
    share = (lv == "Jammed").mean(axis=1) * 100
    pk_am, pk_pm = share.loc[5:11].idxmax(), share.loc[12:21].idxmax()
    a, b, c = st.columns(3)
    a.metric("Morning peak", f"{pk_am:02d}:00", f"{share[pk_am]:.0f}% of roads jammed", delta_color="off")
    b.metric("Evening peak", f"{pk_pm:02d}:00", f"{share[pk_pm]:.0f}% of roads jammed", delta_color="off")
    c.metric("Corridor sensor at evening peak", f"{vol_by_hour[pk_pm]:,.0f} veh/h")
    f2 = go.Figure(go.Scatter(x=list(range(24)), y=share.values, mode="lines+markers", line=dict(color="#d63a3a", width=2.5)))
    f2.update_xaxes(title="Hour of day", dtick=2)
    f2.update_yaxes(title="Roads jammed (%)", rangemode="tozero")
    st.plotly_chart(style(f2, 280, title=dict(text="Share of roads jammed by hour", font=dict(size=15))))
    hm = data.full.dropna()
    hm = hm.groupby([hm.index.dayofweek, hm.index.hour]).mean().unstack()
    f3 = go.Figure(go.Heatmap(z=hm.values, x=[f"{h:02d}" for h in hm.columns],
                              y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], colorscale="YlOrRd",
                              colorbar=dict(title="veh/h")))
    f3.update_yaxes(autorange="reversed")
    st.plotly_chart(style(f3, 320, title=dict(text="Average sensor volume: day of week by hour", font=dict(size=15))))
    st.caption("The sensor record is real. How the jams move between roads is modelled: each road follows the sensor "
               "with a delay of 0 to 2 hours (network assumptions on the Models tab).")

# ============================================================ 5. reliability
with tabs[4]:
    st.subheader("Travel-time reliability")
    route = st.selectbox("Route", list(ROUTES))
    ids = ROUTES[route]
    cA, cB = st.columns(2)
    days = cA.radio("Days", ["Weekdays", "Weekends", "All days"], horizontal=True)
    w0, w1 = cB.select_slider("Departure window (hour of day)", options=list(range(24)), value=(7, 9))
    series = data.full
    tt = sum((SEG[i].length_km / SEG[i].ffs * 60.0) *
             (1 + BPR_ALPHA * ((series.shift(SEG[i].lag) * SEG[i].mult) / SEG[i].capacity).clip(lower=0) ** BPR_BETA)
             for i in ids).dropna()
    dmask = tt.index.dayofweek >= 5 if days == "Weekends" else tt.index.dayofweek < 5 if days == "Weekdays" else np.ones(len(tt), bool)
    sample = tt[dmask & (tt.index.hour >= w0) & (tt.index.hour <= w1)]
    length = sum(SEG[i].length_km for i in ids)
    ff = freeflow_min(ids)
    if len(sample) < 30:
        st.warning("Too few readings for that selection. Widen the window or pick another day type.")
    else:
        p5, p50, p95, mean = sample.quantile(.05), sample.median(), sample.quantile(.95), sample.mean()
        lo_m, hi_m = int(np.floor(p5)), int(np.ceil(p95))
        st.success(f"**Your commute on {route} ({length:.1f} km) takes {lo_m} to {hi_m} minutes 90% of the time** "
                   f"({days.lower()}, departing {w0:02d}:00 to {w1:02d}:59). Typical trip: {p50:.0f} min.")
        a, b, c, d = st.columns(4)
        a.metric("Free-flow time", f"{ff:.1f} min")
        b.metric("Median", f"{p50:.1f} min")
        c.metric("95th percentile", f"{p95:.1f} min")
        d.metric("Buffer index", f"{(p95 - mean) / mean:.0%}", help="Extra time to budget over the average trip so you are on time 19 days in 20.")
        base = tt[dmask]
        g = base.groupby(base.index.hour)
        band = pd.DataFrame({"p5": g.quantile(.05), "p50": g.median(), "p95": g.quantile(.95)})
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=band.index, y=band.p95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        f1.add_trace(go.Scatter(x=band.index, y=band.p5, mode="lines", line=dict(width=0), fill="tonexty",
                                fillcolor="rgba(14,124,134,0.2)", name="5th to 95th percentile"))
        f1.add_trace(go.Scatter(x=band.index, y=band.p50, mode="lines+markers", line=dict(color=INK, width=2.5), name="Median"))
        f1.add_vrect(x0=w0 - 0.5, x1=w1 + 0.5, fillcolor="rgba(0,0,0,0.06)", line_width=0)
        f1.update_xaxes(title="Departure hour", dtick=2)
        f1.update_yaxes(title="Minutes", rangemode="tozero")
        st.plotly_chart(style(f1, 340, title=dict(text="Travel time by departure hour", font=dict(size=15))))
        f2 = go.Figure(go.Histogram(x=sample.values, nbinsx=30, marker_color=ACCENT))
        f2.add_vline(x=p95, line=dict(color="#d63a3a", dash="dash"), annotation_text="95th percentile")
        f2.update_xaxes(title="Minutes")
        f2.update_yaxes(title="Hours observed")
        st.plotly_chart(style(f2, 280, title=dict(text="Distribution in the selected window", font=dict(size=15))))
        st.download_button("Download hourly percentiles (CSV)", band.round(2).to_csv(index_label="hour"),
                           file_name="travel_time_reliability.csv", mime="text/csv")
    st.caption("The dataset has no travel times. These are modelled from hourly volume with a BPR-style curve "
               f"(time = free-flow x (1 + {BPR_ALPHA} x (volume/capacity)^{BPR_BETA:g})), so they show how demand "
               "variability translates into trip-time variability, not measured commutes.")

# ============================================================ 6. before / after
with tabs[5]:
    st.subheader("Fixed signals vs. optimised signals")
    cA, cB = st.columns(2)
    inter_name = cA.selectbox("Intersection", [i.name for i in INTERSECTIONS])
    dtype = cB.radio("Typical day", ["Weekday", "Weekend"], horizontal=True)
    prof = day_profile(dtype)

    def simulate(inter):
        rows = []
        for h in range(24):
            q = sg.flows_for(inter, prof[h])
            old = sg.evaluate(q, inter.lanes, sg.fixed_plan(fixed_cycle))
            new = sg.evaluate(q, inter.lanes, sg.optimise(q, inter.lanes, method, fixed_cycle))
            rows.append(dict(hour=h, flow=q.sum(), fixed_delay=old["avg_delay"], opt_delay=new["avg_delay"],
                             fixed_vh=old["veh_hours"], opt_vh=new["veh_hours"]))
        return pd.DataFrame(rows)

    sim = simulate(INT[next(i.id for i in INTERSECTIONS if i.name == inter_name)])
    tot = sim.flow.sum()
    d_old, d_new = (sim.fixed_delay * sim.flow).sum() / tot, (sim.opt_delay * sim.flow).sum() / tot
    saved_vh = (sim.fixed_vh - sim.opt_vh).sum()
    a, b, c = st.columns(3)
    a.metric("Average delay per vehicle", f"{d_new:.0f} s", f"{d_new - d_old:+.0f} s ({(d_new / d_old - 1):+.0%})",
             delta_color="inverse")
    b.metric("Vehicle-hours saved per day", f"{saved_vh:,.0f}")
    c.metric("Vehicles per day (all approaches)", f"{tot:,.0f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.hour, y=sim.fixed_delay, name="Fixed signals", line=dict(color="#9aa9ae", width=3)))
    fig.add_trace(go.Scatter(x=sim.hour, y=sim.opt_delay, name="Optimised signals", line=dict(color=ACCENT, width=3),
                             fill="tonexty", fillcolor="rgba(14,124,134,0.15)"))
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title="Average delay per vehicle (s)", rangemode="tozero")
    st.plotly_chart(style(fig, 360, title=dict(text=f"{inter_name}: delay through the day", font=dict(size=15))))

    st.markdown("**All modelled intersections**")
    out, all_flow, all_saved = [], 0.0, 0.0
    for i in INTERSECTIONS:
        s_ = simulate(i)
        t_ = s_.flow.sum()
        o, n = (s_.fixed_delay * s_.flow).sum() / t_, (s_.opt_delay * s_.flow).sum() / t_
        saved = (s_.fixed_vh - s_.opt_vh).sum()
        all_flow, all_saved = all_flow + t_, all_saved + saved
        out.append({"Intersection": i.name, "Delay now (s)": round(o), "Delay optimised (s)": round(n),
                    "Change": f"{n / o - 1:+.0%}", "Vehicle-hours saved / day": round(saved)})
    st.dataframe(pd.DataFrame(out), hide_index=True)
    st.caption(f"Across these {len(out)} intersections: **{all_saved:,.0f} vehicle-hours saved per day**, about "
               f"{all_saved * 3600 / max(all_flow, 1):.0f} s per vehicle passing a signal. "
               f"Baseline: {fixed_cycle} s cycle, equal green for every approach, all day. "
               "The optimised plan is recomputed every hour.")

# ============================================================ 7. models and notes
with tabs[6]:
    st.subheader("Model comparison")
    for n in ctx.notes:
        st.warning(n)
    mt = ctx.metrics.copy()
    st.dataframe(mt.assign(MAE=mt.MAE.round(0), RMSE=mt.RMSE.round(0)), hide_index=True)
    fig = go.Figure()
    for m_ in mt.Model.unique():
        g = mt[mt.Model == m_]
        fig.add_bar(x=g.Horizon, y=g.MAE, name=m_)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Mean absolute error (veh/h)")
    st.plotly_chart(style(fig, 320, title=dict(text="Back-test error, Oct 2017 - Sep 2018 (lower is better)", font=dict(size=15))))
    mm = st.selectbox("Actual vs predicted (1 hour ahead) for", [m_ for m_ in ctx.fcs], key="avp_model")
    g = ctx.bt[(ctx.bt.model == mm) & (ctx.bt.h == 1)].head(24 * 14)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=g.time + HOUR, y=g.actual, name="Actual", line=dict(color=INK, width=2)))
    f2.add_trace(go.Scatter(x=g.time + HOUR, y=g.pred, name=mm, line=dict(color=ACCENT, width=2)))
    f2.update_yaxes(title="Vehicles per hour")
    st.plotly_chart(style(f2, 320, title=dict(text="First two weeks of the test period", font=dict(size=15))))
    st.download_button("Download metrics (CSV)", mt.to_csv(index=False), file_name="model_metrics.csv", mime="text/csv")
    st.markdown("""
**How the models work.** GRU and LSTM (from your notebook, saved as `.keras`) read the previous 24 hours of volume and
predict the next hour. The 2-hour forecast feeds the 1-hour prediction back in. Gradient Boosting is a baseline
trained on current and lagged volume, rolling averages, calendar and weather. "Same hour last week" is the naive reference.
Only hours with 25 straight hours of sensor history are tested, so the numbers can differ slightly from the notebook.

**What is real and what is assumed.** The Metro Interstate file is one sensor on I-94. Everything below is modelled
from it and should be replaced with real detector data before any operational use:
- Road segments: volume = sensor volume (shifted 0-2 h) x a multiplier; capacities and free-flow speeds are assumed.
- Congestion colours: share of capacity (Slow and Jammed cut-offs are in the sidebar).
- Intersections: demand is a fixed share of sensor volume, split across four approaches with assumed lanes.
- Travel times: BPR-style curve on volume; not measured.
- Map geometry is approximate and schematic.
""")
