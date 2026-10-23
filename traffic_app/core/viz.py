"""Plotly and Folium builders."""
from __future__ import annotations
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .network import COLORS, SEGMENTS, INTERSECTIONS, LEVELS

INK = "#12343b"
ACCENT = "#0e7c86"
GRID = "#e3e8ea"
CENTER = (44.9600, -93.1750)
_HAS_SM = hasattr(go, "Scattermap")        # Plotly >= 5.24 (MapLibre); older versions use Mapbox


def style(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13, color=INK), legend=dict(orientation="h", y=1.1, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


# ------------------------------------------------------------------ folium map
def folium_map(states: pd.DataFrame, top_ids: set, height_note: str = "") -> str:
    """states: rows with seg (Segment), level, volume, ratio, speed.  Returns full HTML."""
    m = folium.Map(location=list(CENTER), zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    for _, r in states.iterrows():
        seg = r["seg"]
        tip = (f"<b>{seg.corridor}</b><br>{seg.name}<br>{r['level']} - {r['volume']:,.0f} veh/h "
               f"({r['ratio']:.0%} of capacity)<br>Speed about {r['speed']:.0f} km/h")
        folium.PolyLine(list(seg.coords), color=COLORS[r["level"]], weight=7, opacity=0.9,
                        tooltip=folium.Tooltip(tip, sticky=True)).add_to(m)
    for i in INTERSECTIONS:
        top = i.id in top_ids
        folium.CircleMarker((i.lat, i.lon), radius=8 if top else 5, color=INK, weight=2, fill=True,
                            fill_color=INK if top else "white", fill_opacity=1,
                            tooltip=f"{i.name}" + (" (top 5 busiest - signal plan on Signals tab)" if top else "")
                            ).add_to(m)
    legend = ("<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:#fff;padding:8px 12px;"
              "border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:12px sans-serif;color:#12343b'>"
              + "".join(f"<div><span style='display:inline-block;width:22px;height:5px;background:{COLORS[l]};"
                        f"margin-right:6px;vertical-align:middle'></span>{l}</div>" for l in LEVELS)
              + "<div style='margin-top:4px'><span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
                "background:#12343b;margin-right:8px'></span>Top-5 signal</div></div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m.get_root().render()


# ------------------------------------------------------------------ forecast chart
def forecast_figure(hist: pd.Series, t0, fc: dict, actual: dict, thresholds: tuple, title: str) -> go.Figure:
    """hist: observed volume (index = time); fc: {1:(mean,lo,hi), 2:(...)}; actual: {1:v|nan, 2:v|nan}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Observed", mode="lines",
                             line=dict(color=INK, width=2.5)))
    v0 = float(hist.iloc[-1])
    xs = [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)]
    mean = [v0, fc[1][0], fc[2][0]]
    lo = [v0, fc[1][1], fc[2][1]]
    hi = [v0, fc[1][2], fc[2][2]]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", mode="lines",
                             line=dict(width=0), fillcolor="rgba(14,124,134,0.18)", name="90% band",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=mean, name="Forecast", mode="lines+markers",
                             line=dict(color=ACCENT, width=2.5, dash="dash"), marker=dict(size=8)))
    ax = [x for x, k in zip(xs[1:], (1, 2)) if not np.isnan(actual.get(k, np.nan))]
    ay = [actual[k] for k in (1, 2) if not np.isnan(actual.get(k, np.nan))]
    if ax:
        fig.add_trace(go.Scatter(x=ax, y=ay, name="Actual (dataset)", mode="markers",
                                 marker=dict(symbol="diamond", size=11, color="#d63a3a")))
    for val, lab in ((thresholds[0], "Slow"), (thresholds[1], "Jammed")):
        fig.add_hline(y=val, line=dict(color=COLORS[lab], width=1, dash="dot"),
                      annotation_text=lab, annotation_position="top left")
    fig.update_yaxes(title="Vehicles per hour", rangemode="tozero")
    return style(fig, 400, title=dict(text=title, font=dict(size=15)))


# ------------------------------------------------------------------ time-lapse
def _line_trace(seg, color, width=7):
    lats, lons = [c[0] for c in seg.coords], [c[1] for c in seg.coords]
    cls = go.Scattermap if _HAS_SM else go.Scattermapbox
    return cls(lat=lats, lon=lons, mode="lines", line=dict(width=width, color=color), hoverinfo="skip",
               showlegend=False)


def timelapse_figure(levels_by_hour: pd.DataFrame, title: str) -> go.Figure:
    """levels_by_hour: index 0..23, columns = segment ids, values = level names."""
    ids = [s.id for s in SEGMENTS]
    first = levels_by_hour.iloc[0]
    fig = go.Figure(data=[_line_trace(s, COLORS[first[s.id]]) for s in SEGMENTS])
    frames = []
    for h in levels_by_hour.index:
        row = levels_by_hour.loc[h]
        frames.append(go.Frame(data=[_line_trace(s, COLORS[row[s.id]]) for s in SEGMENTS],
                               traces=list(range(len(ids))), name=str(h)))
    fig.frames = frames
    key = "map" if _HAS_SM else "mapbox"
    fig.update_layout(**{key: dict(style="carto-positron", center=dict(lat=CENTER[0], lon=CENTER[1]), zoom=11.3)})
    steps = [dict(method="animate", label=f"{h:02d}:00",
                  args=[[str(h)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for h in levels_by_hour.index]
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=40, b=0), template="plotly_white",
        title=dict(text=title, font=dict(size=15, color=INK)),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top",
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=0, steps=steps, x=0.14, y=-0.02, len=0.85, xanchor="left", yanchor="top",
                      currentvalue=dict(prefix="Hour: ", font=dict(size=14)))])
    return fig
