# Traffic Intelligence System (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```
Python 3.10-3.12 is the safest choice for TensorFlow. The first launch trains a small gradient-boosting baseline
(about 20 s) and caches it in `models/gb_baseline.joblib`.

## Tabs
| Tab | Brief requirement |
|---|---|
| Live map | Colour-coded roads (green / yellow / red), now or +1 h / +2 h; step the clock hour by hour in the sidebar |
| Forecast | 1 h and 2 h forecast per corridor with 90% band and chance of jam |
| Signals | Green-light durations for the 5 busiest intersections, current vs recommended |
| Peak hours | Animated 24 h time-lapse (typical weekday / weekend / any date) |
| Reliability | "Your commute on Road X takes A-B minutes 90% of the time" |
| Before / after | Fixed vs optimised signals: delay and vehicle-hours saved |
| Models and notes | GRU vs LSTM vs gradient boosting vs last-week baseline, assumptions |

## Files
- `app.py` - UI.  `core/data.py` - cleaning (same steps as the notebook).  `core/models.py` - GRU/LSTM loading,
  recursive 2 h forecast, back-test, prediction intervals.  `core/signals.py` - Webster plan and HCM delay.
  `core/network.py` - road network and assumptions.  `core/viz.py` - charts and map.
- `models/` - `traffic_gru.keras`, `traffic_lstm.keras`, `traffic_scaler.pkl` from your notebook.

## Read this before presenting it
The CSV is **one hourly sensor** (I-94). It has no map, speeds, travel times or intersection counts, so these parts are
**modelled and clearly assumptions**: road segments (sensor volume x multiplier, shifted 0-2 h), capacities, intersection
demand shares and lanes, travel times (BPR curve), and the approximate map geometry. All of them live in
`core/network.py`; replace them with real detector data and the rest of the app keeps working.
