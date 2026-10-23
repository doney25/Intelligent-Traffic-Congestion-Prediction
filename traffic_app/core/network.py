"""Schematic road network and congestion model.

The Metro Interstate dataset is ONE hourly sensor (I-94, Minneapolis-St Paul).
It has no map, no speeds and no per-road counts.  To build the city map the
brief asks for, every road segment below is *derived* from that single sensor:

    segment volume(t) = sensor volume(t - lag) * mult

`mult`, `lag`, `capacity`, free-flow speed and the intersection demand shares
are ASSUMPTIONS for demonstration (see the "Models & notes" tab).  Replace this
file with real detector data when you have it - nothing else needs to change.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

COLORS = {"Flowing": "#2e9e5b", "Slow": "#f0a81c", "Jammed": "#d63a3a"}
LEVELS = ("Flowing", "Slow", "Jammed")
DEFAULT_THRESHOLDS = (0.45, 0.70)   # volume/capacity: Slow from, Jammed from
BPR_ALPHA, BPR_BETA = 1.0, 4.0      # travel-time = t0 * (1 + a * (v/c)^b)


def _hav(a, b) -> float:
    r = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dphi, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


@dataclass(frozen=True)
class Segment:
    id: str
    name: str
    corridor: str
    coords: tuple
    mult: float
    lag: int
    capacity: float      # veh/h
    ffs: float           # free-flow speed km/h

    @property
    def length_km(self) -> float:
        return sum(_hav(self.coords[i], self.coords[i + 1]) for i in range(len(self.coords) - 1))

    @property
    def label(self) -> str:
        return f"{self.corridor} - {self.name}"


SENSOR_SEGMENT = "i94_snelling_lex"

SEGMENTS = [
    Segment("i94_cbd_river", "Minneapolis CBD to Riverside", "I-94",
            ((44.9735, -93.2810), (44.9712, -93.2650), (44.9690, -93.2480)), 1.08, 1, 7300, 95),
    Segment("i94_river_snel", "Riverside to Snelling", "I-94",
            ((44.9690, -93.2480), (44.9650, -93.2300), (44.9600, -93.2000), (44.9556, -93.1690)), 1.00, 0, 7300, 95),
    Segment("i94_snelling_lex", "Snelling to Lexington (sensor)", "I-94",
            ((44.9556, -93.1690), (44.9553, -93.1480), (44.9552, -93.1290)), 1.00, 0, 7300, 95),
    Segment("i94_lex_stp", "Lexington to St Paul CBD", "I-94",
            ((44.9552, -93.1290), (44.9540, -93.1130), (44.9500, -93.0960), (44.9490, -93.0900)), 0.92, 0, 7300, 95),
    Segment("i35w_n", "North of downtown", "I-35W",
            ((44.9725, -93.2560), (44.9900, -93.2600), (45.0100, -93.2580)), 0.90, 1, 7300, 95),
    Segment("i35w_s", "South of downtown", "I-35W",
            ((44.9725, -93.2560), (44.9500, -93.2640), (44.9300, -93.2660)), 0.80, 1, 7300, 95),
    Segment("i35e_n", "North of St Paul CBD", "I-35E",
            ((44.9490, -93.0980), (44.9650, -93.0970), (44.9850, -93.1000)), 0.72, 1, 7000, 90),
    Segment("i35e_s", "South of St Paul CBD", "I-35E",
            ((44.9490, -93.0980), (44.9350, -93.0950), (44.9200, -93.0930)), 0.66, 1, 7000, 90),
    Segment("hiawatha", "Downtown to Lake St", "Hiawatha Ave (MN-55)",
            ((44.9710, -93.2520), (44.9500, -93.2330), (44.9300, -93.2270)), 0.52, 2, 4300, 65),
    Segment("univ_w", "Raymond to Snelling", "University Ave",
            ((44.9650, -93.1960), (44.9600, -93.1830), (44.9560, -93.1690)), 0.38, 1, 2600, 45),
    Segment("univ_e", "Snelling to Rice St", "University Ave",
            ((44.9560, -93.1690), (44.9560, -93.1300), (44.9560, -93.1050)), 0.34, 1, 2600, 45),
    Segment("lake", "Hennepin to Hiawatha", "Lake St",
            ((44.9488, -93.3000), (44.9488, -93.2650), (44.9488, -93.2300)), 0.36, 2, 2700, 40),
    Segment("snel_n", "University to Como", "Snelling Ave",
            ((44.9556, -93.1670), (44.9700, -93.1670), (44.9870, -93.1670)), 0.33, 1, 2400, 45),
    Segment("snel_s", "University to Selby", "Snelling Ave",
            ((44.9556, -93.1670), (44.9420, -93.1670), (44.9250, -93.1670)), 0.30, 1, 2400, 45),
]
SEG = {s.id: s for s in SEGMENTS}

ROUTES = {
    "I-94 full (Minneapolis CBD to St Paul CBD)": ["i94_cbd_river", "i94_river_snel", "i94_snelling_lex", "i94_lex_stp"],
    "I-94 Riverside to Lexington": ["i94_river_snel", "i94_snelling_lex"],
    "I-35W through downtown": ["i35w_n", "i35w_s"],
    "I-35E through St Paul": ["i35e_n", "i35e_s"],
    "Hiawatha Ave (MN-55)": ["hiawatha"],
    "University Ave (Raymond to Rice St)": ["univ_w", "univ_e"],
    "Snelling Ave (Como to Selby)": ["snel_n", "snel_s"],
    "Lake St (Hennepin to Hiawatha)": ["lake"],
}


@dataclass(frozen=True)
class Intersection:
    id: str
    name: str
    lat: float
    lon: float
    scale: float          # approach demand = scale * sensor volume
    shares: tuple         # N, S, E, W share of demand (sums to 1)
    lanes: tuple          # N, S, E, W lanes


INTERSECTIONS = [
    Intersection("snel_univ", "Snelling Ave & University Ave", 44.9556, -93.1670, 0.29, (.30, .22, .26, .22), (2, 2, 1, 1)),
    Intersection("lake_hiaw", "Lake St & Hiawatha Ave", 44.9488, -93.2310, 0.31, (.20, .30, .28, .22), (2, 2, 1, 1)),
    Intersection("univ_rice", "University Ave & Rice St", 44.9560, -93.1050, 0.24, (.27, .23, .27, .23), (1, 1, 1, 1)),
    Intersection("hennepin_lake", "Lake St & Hennepin Ave", 44.9488, -93.2985, 0.26, (.28, .27, .23, .22), (1, 1, 1, 1)),
    Intersection("cedar_riv", "Cedar Ave & Riverside Ave", 44.9690, -93.2480, 0.26, (.24, .26, .28, .22), (1, 1, 1, 1)),
    Intersection("lex_univ", "Lexington Pkwy & University Ave", 44.9560, -93.1300, 0.22, (.26, .26, .24, .24), (1, 1, 1, 1)),
    Intersection("seventh_wab", "7th St & Wabasha St", 44.9490, -93.0970, 0.20, (.22, .28, .26, .24), (1, 1, 1, 1)),
    Intersection("snel_selby", "Snelling Ave & Selby Ave", 44.9470, -93.1670, 0.19, (.30, .28, .22, .20), (1, 1, 1, 1)),
]
INT = {i.id: i for i in INTERSECTIONS}


def level_from_ratio(ratio: float, th=DEFAULT_THRESHOLDS) -> str:
    if ratio >= th[1]:
        return "Jammed"
    if ratio >= th[0]:
        return "Slow"
    return "Flowing"


def travel_time_min(seg: Segment, volume: float) -> float:
    """Modelled travel time (minutes) for one segment at a given hourly volume."""
    ratio = max(volume, 0.0) / seg.capacity
    t0 = seg.length_km / seg.ffs * 60.0
    return t0 * (1.0 + BPR_ALPHA * ratio ** BPR_BETA)


def speed_kmh(seg: Segment, volume: float) -> float:
    return seg.length_km / (travel_time_min(seg, volume) / 60.0)


def freeflow_min(seg_ids) -> float:
    return sum(SEG[i].length_km / SEG[i].ffs * 60.0 for i in seg_ids)
