"""Edinburgh Nature Equitable Access Index — genuine datasets, provisional straight-line baseline.

Network routing is deliberately NOT claimed until a validated network distance file exists.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import spearmanr

st.set_page_config(page_title="Edinburgh Nature Equitable Access Index", page_icon="🌳", layout="wide")
DATA = Path(__file__).parent / "data"

@st.cache_data

def load_data():
    zones = pd.read_csv(DATA / "edinburgh_zones.csv", dtype={"DataZone": str})
    sites = pd.read_csv(DATA / "edinburgh_greenspaces.csv", dtype={"site_id": str})
    access = pd.read_csv(DATA / "edinburgh_access_points.csv", dtype={"site_id": str})
    shapes = json.loads((DATA / "edinburgh_zones.geojson").read_text(encoding="utf-8"))
    return zones, sites, access, shapes

@st.cache_data

def load_distances(filename):
    return pd.read_csv(DATA / filename, dtype={"DataZone": str, "site_id": str})


def score_zones(zones, sites, distances, threshold, min_area, weight, allowed_types):
    selected = sites[(sites.area_ha >= min_area) & sites.type.isin(allowed_types)]
    valid = distances[
        (distances.distance_m <= threshold) & distances.site_id.isin(selected.site_id)
    ]
    # Keep the shortest distance per site per zone, even if an optional routing
    # table contains multiple entrances for the same site.
    valid = valid.groupby(["DataZone", "site_id"], as_index=False).distance_m.min()
    agg = valid.groupby("DataZone").agg(
        nearest_m=("distance_m", "min"), sites_reachable=("site_id", "nunique")
    ).reset_index()
    out = zones.merge(agg, on="DataZone", how="left", validate="one_to_one")
    out["sites_reachable"] = out.sites_reachable.fillna(0).astype(int)
    out["proximity"] = (1 - out.nearest_m / threshold).clip(0, 1).fillna(0)
    out["provision"] = (out.sites_reachable / 5).clip(0, 1)
    out["NEAI"] = weight * out.proximity + (1 - weight) * out.provision
    return out, selected


st.title("Edinburgh Nature Equitable Access Index")
st.caption("OS Open Greenspace (2026) · SIMD 2020v2 · Scottish 2011 Data Zones")
st.write("Where do deprivation and limited access to mapped greenspaces coincide in Edinburgh?")

try:
    zones, sites, access, shapes = load_data()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Required data file missing or unreadable: {error}")
    st.info("Upload the complete `data/` folder alongside `app.py` in GitHub.")
    st.stop()

# A road-network matrix is optional and will only be used when explicitly present.
network_file = DATA / "network_distances.csv"
network_available = network_file.exists()
if network_available:
    mode = st.sidebar.radio("Distance model", ["Straight-line baseline", "Network model"])
else:
    mode = "Straight-line baseline"
    st.warning(
        "**Straight-line accessibility baseline (not pedestrian routing).** "
        "All mapped greenspaces and SIMD data are real, but route distances are "
        "currently measured as straight lines to mapped access points. "
        "OS Open Roads has not yet been processed. Do not describe these as walking distances."
    )

filename = "network_distances.csv" if mode == "Network model" else "straight_line_distances.csv"
try:
    distances = load_distances(filename)
except (FileNotFoundError, ValueError) as error:
    st.error(f"Distance data unavailable: {error}")
    st.stop()
required = {"DataZone", "site_id", "distance_m"}
if not required.issubset(distances.columns) or distances.distance_m.isna().any() or (distances.distance_m < 0).any():
    st.error("Distance matrix must contain DataZone, site_id and valid nonnegative distance_m values.")
    st.stop()

st.sidebar.header("Accessibility scenario")
threshold = st.sidebar.slider("Maximum distance (metres)", 200, 2000, 800, 100)
min_area = st.sidebar.slider("Minimum site area (hectares)", 0.0, 5.0, 0.2, 0.1)
weight = st.sidebar.slider("Proximity weighting", 0.0, 1.0, 0.5, 0.05)
all_types = sorted(sites.type.dropna().unique())
# Default to public park/garden; other classes can be included by user.
default_types = [t for t in all_types if "Public Park" in t]
if not default_types:
    default_types = all_types
selected_types = st.sidebar.multiselect("Greenspace classifications", all_types, default=default_types)
st.sidebar.caption("Not all mapped types permit unrestricted access. Site classifications are not ecological quality assessments.")
