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

results, selected_sites = score_zones(zones, sites, distances, threshold, min_area, weight, selected_types)
covered = results[results.sites_reachable > 0]
pop_total = results.Population.sum()
pop_covered = covered.Population.sum()
weighted_index = np.average(results.NEAI, weights=results.Population) if pop_total > 0 else results.NEAI.mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Edinburgh data zones", f"{len(zones):,}")
m2.metric("Included greenspaces", f"{len(selected_sites):,}")
m3.metric("Zone populations with access", f"{100 * pop_covered / pop_total:.1f}%" if pop_total else "N/A")
m4.metric("Population-weighted NEAI", f"{weighted_index:.3f}")
st.caption("Population coverage is based on each data zone's representative point, not individual household locations.")

map_tab, equity_tab, explorer_tab, methodology_tab = st.tabs([
    "Accessibility map", "Environmental justice", "Scenario explorer", "Data and methodology"
])

with map_tab:
    metric = st.radio("Map colour", ["NEAI", "SIMD_decile", "sites_reachable"], horizontal=True)
    colour_names = {"NEAI": "NEAI score", "SIMD_decile": "SIMD decile (1 = most deprived)", "sites_reachable": "Reachable sites"}
    fig = px.choropleth_mapbox(
        results,
        geojson=shapes,
        locations="DataZone",
        featureidkey="properties.DataZone",
        color=metric,
        color_continuous_scale="RdYlGn" if metric != "SIMD_decile" else "RdYlBu",
        mapbox_style="carto-positron",
        center={"lat": 55.95, "lon": -3.20},
        zoom=10.1,
        opacity=0.69,
        hover_data={"SIMD_rank": True, "SIMD_decile": True, "NEAI": ":.3f", "sites_reachable": True},
        labels={metric: colour_names[metric]},
    )
    fig.update_layout(height=610, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Coloured polygons are the actual 2011 data-zone boundaries, not synthetic point locations.")
    st.subheader("Qualifying mapped greenspaces")
    st.dataframe(selected_sites[["name", "type", "area_ha"]].sort_values("area_ha", ascending=False),
                 use_container_width=True, hide_index=True)

with equity_tab:
    st.subheader("Relationship between deprivation and accessible nature")
    a, b = st.columns(2)
    with a:
        scatter = px.scatter(results, x="SIMD_rank", y="NEAI", color="SIMD_decile", hover_name="DataZone",
                             labels={"SIMD_rank": "SIMD rank (higher = less deprived)"})
        st.plotly_chart(scatter, use_container_width=True)
    with b:
        box = px.box(results, x="SIMD_decile", y="NEAI", points=False,
                     labels={"SIMD_decile": "SIMD deprivation decile"})
        st.plotly_chart(box, use_container_width=True)
    if results.NEAI.nunique() > 1 and results.SIMD_rank.nunique() > 1:
        rho, p = spearmanr(results.SIMD_rank, results.NEAI)
        st.metric("Spearman rank correlation", f"{rho:.3f}")
        st.caption(f"Exploratory p-value: {p:.3g}. Correlation is not causation; spatial dependence affects inference.")
    st.subheader("Most deprived areas with low modelled access")
    priority = results[results.SIMD_decile <= 2].sort_values("NEAI")
    st.dataframe(priority[["DataZone", "SIMD_decile", "Population", "sites_reachable", "NEAI"]].head(30),
                 use_container_width=True, hide_index=True)

with explorer_tab:
    st.subheader("How assumptions change the index")
    st.write("Use the left-hand controls to vary distance thresholds, minimum park size, classifications and index weighting.")
    hist = px.histogram(results, x="NEAI", nbins=20, title="Distribution of accessibility scores")
    st.plotly_chart(hist, use_container_width=True)
    choice = st.selectbox("Inspect data zone", sorted(results.DataZone))
    r = results[results.DataZone == choice].iloc[0]
    x, y, z = st.columns(3)
    x.metric("Index", f"{r.NEAI:.3f}")
    y.metric("Sites in range", int(r.sites_reachable))
    z.metric("Nearest site", f"{r.nearest_m:.0f} m" if pd.notna(r.nearest_m) else "None within threshold")

with methodology_tab:
    st.markdown("""
**Data:** OS Open Greenspace sites and access points (2026 extract), SIMD 2020v2 ranks and populations,
2011 Scottish data-zone boundaries. All data are real, but straight-line access is only a baseline.

**Proximity:** `max(0, 1 − nearest qualifying site distance / threshold)`.

**Provision:** `min(1, distinct qualifying reachable sites / 5)`.

**Index:** `NEAI = weight × proximity + (1 − weight) × provision`.

**Important limitations:** The 2011-zone centroid/representative point is not a population-weighted dwelling location.
OS Open Greenspace categories do not verify public accessibility or ecological quality. Baseline distances
measure straight-line proximity to mapped access points; they do **not** account for barriers, street layout
or pedestrian restrictions. OS Open Roads alone is not a complete pedestrian network. SIMD 2020v2 is a
historical, relative deprivation measure. This project does not establish causal relationships.

**Next methodological stage:** Independently preprocess and validate a pedestrian-accessible road/path network,
then add `data/network_distances.csv` containing one record per data-zone/site pair with the shortest
verified network distance in metres (covering at least the full 2 km scenario threshold). An absent route
must be distinguished from a route not computed. Until then the application remains a proximity baseline.
""")
    st.download_button("Download scenario results (CSV)", results.to_csv(index=False).encode("utf-8"),
                       file_name="neai_edinburgh_results.csv", mime="text/csv")
    st.caption("Contains Ordnance Survey data © Crown copyright and database right 2026. "
               "Scottish Government data-zone and SIMD data under the Open Government Licence.")

st.caption("NEAI · Research prototype · Independent analysis, with AI-assisted development")
