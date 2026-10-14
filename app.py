
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import spearmanr


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Edinburgh Nature Equitable Access Index",
    page_icon="🌳",
    layout="wide"
)

DATA = Path(__file__).resolve().parent / "data"


# ============================================================
# LOAD REAL EDINBURGH DATA
# ============================================================

@st.cache_data(show_spinner="Loading Edinburgh data...")
def load_data():

    zones = pd.read_csv(
        DATA / "edinburgh_zones.csv",
        dtype={"DataZone": str}
    )

    sites = pd.read_csv(
        DATA / "edinburgh_greenspaces.csv",
        dtype={"site_id": str}
    )

    entrances = pd.read_csv(
        DATA / "edinburgh_access_points.csv",
        dtype={"site_id": str}
    )

    shapes = json.loads(
        (DATA / "edinburgh_zones.geojson").read_text(
            encoding="utf-8"
        )
    )

    return zones, sites, entrances, shapes


# ============================================================
# CALCULATE STRAIGHT-LINE ACCESSIBILITY
# ============================================================

@st.cache_data(
    show_spinner="Calculating straight-line accessibility once..."
)
def distance_matrix(zones, sites, entrances):

    """
    Calculate the shortest great-circle distance
    from each Edinburgh data-zone reference point
    to each greenspace's mapped access points.

    Distances are in metres.

    This does NOT calculate pedestrian walking routes.

    No external API or distance CSV is required.
    """

    points = entrances[
        entrances.site_id.isin(sites.site_id)
    ].copy()

    if points.empty:
        raise ValueError(
            "No access points match the greenspace site IDs"
        )

    # Convert geographical coordinates to radians.

    zone_lat = np.radians(
        zones.latitude.to_numpy(dtype=float)
    )[:, None]

    zone_lon = np.radians(
        zones.longitude.to_numpy(dtype=float)
    )[:, None]

    point_lat = np.radians(
        points.latitude.to_numpy(dtype=float)
    )[None, :]

    point_lon = np.radians(
        points.longitude.to_numpy(dtype=float)
    )[None, :]

    # Haversine formula for great-circle distances.

    a = (
        np.sin((point_lat - zone_lat) / 2) ** 2
        + np.cos(zone_lat)
        * np.cos(point_lat)
        * np.sin((point_lon - zone_lon) / 2) ** 2
    )

    metres = (
        2
        * 6371000
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    ).astype("float32")

    # Multiple access points belonging to the same
    # greenspace must count as one greenspace site.

    labels, unique_ids = pd.factorize(
        points.site_id,
        sort=True
    )

    matrix = np.full(
        (len(zones), len(unique_ids)),
        np.inf,
        dtype="float32"
    )

    for j in range(len(unique_ids)):
        matrix[:, j] = metres[
            :, labels == j
        ].min(axis=1)

    return pd.DataFrame(
        matrix,
        index=zones.DataZone,
        columns=unique_ids
    )


# ============================================================
# CALCULATE THE NATURE EQUITABLE ACCESS INDEX
# ============================================================

def score_zones(
    zones,
    sites,
    dist,
    threshold,
    min_area,
    weight,
    categories
):

    # Select greenspaces satisfying the scenario.

    eligible = sites[
        (sites.area_ha >= min_area)
        & sites.type.isin(categories)
    ]

    ids = eligible.site_id[
        eligible.site_id.isin(dist.columns)
    ]

    out = zones.copy()

    if len(ids):

        m = dist.loc[
            out.DataZone,
            ids
        ].to_numpy()

        reachable = m <= threshold

        counts = reachable.sum(axis=1)

        nearest = np.where(
            reachable,
            m,
            np.inf
        ).min(axis=1)

    else:

        counts = np.zeros(
            len(out),
            dtype=int
        )

        nearest = np.full(
            len(out),
            np.inf
        )

    out["sites_reachable"] = counts

    out["nearest_m"] = np.where(
        np.isfinite(nearest),
        nearest,
        np.nan
    )

    # Proximity score, ranging from 0 to 1.

    out["proximity"] = np.where(
        np.isfinite(nearest),
        np.maximum(
            0,
            1 - nearest / threshold
        ),
        0
    )

    # Provision score.
    # Five or more qualifying sites = maximum score.

    out["provision"] = np.minimum(
        1,
        counts / 5
    )

    # Final scenario-based NEAI score.

    out["NEAI"] = (
        weight * out.proximity
        + (1 - weight) * out.provision
    )

    return out, eligible


# ============================================================
# TITLE AND INTRODUCTION
# ============================================================

st.title("Edinburgh Nature Equitable Access Index")

st.caption(
    "OS Open Greenspace · SIMD 2020v2 · "
    "2011 Scottish Data Zones"
)

st.write(
    "Where do deprivation and limited proximity "
    "to mapped greenspaces coincide in Edinburgh?"
)

st.warning(
    "**Straight-line proximity baseline only.** "
    "Distances are calculated to mapped greenspace "
    "access points, not along walking routes. "
    "Roads, barriers and public access restrictions "
    "are not modelled. Do not interpret the scores "
    "as walking accessibility."
)


# ============================================================
# LOAD DATA AND CALCULATE DISTANCES
# ============================================================

try:

    zones, sites, entrances, shapes = load_data()

    distances = distance_matrix(
        zones,
        sites,
        entrances
    )

except Exception as exc:

    st.error(
        "Could not load or calculate the real "
        f"Edinburgh data: {exc}"
    )

    st.info(
        "Check that the four input files are inside "
        "GitHub's `data/` folder. "
        "The distance CSV is no longer required."
    )

    st.stop()


# ============================================================
# INTERACTIVE SCENARIO CONTROLS
# ============================================================

st.sidebar.header("Accessibility scenario")

threshold = st.sidebar.slider(
    "Maximum straight-line distance (m)",
    min_value=200,
    max_value=2000,
    value=800,
    step=100
)

min_area = st.sidebar.slider(
    "Minimum greenspace area (ha)",
    min_value=0.0,
    max_value=5.0,
    value=0.2,
    step=0.1
)

weight = st.sidebar.slider(
    "Proximity weighting",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

categories = sorted(
    sites.type.dropna().unique().tolist()
)

default_categories = [
    x for x in categories
    if "Public Park" in x
] or categories

selected_categories = st.sidebar.multiselect(
    "Greenspace classifications",
    categories,
    default=default_categories
)

st.sidebar.caption(
    "Some OS classifications represent "
    "restricted-access facilities."
)


# ============================================================
# CALCULATE CURRENT SCENARIO
# ============================================================

results, selected_sites = score_zones(
    zones,
    sites,
    distances,
    threshold,
    min_area,
    weight,
    selected_categories
)

covered_pop = results.loc[
    results.sites_reachable > 0,
    "Population"
].sum()

pop = results.Population.sum()

mean_score = (
    np.average(
        results.NEAI,
        weights=results.Population
    )
    if pop > 0
    else results.NEAI.mean()
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

a, b, c, d = st.columns(4)

a.metric(
    "Edinburgh data zones",
    f"{len(zones):,}"
)

b.metric(
    "Eligible greenspaces",
    f"{len(selected_sites):,}"
)

c.metric(
    "Zone population near a site",
    (
        f"{100 * covered_pop / pop:.1f}%"
        if pop > 0
        else "N/A"
    )
)

d.metric(
    "Population-weighted baseline NEAI",
    f"{mean_score:.3f}"
)

st.caption(
    "Population percentage is an approximation "
    "based on one representative point per data zone, "
    "not individual homes."
)


# ============================================================
# APPLICATION TABS
# ============================================================

map_tab, equity_tab, scenario_tab, methodology_tab = st.tabs([
    "Edinburgh map",
    "Deprivation comparison",
    "Scenario explorer",
    "Methods & downloads"
])


# ============================================================
# TAB 1 — EDINBURGH MAP
# ============================================================

with map_tab:

    colour = st.radio(
        "Map data",
        [
            "NEAI",
            "SIMD_decile",
            "sites_reachable"
        ],
        horizontal=True
    )

    fig = px.choropleth_mapbox(
        results,
        geojson=shapes,
        locations="DataZone",
        featureidkey="properties.DataZone",
        color=colour,
        color_continuous_scale="RdYlGn",
        mapbox_style="carto-positron",
        center={
            "lat": 55.95,
            "lon": -3.20
        },
        zoom=10.1,
        opacity=0.7,
        hover_data=[
            "SIMD_rank",
            "SIMD_decile",
            "NEAI",
            "sites_reachable"
        ]
    )

    fig.update_layout(
        height=610,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Real data-zone polygons. The NEAI layer "
        "uses straight-line proximity, "
        "not pedestrian routing."
    )

    st.subheader("Mapped greenspace sites")

    st.dataframe(
        selected_sites[
            ["name", "type", "area_ha"]
        ].sort_values(
            "area_ha",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 — ENVIRONMENTAL DEPRIVATION
# ============================================================

with equity_tab:

    st.subheader(
        "Relationship between deprivation "
        "and greenspace proximity"
    )

    left, right = st.columns(2)

    with left:

        fig_scatter = px.scatter(
            results,
            x="SIMD_rank",
            y="NEAI",
            color="SIMD_decile",
            hover_name="DataZone",
            title="Deprivation versus proximity baseline"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

    with right:

        fig_box = px.box(
            results,
            x="SIMD_decile",
            y="NEAI",
            title="Proximity baseline by SIMD decile"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

    # Spearman's rank correlation.

    if (
        results.NEAI.nunique() > 1
        and results.SIMD_rank.nunique() > 1
    ):

        rho, p = spearmanr(
            results.SIMD_rank,
            results.NEAI
        )

        st.metric(
            "Spearman rank correlation",
            f"{rho:.3f}"
        )

        st.caption(
            f"Exploratory p = {p:.3g}; "
            "spatial autocorrelation can affect "
            "statistical significance."
        )

    st.subheader(
        "Most deprived data zones with "
        "lowest proximity scores"
    )

    priority_zones = results[
        results.SIMD_decile <= 2
    ].sort_values("NEAI")

    st.dataframe(
        priority_zones[
            [
                "DataZone",
                "SIMD_decile",
                "Population",
                "sites_reachable",
                "NEAI"
            ]
        ].head(30),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3 — SCENARIO EXPLORER
# ============================================================

with scenario_tab:

    st.subheader("Interactive scenario analysis")

    st.write(
        "Change the sidebar controls to examine "
        "how different assumptions affect "
        "greenspace accessibility scores."
    )

    fig_histogram = px.histogram(
        results,
        x="NEAI",
        nbins=20,
        title="Distribution of proximity-based NEAI"
    )

    st.plotly_chart(
        fig_histogram,
        use_container_width=True
    )

    zone = st.selectbox(
        "Inspect data zone",
        sorted(results.DataZone)
    )

    r = results.loc[
        results.DataZone == zone
    ].iloc[0]

    x, y, z = st.columns(3)

    x.metric(
        "Baseline index",
        f"{r.NEAI:.3f}"
    )

    y.metric(
        "Sites within threshold",
        int(r.sites_reachable)
    )

    z.metric(
        "Nearest site (straight line)",
        (
            f"{r.nearest_m:.0f} m"
            if pd.notna(r.nearest_m)
            else "None"
        )
    )


# ============================================================
# TAB 4 — METHODS AND DOWNLOADS
# ============================================================

with methodology_tab:

    st.subheader("Data and methodology")

    st.markdown(
        """
### Data sources

The application combines:

- OS Open Greenspace sites and access points.
- Scottish Index of Multiple Deprivation 2020v2.
- Scottish Data Zone Boundaries 2011.

### Distance calculation

The shortest straight-line distance is measured
between each data-zone reference point and
mapped greenspace access points.

The Haversine formula calculates geographical
distance using latitude and longitude.

Multiple entrances belonging to the same
greenspace count as one site.

### Proximity score

P = max(0, 1 - nearest distance / threshold)

If no qualifying greenspace lies within
the threshold, the proximity score is zero.

### Provision score

G = min(1, reachable greenspaces / 5)

Five or more qualifying greenspaces receive
the maximum provision score.

### Nature Equitable Access Index

NEAI = weight × proximity + (1 - weight) × provision

Scores range from zero to one.

### Limitations

This version does not model pedestrian routes.

Straight-line distances ignore:

- Road layout.
- Railway lines and other barriers.
- Footpaths and pedestrian crossings.
- Site-specific access restrictions.
- Differences in greenspace quality.

Data-zone reference points do not represent
every individual residential location.

SIMD 2020v2 is a historical dataset.

Statistical correlation does not establish causation.

### Future development

Replace the straight-line distance matrix with
validated network distances derived from OS Open
Roads or an appropriate pedestrian network.
"""
    )

    st.subheader("Download NEAI results")

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download scenario results (CSV)",
        data=csv,
        file_name="neai_edinburgh_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Contains Ordnance Survey data "
        "© Crown copyright and database right 2026. "
        "Scottish Government data under the "
        "Open Government Licence."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Edinburgh Nature Equitable Access Index | "
    "Research prototype | AI-assisted development"
)
