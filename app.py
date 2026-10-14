
"""NEAI: Stage 1 — Edinburgh greenspace exploration.

OpenStreetMap features, not an OS Open Greenspace or accessibility study.
"""

import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import html

st.set_page_config(
    page_title="Edinburgh Nature Equitable Access Index",
    page_icon="🌳",
    layout="wide"
)

st.title("Edinburgh Nature Equitable Access Index")

st.caption(
    "Stage 1 · Explore Edinburgh greenspaces using OpenStreetMap"
)

st.info(
    "This map displays OpenStreetMap greenspace features. "
    "Walking-network accessibility, park entrances, SIMD and "
    "the final index will be added in subsequent stages."
)

# --------------------------------------------------
# OPENSTREETMAP DATA
# --------------------------------------------------

OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter"
]

# Central Edinburgh bounding box
QUERY = """[out:json][timeout:25];
(
 way["leisure"="park"](55.900,-3.300,56.000,-3.100);
 relation["leisure"="park"](55.900,-3.300,56.000,-3.100);
 way["leisure"="garden"](55.900,-3.300,56.000,-3.100);
 way["leisure"="nature_reserve"](55.900,-3.300,56.000,-3.100);
);
out center;"""


@st.cache_data(
    ttl=86400,
    show_spinner="Loading mapped greenspaces from OpenStreetMap..."
)
def load_parks():
    errors = []

    for url in OVERPASS_URLS:
        try:
            response = requests.post(
                url,
                data={"data": QUERY},
                timeout=45,
                headers={
                    "User-Agent": "NEAI-educational-demo/1.0"
                }
            )

            response.raise_for_status()
            elements = response.json().get("elements", [])

            records = []

            for element in elements:
                center = element.get("center", {})
                tags = element.get("tags", {})

                lat = center.get("lat")
                lon = center.get("lon")

                if lat is None or lon is None:
                    continue

                if not (
                    55.90 <= lat <= 56.0
                    and -3.30 <= lon <= -3.10
                ):
                    continue

                records.append({
                    "osm_id": f"{element['type']}/{element['id']}",
                    "name": tags.get(
                        "name", "Unnamed greenspace"
                    ),
                    "type": tags.get("leisure", "Other"),
                    "latitude": float(lat),
                    "longitude": float(lon)
                })

            if records:
                df = pd.DataFrame(records)
                return (
                    df.drop_duplicates(subset="osm_id"),
                    None
                )

            errors.append("No records returned")

        except (
            requests.RequestException,
            ValueError,
            KeyError
        ) as exc:
            errors.append(str(exc))

    empty = pd.DataFrame(columns=[
        "osm_id",
        "name",
        "type",
        "latitude",
        "longitude"
    ])

    return empty, "; ".join(errors)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("Explore greenspaces")

    st.write(
        "Source: © OpenStreetMap contributors (ODbL)"
    )

    if st.button("Refresh source data"):
        load_parks.clear()
        st.rerun()


# --------------------------------------------------
# LOAD GREENSPACE RECORDS
# --------------------------------------------------

parks, error = load_parks()

if parks.empty:

    st.error(
        "The live OpenStreetMap service is currently "
        "unavailable or returned no data. "
        "Please try Refresh source data later."
    )

    if error:
        st.caption(
            f"Data service details: {error[:400]}"
        )

    st.stop()


# --------------------------------------------------
# GREENSPACE FILTERS
# --------------------------------------------------

kinds = sorted(parks["type"].unique())

selected = st.sidebar.multiselect(
    "Greenspace categories",
    kinds,
    default=kinds
)

name_search = st.sidebar.text_input(
    "Search by name",
    ""
)

filtered = parks[
    parks["type"].isin(selected)
].copy()

if name_search.strip():

    filtered = filtered[
        filtered["name"].str.contains(
            name_search.strip(),
            case=False,
            regex=False
        )
    ]


# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

col1, col2 = st.columns(2)

col1.metric(
    "Mapped greenspace features",
    f"{len(parks):,}"
)

col2.metric(
    "Matching your filters",
    f"{len(filtered):,}"
)

st.caption(
    "Points are feature centres, not park entrances. "
    "Counts refer to mapped OSM features and may "
    "include overlapping features."
)


# --------------------------------------------------
# INTERACTIVE EDINBURGH MAP
# --------------------------------------------------

st.subheader("Edinburgh Greenspace Map")

map_obj = folium.Map(
    location=[55.9533, -3.1883],
    zoom_start=12,
    tiles="CartoDB positron",
    control_scale=True
)

for p in filtered.itertuples(index=False):

    label = html.escape(str(p.name))
    kind = html.escape(str(p.type))

    folium.CircleMarker(
        location=[
            p.latitude,
            p.longitude
        ],
        radius=5,
        color="#116c45",
        fill=True,
        fill_color="#2da66c",
        fill_opacity=0.8,
        popup=folium.Popup(
            f"<strong>{label}</strong><br>{kind}",
            max_width=250
        ),
        tooltip=label
    ).add_to(map_obj)

st_folium(
    map_obj,
    height=540,
    use_container_width=True,
    returned_objects=[]
)


# --------------------------------------------------
# GREENSPACE DATA TABLE
# --------------------------------------------------

st.subheader("Greenspace Records")

st.dataframe(
    filtered[[
        "name",
        "type",
        "latitude",
        "longitude",
        "osm_id"
    ]],
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

st.download_button(
    label="Download filtered records (CSV)",
    data=filtered.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="edinburgh_osm_greenspaces.csv",
    mime="text/csv"
)

st.caption(
    "Data © OpenStreetMap contributors · "
    "https://www.openstreetmap.org/copyright · "
    "Uses Overpass API. "
    "Not yet an OS Open Greenspace dataset or "
    "a measure of equitable access."
)
