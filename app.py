
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Edinburgh Nature Equitable Access Index",
    page_icon="🌳",
    layout="wide"
)

st.title("Edinburgh Nature Equitable Access Index")

st.caption(
    "Exploring greenspace distribution and environmental "
    "inequality across Edinburgh"
)

st.info(
    "Stage 1: Interactive greenspace map. "
    "Walking accessibility and SIMD deprivation "
    "analysis will be added in subsequent stages."
)

# ------------------------------------------
# EDINBURGH GREENSPACES
# ------------------------------------------
# Approximate representative coordinates.
# These are NOT official park boundaries
# or verified pedestrian entrances.

@st.cache_data
def load_greenspaces():

    records = [
        ("The Meadows", "Park", 55.9414, -3.1931),
        ("Princes Street Gardens", "Park", 55.9507, -3.2040),
        ("Holyrood Park", "Park", 55.9440, -3.1570),
        ("Inverleith Park", "Park", 55.9676, -3.2136),
        ("Saughton Park", "Park", 55.9344, -3.2485),
        ("Leith Links", "Park", 55.9718, -3.1628),
        ("Harrison Park", "Park", 55.9345, -3.2246),
        ("Victoria Park", "Park", 55.9757, -3.1959),
        ("Lochend Park", "Park", 55.9635, -3.1498),
        ("Figgate Park", "Park", 55.9529, -3.1185),
        ("Craigmillar Castle Park", "Park", 55.9222, -3.1476),
        ("Corstorphine Hill", "Woodland", 55.9573, -3.2790),
        ("Blackford Hill", "Woodland", 55.9234, -3.1946),
        ("Braid Hills", "Open Space", 55.9125, -3.2030),
        ("Calton Hill", "Park", 55.9550, -3.1827),
        ("Royal Botanic Garden", "Garden", 55.9661, -3.2090),
        ("Starbank Park", "Park", 55.9758, -3.2043),
        ("Pilrig Park", "Park", 55.9673, -3.1811),
        ("Duddingston Loch", "Blue Space", 55.9416, -3.1460),
        ("Portobello Beach", "Blue Space", 55.9536, -3.1110)
    ]

    return pd.DataFrame(
        records,
        columns=["Name", "Type", "lat", "lon"]
    )


greenspaces = load_greenspaces()

# ------------------------------------------
# SIDEBAR
# ------------------------------------------

st.sidebar.header("Map controls")

categories = sorted(greenspaces["Type"].unique())

selected = st.sidebar.multiselect(
    "Greenspace type",
    categories,
    default=categories
)

search = st.sidebar.text_input(
    "Search greenspace",
    ""
)

filtered = greenspaces[
    greenspaces["Type"].isin(selected)
].copy()

if search.strip():
    filtered = filtered[
        filtered["Name"].str.contains(
            search.strip(),
            case=False,
            regex=False
        )
    ]

# ------------------------------------------
# STATISTICS
# ------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Reference locations",
    len(greenspaces)
)

col2.metric(
    "Visible locations",
    len(filtered)
)

col3.metric(
    "Greenspace categories",
    filtered["Type"].nunique()
)

# ------------------------------------------
# INTERACTIVE MAP
# ------------------------------------------

st.subheader("Edinburgh Greenspace Map")

if not filtered.empty:

    st.map(
        filtered,
        latitude="lat",
        longitude="lon",
        size=90,
        zoom=11
    )

else:
    st.warning(
        "No locations match your selected filters."
    )

st.caption(
    "Locations are illustrative reference points "
    "with approximate coordinates, not complete "
    "greenspace polygons or official access points."
)

# ------------------------------------------
# DATA TABLE
# ------------------------------------------

st.subheader("Greenspace Records")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------
# DOWNLOAD
# ------------------------------------------

st.download_button(
    "Download greenspace records (CSV)",
    data=filtered.to_csv(index=False),
    file_name="edinburgh_greenspaces_demo.csv",
    mime="text/csv"
)

# ------------------------------------------
# PROJECT INFORMATION
# ------------------------------------------

st.subheader("About this project")

st.write(
    "The Edinburgh Nature Equitable Access Index "
    "(NEAI) aims to examine how access to urban "
    "greenspace varies across Edinburgh and how "
    "these differences relate to social deprivation."
)

st.write(
    "Future development will integrate Ordnance "
    "Survey greenspace data, walking networks and "
    "the Scottish Index of Multiple Deprivation "
    "(SIMD)."
)

st.caption(
    "Demonstration dataset only. "
    "Not suitable for quantitative accessibility "
    "or deprivation conclusions."
)
