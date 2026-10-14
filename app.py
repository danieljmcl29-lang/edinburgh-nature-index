import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import folium
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(page_title="Edinburgh Nature Equitable Access Index", layout="wide")
CRS = "EPSG:27700"  # British National Grid — keep all layers in this CRS

st.title("Edinburgh Nature Equitable Access Index (NEAI)")
st.caption("OS Open Greenspace + OS Open Roads + SIMD — scenario-based equity of access to nature")

# ---------------- DATA LOADING ----------------
DATA_MODE = st.sidebar.radio("Data source", ["Demo (synthetic)", "Real OS/SIMD files"])

@st.cache_data
def make_demo_data(seed=42, n_zones=120, n_green=300, grid_n=40):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = 318000, 663000, 334000, 677000  # approx Edinburgh BNG bbox

    centroids = [Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy)) for _ in range(n_zones)]
    simd = gpd.GeoDataFrame({
        "DataZone": [f"S0100{i:04d}" for i in range(n_zones)],
        "SIMD_rank": rng.integers(1, 6977, n_zones),
        "SIMD_decile": rng.integers(1, 11, n_zones),
        "Population": rng.integers(400, 1200, n_zones),
    }, geometry=centroids, crs=CRS)

    green = gpd.GeoDataFrame({
        "id": range(n_green),
        "function": rng.choice(
            ["Public Park", "Playing Field", "Religious Grounds", "Play Space", "Amenity Greenspace"],
            n_green),
        "area_ha": np.round(rng.exponential(1.2, n_green) + 0.05, 2)
    }, geometry=[Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy)) for _ in range(n_green)], crs=CRS)

    gx, gy = np.linspace(minx, maxx, grid_n), np.linspace(miny, maxy, grid_n)
    pos = {i: (gx[i % grid_n], gy[i // grid_n]) for i in range(grid_n * grid_n)}
    G = nx.grid_2d_graph(grid_n, grid_n)
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        G[u][v]["weight"] = Point(pos[u]).distance(Point(pos[v]))
    nx.set_node_attributes(G, pos, "pos")
    return green, simd, G, pos

@st.cache_data
def load_real_data(green_fp, roads_fp, simd_fp):
    """Swap real OS/SIMD files in here.
    green_fp  -> OS Open Greenspace .gpkg (layer 'GreenspaceSite' + 'AccessPoint')
    roads_fp  -> OS Open Roads .gpkg (layer 'RoadLink')
    simd_fp   -> SIMD 2020 data zone boundaries joined to SIMD indicators
    """
    access = gpd.read_file(green_fp, layer="AccessPoint").to_crs(CRS)
    green_sites = gpd.read_file(green_fp, layer="GreenspaceSite").to_crs(CRS)
    roads = gpd.read_file(roads_fp, layer="RoadLink").to_crs(CRS)
    simd = gpd.read_file(simd_fp).to_crs(CRS)
    simd["geometry"] = simd.geometry.centroid

    green = gpd.sjoin(access, green_sites[["geometry", "function"]], how="left", predicate="intersects")
    if "area_ha" not in green.columns:
        green["area_ha"] = green_sites.geometry.area.mean() / 10000

    G = nx.Graph()
    for geom in roads.geometry:
        coords = list(geom.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            G.add_edge(a, b, weight=Point(a).distance(Point(b)))
    pos = {n: n for n in G.nodes()}
    return green, simd, G, pos

if DATA_MODE == "Demo (synthetic)":
    green, simd, G, pos = make_demo_data()
    st.sidebar.warning("Running on SYNTHETIC demo data — for structure/testing only.")
else:
    green_fp = st.sidebar.text_input("Greenspace .gpkg path")
    roads_fp = st.sidebar.text_input("Roads .gpkg path")
    simd_fp = st.sidebar.text_input("SIMD file path")
    if green_fp and roads_fp and simd_fp:
        green, simd, G, pos = load_real_data(green_fp, roads_fp, simd_fp)
    else:
        st.info("Enter file paths in the sidebar to load real data.")
        st.stop()

node_list = list(G.nodes())
node_xy = np.array([pos[n] for n in node_list])
kdtree = cKDTree(node_xy)

def snap(point):
    _, idx = kdtree.query([point.x, point.y])
    return node_list[idx]

# ---------------- SCENARIO CONTROLS ----------------
st.sidebar.header("Scenario controls")
threshold = st.sidebar.slider("Walking distance threshold (m)", 200, 2000, 800, 100,
                               help="Scottish Planning Policy open-space guidance uses ~500m as a walkable benchmark.")
min_size = st.sidebar.slider("Minimum greenspace size counted (ha)", 0.0, 5.0, 0.2, 0.1)
w_prox = st.sidebar.slider("Weight: proximity vs provision", 0.0, 1.0, 0.5)

qualifying_green = green[green["area_ha"] >= min_size] if "area_ha" in green.columns else green
green_nodes = tuple(sorted(set(qualifying_green.geometry.apply(snap).tolist())))

# ---------------- INDEX COMPUTATION ----------------
@st.cache_data(show_spinner="Computing network accessibility…")
def compute_index(green_nodes_key, threshold_val, simd_geom_wkb):
    from shapely import wkb as shapely_wkb
    results = []
    green_set = set(green_nodes_key)
    for wkb_geom in simd_geom_wkb:
        pt = shapely_wkb.loads(wkb_geom)
        origin = snap(pt)
        lengths = nx.single_source_dijkstra_path_length(G, origin, cutoff=threshold_val, weight="weight")
        reached = [n for n in green_set if n in lengths]
        nearest = min((lengths[n] for n in reached), default=np.inf)
        proximity = 1 - min(nearest / threshold_val, 1) if nearest != np.inf else 0
        provision = min(len(reached) / 5, 1)
        results.append((proximity, provision))
    return results

simd_geom_wkb = tuple(g.wkb for g in simd.geometry)
scores = compute_index(green_nodes, threshold, simd_geom_wkb)
simd = simd.copy()
simd["proximity"] = [s[0] for s in scores]
simd["provision"] = [s[1] for s in scores]
simd["NEAI"] = w_prox * simd["proximity"] + (1 - w_prox) * simd["provision"]

# ---------------- AREA SELECTION ----------------
st.subheader("Select data zones")
areas = st.multiselect("Choose data zones to inspect", simd["DataZone"].tolist(),
                        default=simd["DataZone"].tolist()[:10])
subset = simd[simd["DataZone"].isin(areas)] if areas else simd

# ---------------- MAP ----------------
st.subheader("Map: Nature Equitable Access Index")
m = folium.Map(location=[55.9533, -3.1883], zoom_start=12, tiles="cartodbpositron")
subset_wgs = subset.to_crs(4326)
for _, r in subset_wgs.iterrows():
    folium.CircleMarker(
        location=[r.geometry.y, r.geometry.x], radius=6,
        color=None, fill=True, fill_opacity=0.8,
        fill_color=f"rgb({int(255*(1-r.NEAI))},{int(255*r.NEAI)},80)",
        popup=f"{r.DataZone}: NEAI={r.NEAI:.2f}, SIMD decile={r.get('SIMD_decile','?')}"
    ).add_to(m)
st_folium(m, width=900, height=500)

# ---------------- DEPRIVATION COMPARISON ----------------
st.subheader("Equity check: NEAI vs deprivation")
col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(simd, x="SIMD_rank", y="NEAI", trendline="ols",
                      hover_data=["DataZone"], title="NEAI vs SIMD rank (lower rank = more deprived)")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.box(simd, x="SIMD_decile", y="NEAI", title="NEAI by SIMD decile")
    st.plotly_chart(fig2, use_container_width=True)

rho, p = spearmanr(simd["SIMD_rank"], simd["NEAI"])
st.metric("Spearman's rho (SIMD rank vs NEAI)", f"{rho:.2f}", help=f"p = {p:.3f}")

# ---------------- DOWNLOADS ----------------
st.subheader("Download results")
csv = simd.drop(columns="geometry").to_csv(index=False).encode()
st.download_button("Download CSV", csv, "neai_results.csv", "text/csv")
geojson = simd.to_json()
st.download_button("Download GeoJSON", geojson, "neai_results.geojson", "application/geo+json")
