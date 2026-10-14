
import streamlit as st

st.set_page_config(
    page_title="Edinburgh Nature Equitable Access Index",
    page_icon="🌳",
    layout="wide"
)

st.title("Edinburgh Nature Equitable Access Index")
st.subheader("Understanding access to urban greenspace")

st.write(
    "This project investigates differences in access to "
    "greenspace across Edinburgh, with particular attention "
    "to environmental deprivation."
)

st.success("The NEAI application is running successfully!")

distance = st.slider(
    "Maximum walking distance (metres)",
    min_value=200,
    max_value=2000,
    value=800,
    step=100
)

st.write(f"Selected walking distance: {distance} metres")

st.info(
    "This is an initial working version. Real greenspace, "
    "road-network and deprivation analysis will be added next."
)
