# Edinburgh Nature Equitable Access Index (NEAI)

Real Edinburgh data-zone boundaries, SIMD 2020v2 deprivation ranks, and OS Open Greenspace sites/access points. The supplied `straight_line_distances.csv` is a **straight-line baseline** measured from zone representative points to mapped access points in British National Grid metres. It is **not a walking network**.

## Streamlit Community Cloud
Upload `app.py`, `requirements.txt` and the **entire `data/` folder** to the root of the public GitHub repository. Deploy `app.py` from branch `main`. All files are small; no external data API calls are required.

## Future road-network mode
Only after producing a *validated* network matrix with `DataZone,site_id,distance_m`, save it as `data/network_distances.csv`. Distances should be minimum verified access-point routing distances for each site and zone, and coverage should be complete to at least 2,000 m; unreachable versus unprocessed results must be distinguished. OS Open Roads alone is not complete pedestrian routing data.

## Data and acknowledgements
Contains Ordnance Survey data © Crown copyright and database right 2026. Scottish Government 2011 Data Zone Boundaries and SIMD 2020v2, Open Government Licence. Academic research prototype with AI-assisted development.
