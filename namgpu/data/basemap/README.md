Basemap for the weather maps (coastlines, lakes, national borders, state lines, US counties, land and
ocean fills), read by gpuwm's Rust renderer rw_wrfbatch through RUSTWX_BASEMAP_DIR when the renderer
finds no basemap of its own (namgpu.io.maps.ensure_basemap). The released gpuwm wheel ships the renderer
without these files, so a bare `pip install recast-woof-nam` draws its maps from this copy.

Files: unmodified copies of the gpuwm 2.9.11 renderer bundle's assets/basemap (natural_earth_10m,
natural_earth_110m, us_counties_5m; see the README in each folder for the source URLs).

- Natural Earth (naturalearthdata.com): public domain.
- US Census Bureau cartographic boundary file cb_2023_us_county_5m: work of the United States
  Government, not subject to copyright in the United States (17 U.S.C. 105).
