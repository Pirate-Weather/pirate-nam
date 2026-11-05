"""The WMS application: one dataset, the newest regridded store. Started by uvicorn behind nginx, which is the only public entry.
A restart picks up a new store, because the path is read when the app starts."""
import os
import cf_xarray  # noqa: F401  Registers the .cf accessor that xpublish-wms needs.
import xarray as xr, xpublish
from xpublish_wms import CfWmsPlugin
ds = xr.open_zarr(os.path.realpath(os.environ.get("WMS_STORE", "/srv/wms/current")), consolidated=True, zarr_format=3)
app = xpublish.Rest({"pirate-nam": ds}, plugins={"wms": CfWmsPlugin()}).app
