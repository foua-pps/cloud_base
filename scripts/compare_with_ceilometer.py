# %%
import os
import re
from collections import OrderedDict
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pyresample


# %%
def check_if_in_domain(clat, clon, slats, slons):
    if (
        np.nanmin(slons) < clon
        and np.nanmax(slons) > clon
        and np.nanmin(slats) < clat
        and np.nanmax(slats) > clat
    ):
        return True
    return False


def find_nearest_index_in_space(clat, clon, slats, slons):
    grid = pyresample.geometry.GridDefinition(lats=slats, lons=slons)
    swath = pyresample.geometry.SwathDefinition(lons=[clon], lats=[clat])

    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
        source_geo_def=grid,
        target_geo_def=swath,
        radius_of_influence=2000,
        neighbours=1,
    )

    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
    # the 2D grid indices:
    if np.all(~np.isfinite(distance_array)):
        return [], []
    return np.unravel_index(index_array, grid.shape)


def find_nearest_index_in_time(cm_time, sat_time):
    diff = np.abs(cm_time.astype("int64") - sat_time.astype("int64"))
    index = np.argmin(diff)
    print(cm_time[index].astype("datetime64[ns]"), sat_time)
    return index


def collect_collocated_data(cm, cbhfiles):
    """collects saellite data  matched with Lidar data."""
    cbhs = []
    cbhs_cm = []
    cths = []
    cths_cm = []
    times = []
    sunzenith = []
    sat = []
    target_lon = cm.longitude.data[0]
    target_lat = cm.latitude.data[0]
    for ix, file in enumerate(cbhfiles[:]):
        with xr.open_dataset(file) as ds:
            mask = np.isfinite(ds.cbh_alti.data[0])
            if (
                check_if_in_domain(
                    target_lat, target_lon, ds.lat.data[mask], ds.lon.data[mask]
                )
                is True
            ):
                imagerfile = glob.glob(
                    os.path.join(
                        "/home/sm_indka/data/pps/import/IMAGER_data/",
                        "*" + os.path.basename(file)[15:],
                    )
                )[0]
                cthfile = os.path.join(
                    "/nobackup/smhid20/proj/foua/data/NWCSAF/CBH_FMI_MAR25/FMI_CBH_PPS/",
                    os.path.basename(file).replace("CBH", "CTTH"),
                )
                with xr.open_dataset(cthfile) as cth:
                    ii, jj = find_nearest_index_in_space(
                        target_lat, target_lon, ds.lat.data, ds.lon.data
                    )
                    if len(ii) > 0:
                        ii = ii[0]
                        jj = jj[0]
                        times.append(ds.time.data[0])
                        cbhs.append(
                            float(
                                ds.cbh_alti[0].data[ii, jj]
                                # + ds.surface_alti[0].data[ii, jj]
                            )
                        )
                        cths.append(
                            float(
                                cth.ctth_alti[0].data[ii, jj]
                                # + cth.surface_alti[0].data[ii, jj]
                            )
                        )
                        time_scan = np.linspace(
                            ds.time_bnds.data[0, 0].astype("int64"),
                            ds.time_bnds.data[0, 1].astype("int64"),
                            len(ds.ny.data),
                        )
                        time_scan = time_scan.astype("datetime64[ns]")
                        index = find_nearest_index_in_time(cm.time.data, time_scan[ii])
                        cbhs_cm.append(float(cm.cloud_base_height_amsl.data[index]))
                        cths_cm.append(float(cm.cloud_top_height_amsl.data[index]))
                        sat_name = re.split(r"_", os.path.basename(file))[3]
                        sat.append(sat_name)
                        with xr.open_dataset(imagerfile) as im:
                            sunzenith.append(im.sunzenith.data[0, ii, jj])
    return cbhs, cbhs_cm, cths, cths_cm, sunzenith, times, sat


# %%
def run_process(station, date):
    cfile = f"/home/sm_indka/data/Celiometer/{date}_{station}_classification.nc"
    cm = xr.open_dataset(cfile)
    cm = cm.resample(time="2min").mean()

    viirsfiles = glob.glob(
        f"/nobackup/smhid20/proj/foua/data/NWCSAF/CBH_FMI_MAR25/FMI_CBH_PPS/*CBH*{date}*nc"
    )
    return collect_collocated_data(cm, viirsfiles)


def plot_data(station, data):
    fig, ax = plt.subplots(1, 1, figsize=[12, 6])

    lidarfiles = []
    for date in dates:
        lidarfiles.append(
            f"/home/sm_indka/data/Celiometer/{date}_{station}_classification.nc"
        )
    with xr.open_mfdataset(lidarfiles, combine="by_coords") as cm:
        # # norm = mcolors.LogNorm(vmin=1e-7, vmax=1e-4)
        # # with xr.open_dataset(betafile) as ds:
        # #     ds.beta_smooth.T.plot.pcolormesh(ax=ax, norm=norm)
        # with xr.open_dataset(lidarfile) as cm:
        cm = cm.resample(time="5min").mean()
        ax.scatter(
            cm.time.data,
            cm.cloud_base_height_amsl.data,
            s=3,
            c="g",
            label="CBH Lidar",
        )
        ax.scatter(
            cm.time.data,
            cm.cloud_top_height_amsl.data,
            s=3,
            c="b",
            label="CTH Lidar",
        )
        ax.scatter(
            data[station]["time"],
            data[station]["cbh_sat"],
            s=10,
            c="r",
            label="CBH PPS",
        )
        ax.scatter(
            data[station]["time"],
            data[station]["cth_sat"],
            s=10,
            c="k",
            label="CTH PPS",
        )
        # ax.set_ylim([0, 8000])
    ax.legend()
    ax.set_ylabel("Height [m]")
    ax.set_xlabel("Time")
    fig.suptitle(station)
    fig.savefig(f"{station}_all_amsl.png")


# %%

cbh_all_days = OrderedDict()
cbh_all_days_cm = OrderedDict()
cth_all_days = OrderedDict()
time_all_days = OrderedDict()

# %%
stations = [
    "bucharest",
    "cabauw",
    "cluj",
    "galati",
    "granada",
    "hyytiala",
    "juelich",
    # "lampedusa",
    "leipzig",
    "limassol",
    "lindenberg",
    "munich",
    "norunda",
    "ny-alesund",
    "palaiseau",
    "payerne",
    "potenza",
]

dates = ["20250309", "20250310", "20250311"]

data = {
    station: {
        "cbh_sat": [],
        "cth_sat": [],
        "cbh_cm": [],
        "cth_cm": [],
        "sunzenith": [],
        "time": [],
        "sat": [],
    }
    for station in stations
}

for station in stations:
    print(f"Doing {station}")
    for date in dates:
        print(f"Doing day {date}")
        cbh_sat, cbh_cm, cth_sat, cth_cm, sunzenith, time, sat_name = run_process(
            station, date
        )
        data[station]["cbh_sat"].append(cbh_sat)
        data[station]["cbh_cm"].append(cbh_cm)
        data[station]["cth_sat"].append(cth_sat)
        data[station]["cth_cm"].append(cth_cm)
        data[station]["time"].append(time)
        data[station]["sunzenith"].append(sunzenith)
        data[station]["sat"].append(sat_name)

    for key in ["cbh_sat", "cbh_cm", "cth_sat", "cth_cm", "sunzenith", "time", "sat"]:
        data[station][key] = np.concatenate(data[station][key])
    plot_data(station, data)

with open("collocated_data_resampled_allfiles.pickle", "wb") as f:
    pickle.dump(data, f)
