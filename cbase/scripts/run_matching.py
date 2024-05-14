from cbase.data_readers import cloudsat, viirs, era5
from cbase.matching.match_vgac_cloudsat_nwp import DataMatcher


filename = (
        "/nobackup/smhid19/proj/foua/data/satellite/cloudsat/2B_GEORPOF_V5/2018/2018150015649_64371_CS_2B-GEOPROF_GRANULE_P1_R05_E07_F03.hdf"
        )

vgac_file = (
    "/nobackup/smhid17/proj/foua/data/satellit/VGAC/2018/05/2018VGAC_VJ102MOD_A2018150_0130_n002738_K005.nc"
)
era5_file = "/nobackup/smhid20/proj/safnwccm/data/nwp/ERA5/2018/05/GAC_ECMWF_ERA5_201805010000+000H00M"



cld = cloudsat.CloudsatData.from_file(filename)

vgc = viirs.VGACData.from_file(vgac_file)
nwp = era5.Era5.from_file(era5_file)

dm = DataMatcher(cld, vgc, nwp)

dm.match_vgac_cloudsat()

dm.create_cnn_dataset_with_nwp()

# check data
