from cbase.data_readers import cloudsat, viirs, era5
from cbase.matching.match_vgac_cloudsat_nwp import DataMatcher


filename = "/home/a002602/data/cloud_base/cloudsat/2018150015649_64371_CS_2B-GEOPROF_GRANULE_P1_R05_E07_F03.hdf"
vgac_file = (
    "/home/a002602/data/cloud_base/vgac/VGAC_VJ102MOD_A2018150_0130_n002738_K005.nc"
)
era5_file = "/home/a002602/data/cloud_base/NWP/GAC_ECMWF_ERA5_201801010100+000H00M"

filename1 = "/home/a002602/data/cloud_base/cloudsat/2018150033525_64372_CS_2B-GEOPROF_GRANULE_P1_R05_E07_F03.hdf"
vgac_file1 = (
    "/home/a002602/data/cloud_base/vgac/VGAC_VJ102MOD_A2018150_0312_n002739_K005.nc"
)


cld = cloudsat.CloudsatData.from_file(filename)

vgc = viirs.VGACData.from_file(vgac_file)
nwp = era5.Era5.from_file(era5_file)

dm = DataMatcher(cld, vgc, nwp)

dm.match_vgac_cloudsat()

dm.create_cnn_dataset_with_nwp()

# check data
