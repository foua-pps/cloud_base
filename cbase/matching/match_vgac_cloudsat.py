from dataclasses import dataclass
import numpy as np
from matplotlib.pyplot import plt
from data_readers.viirs import VGACData
from data_readers.cloudsat import CloudsatData


class DataMatcher:
    """ """

    def __init__(self, cloudsat: CloudsatData, vgac: VGACData):
        self.cloudsat = cloudsat
        self.vgac = vgac

        tlist, indices = self.get_overlapping_time()
        self.indices = indices
        if len(tlist) == 0:
            raise ValueError("The two passes are not at same time")

    def get_overlapping_time(self) -> tuple[list, list]:
        """
        check if the two satellites passes are overlapping
        """
        tlist = []
        indices = []
        t1 = self.cloudsat.time[0]  # start time
        t2 = self.cloudsat.time[-1]  # end time
        if t1 > t2:
            raise ValueError("start time cannot be after end time")
        for index, dt in enumerate(self.vgac.time):
            if t1 <= dt <= t2:
                tlist.append(dt)
                indices.append(index)

        return tlist, indices
