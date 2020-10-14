
from .mot_wrapper import MOT17Wrapper, MOT19Wrapper, MOT17LOWFPSWrapper, MOT20Wrapper
from .mot_reid_wrapper import MOTreIDWrapper
from .mot15_wrapper import MOT15Wrapper
from .marcuhmot import MarCUHMOT


_sets = {}


# Fill all available datasets, change here to modify / add new datasets.
for split in ['train']:
    for dets in ['FRCNN17']:
        name = f'mot17_{split}_{dets}'
        _sets[name] = (lambda split=split,
                       dets=dets: MOT17Wrapper(split, dets))

class Datasets(object):
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, dataset):
        """Initialize the corresponding dataloader.

        Keyword arguments:
        dataset --  the name of the dataset
        args -- arguments used to call the dataloader
        """
        assert dataset in _sets, "[!] Dataset not found: {}".format(dataset)

        self._data = MOT17Wrapper('train', 'FRCNN17')

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
