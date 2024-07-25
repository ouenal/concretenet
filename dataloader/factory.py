# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from dataloader.scannet import ScanNet
from dataloader.scanrefer import ScanRefer


class DatasetFactory:
    _registry = {
        'scannet': ScanNet,
        'scanrefer': ScanRefer,
    }
    def __new__(cls, prefix:str, **kwargs):
        return cls._registry[prefix](**kwargs)
