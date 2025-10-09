from .coco import COCO
from .openimages import OpenImages
from .perseg import PerSegDataset, EpisodicPerSeg
from .poddataset import PODDataset, EpisodicPOD
from .core50dataset import CORe50Dataset, EpisodicCORe50
from .icubworlddataset import iCubWorldDataset, EpisodiciCubWorld

__all__ = [
    "COCO",
    "OpenImages",
    "PerSegDataset", "EpisodicPerSeg",
    "PODDataset", "EpisodicPOD",
    "CORe50Dataset", "EpisodicCORe50",
    "iCubWorldDataset", "EpisodiciCubWorld"
]
