from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from .segment import Segment, SegmentId

@dataclass
class Episode:
    obs: torch.FloatTensor
    act: torch.LongTensor
    info: Dict[str, Any]

    def __len__(self) -> int:
        return self.obs.size(0)

    def __add__(self, other: Episode) -> Episode:
        d = {k: torch.cat((v, other.__dict__[k]), dim=0) for k, v in self.__dict__.items() if k != "info"}
        return Episode(**d, info=merge_info(self.info, other.info))

    def to(self, device) -> Episode:
        return Episode(**{k: v.to(device) if k != "info" else v for k, v in self.__dict__.items()})

    def compute_metrics(self) -> Dict[str, Any]:
        return {"length": len(self)}

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        return cls(
            **{
                k: v.div(255).mul(2).sub(1) if k == "obs" else v
                for k, v in torch.load(Path(path), map_location=map_location).items()
            }
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v.add(1).div(2).mul(255).byte() if k == "obs" else v for k, v in self.__dict__.items()}
        torch.save(d, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)


def merge_info(info_a, info_b):
    keys_a = set(info_a)
    keys_b = set(info_b)
    intersection = keys_a & keys_b
    info = {
        **{k: info_a[k] for k in keys_a if k not in intersection},
        **{k: info_b[k] for k in keys_b if k not in intersection},
        **{k: torch.cat((info_a[k], info_b[k]), dim=0) for k in intersection},
    }
    return info


@classmethod
def load_and_make_segment(cls, zarr_root: Any, segment_id: SegmentId, should_pad: bool = True, map_location: Optional[torch.device] = None) -> Segment:
    # Load the episode data
    episode_data = torch.load(Path(path), map_location=map_location)
    
    # Process the loaded data
    processed_data = {
        k: v.div(255).mul(2).sub(1) if k == "obs" else v
        for k, v in episode_data.items()
    }
    
    # Create a temporary Episode object
    temp_episode = cls(**processed_data)
    
    # Make the segment
    assert segment_id.start < len(temp_episode) and segment_id.stop > 0 and segment_id.start < segment_id.stop
    pad_len_right = max(0, segment_id.stop - len(temp_episode))
    pad_len_left = max(0, -segment_id.start)
    assert pad_len_right == pad_len_left == 0 or should_pad

    def pad(x):
        right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
        return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

    start = max(0, segment_id.start)
    stop = min(len(temp_episode), segment_id.stop)
    mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

    return Segment(
        pad(temp_episode.obs[start:stop]),
        pad(temp_episode.act[start:stop]),
        pad(temp_episode.rew[start:stop]),
        pad(temp_episode.end[start:stop]),
        pad(temp_episode.trunc[start:stop]),
        mask_padding,
        info=temp_episode.info,
        id=SegmentId(segment_id.episode_id, start, stop),
    )
