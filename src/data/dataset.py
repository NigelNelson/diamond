from collections import Counter
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment
from utils import StateDictMixin

import zarr

class Dataset(StateDictMixin, torch.utils.data.Dataset):
    def __init__(
        self,
        directory: Path,
        name: Optional[str] = None,
        cache_in_ram: bool = False,
        use_manager: bool = False,
        save_on_disk: bool = True,
        zarr_path: Optional[Path] = None,
    ) -> None:
        super().__init__()

        # zarr
        self.zarr_path = zarr_path
        self.zarr_dataset = zarr.open(self.zarr_path, mode='r')

        self.kinematics_mean, self.kinematics_std = self.get_norm_stats()
        self.root = zarr.open(self.zarr_path, mode='r')
        self.frames = self.root['frames']
        self.kinematics = self.root['kinematics']
        self.episode_ends = self.root['episode_ends'][:]


        # State
        self.is_static = False
        self.num_episodes = None
        self.num_steps = None
        self.start_idx = None
        self.lengths = None

        self._directory = Path(directory).expanduser()
        self._name = name if name is not None else self._directory.stem
        self._cache_in_ram = cache_in_ram
        self._save_on_disk = save_on_disk
        self._default_path = self._directory / "info.pt"
        self._cache = mp.Manager().dict() if use_manager else {}
        self._reset()

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        episode = self.load_episode(segment_id.episode_id)
        return make_segment(episode, segment_id, should_pad=True)

    def __str__(self) -> str:
        return f"{self.name}: {self.num_episodes} episodes, {self.num_steps} steps."

    @property
    def name(self) -> str:
        return self._name

    @property
    def counts_end(self) -> List[int]:
        return [self.counter_end[e] for e in [0, 1]]

    def _reset(self) -> None:
        self.num_episodes = 0
        self.num_steps = 0
        self.start_idx = np.array([], dtype=np.int64)
        self.lengths = np.array([], dtype=np.int64)
        self._cache.clear()

    def clear(self) -> None:
        self.assert_not_static()
        if self._directory.is_dir():
            shutil.rmtree(self._directory)
        self._reset()

    def load_episode(self, episode_id: int) -> Episode:
        episode_start = self.episode_starts[episode_id] - 1 if episode_id > 0 else 0
        episode_end = self.episode_ends[episode_id]
        episode = Episode.load_and_make_segment(
            self.root,
            segment_id=SegmentId(episode_id, episode_start, episode_end),
            should_pad=True,
            map_location=None
        )
        if self._cache_in_ram and episode_id in self._cache:
            episode = self._cache[episode_id]
        else:
            episode = Episode.load(self._get_episode_path(episode_id))
            if self._cache_in_ram:
                self._cache[episode_id] = episode
        return episode

    def add_episode(self, episode: Episode, *, episode_id: Optional[int] = None) -> int:
        self.assert_not_static()
        episode = episode.to("cpu")

        if episode_id is None:
            episode_id = self.num_episodes
            self.start_idx = np.concatenate((self.start_idx, np.array([self.num_steps])))
            self.lengths = np.concatenate((self.lengths, np.array([len(episode)])))
            self.num_steps += len(episode)
            self.num_episodes += 1

        else:
            assert episode_id < self.num_episodes
            old_episode = self.load_episode(episode_id)
            incr_num_steps = len(episode) - len(old_episode)
            self.lengths[episode_id] = len(episode)
            self.start_idx[episode_id + 1 :] += incr_num_steps
            self.num_steps += incr_num_steps

        if self._save_on_disk:
            episode.save(self._get_episode_path(episode_id))

        if self._cache_in_ram:
            self._cache[episode_id] = episode

        return episode_id

    def _get_episode_path(self, episode_id: int) -> Path:
        n = 3  # number of hierarchies
        powers = np.arange(n)
        subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
        subfolders = [int(x) for x in subfolders[::-1]]
        subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
        return self._directory / subfolders / f"{episode_id}.pt"

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._cache.clear()

    def assert_not_static(self) -> None:
        assert not self.is_static, "Trying to modify a static dataset."

    def save_to_default_path(self) -> None:
        self._default_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), self._default_path)

    def load_from_default_path(self) -> None:
        if self._default_path.is_file():
            self.load_state_dict(torch.load(self._default_path))


    def get_norm_stats(self):
        kinematics_sum = 0
        kinematics_sq_sum = 0
        total_frames = 0

        with zarr.open(self.zarr_path, mode='r') as root:
            kinematics = root['kinematics']
            
            for episode_start, episode_end in zip(self.episode_starts, self.episode_ends):
                episode_kinematics = kinematics[episode_start:episode_end, :38]
                kinematics_sum += np.sum(episode_kinematics, axis=0)
                kinematics_sq_sum += np.sum(np.square(episode_kinematics), axis=0)
                total_frames += episode_end - episode_start

        kinematics_mean = kinematics_sum / total_frames
        kinematics_std = np.sqrt(kinematics_sq_sum / total_frames - np.square(kinematics_mean))
        kinematics_std = np.clip(kinematics_std, 1e-2, np.inf)  # clipping to avoid very small std

        return torch.tensor(kinematics_mean, dtype=torch.float32), torch.tensor(kinematics_std, dtype=torch.float32)
