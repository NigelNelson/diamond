import math
from typing import Generator, List

import torch
import torch.nn.functional as F

from .batch import Batch
from .episode import Episode
from .segment import Segment, SegmentId
import numpy as np


def collate_segments_to_batch(segments: List[Segment]) -> Batch:
    attrs = ("obs", "act", "mask_padding")
    stack = (torch.stack([getattr(s, x) for s in segments]) for x in attrs)
    return Batch(*stack, [s.info for s in segments], [s.id for s in segments])


def make_segment(episode: Episode, segment_id: SegmentId, should_pad: bool = True) -> Segment:
    assert segment_id.start < len(episode) and segment_id.stop > 0 and segment_id.start < segment_id.stop
    pad_len_right = max(0, segment_id.stop - len(episode))
    pad_len_left = max(0, -segment_id.start)
    assert pad_len_right == pad_len_left == 0 or should_pad

    def pad(x):
        right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
        return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

    start = max(0, segment_id.start)
    stop = min(len(episode), segment_id.stop)
    mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

    return Segment(
        pad(episode.obs[start:stop]),
        pad(episode.act[start:stop]),
        pad(episode.rew[start:stop]),
        pad(episode.end[start:stop]),
        pad(episode.trunc[start:stop]),
        mask_padding,
        info=episode.info,
        id=SegmentId(segment_id.episode_id, start, stop),
    )


class DatasetTraverser:
    def __init__(self, dataset, batch_num_samples: int, chunk_size: int) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.chunk_size = chunk_size

    def __len__(self):
        return math.ceil(
            sum(
                [
                    math.ceil(self.dataset.lengths[episode_id] / self.chunk_size)
                    - int(self.dataset.lengths[episode_id] % self.chunk_size == 1)
                    for episode_id in range(self.dataset.num_episodes)
                ]
            )
            / self.batch_num_samples
        )

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_episode(episode_id)
            chunks.extend(
                make_segment(
                    episode,
                    SegmentId(episode_id, start=i * self.chunk_size, stop=(i + 1) * self.chunk_size),
                    should_pad=True,
                )
                for i in range(math.ceil(len(episode) / self.chunk_size))
            )
            if chunks[-1].effective_size < 2:
                chunks.pop()

            while len(chunks) >= self.batch_num_samples:
                yield collate_segments_to_batch(chunks[: self.batch_num_samples])
                chunks = chunks[self.batch_num_samples :]

        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)



class GameDatasetTraverser:
    def __init__(self, dataset, batch_num_samples: int, chunk_size: int) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.chunk_size = chunk_size

    def __len__(self):
        length = math.ceil(
            sum(
                [
                    math.ceil((self.dataset.episode_ends[-1] - self.dataset.episode_starts[0]) / self.chunk_size)
                ]
            )
            / self.batch_num_samples
        )
        length = 1
        return length

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.dataset.num_episodes):
            chunks.extend(
                i
                for i in self.dataset.getOrderedSegment(episode_id, self.chunk_size)
            )
            if chunks[-1].effective_size < 2:
                chunks.pop()

            while len(chunks) >= self.batch_num_samples:
                yield collate_segments_to_batch(chunks[: self.batch_num_samples])
                chunks = chunks[self.batch_num_samples :]

        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)

def batch_rotation_matrix_to_quaternion(R_batch):
    """
    Convert a batch of 3x3 rotation matrices to quaternions.
    
    Args:
    R_batch: Array of shape (N, 3, 3) where N is the batch size
    
    Returns:
    q_batch: Array of shape (N, 4) containing the quaternions
    """
    return np.array([rotation_matrix_to_quaternion(R) for R in R_batch])

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Args:
    R: 3x3 rotation matrix
    
    Returns:
    q: 4D quaternion [w, x, y, z]
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    return np.array([w, x, y, z])
