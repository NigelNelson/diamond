import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
from .segment import Segment, SegmentId
from .utils import rotation_matrix_to_quaternion, batch_rotation_matrix_to_quaternion


class GameDataset(Dataset):
    def __init__(self, zarr_path, context_length, image_size, validation=False):
        self.zarr_path = zarr_path
        self.context_length = context_length
        self.image_size = image_size
        assert len(self.image_size) == 2, "Image size must be a tuple of (height, width)"
        
        # Open zarr store to get metadata
        with zarr.open(self.zarr_path, mode='r') as root:
            self.total_frames = root['frames'].shape[0]
            self.episode_ends = root['episode_ends'][:]
        
        if validation:
            self.episode_starts = [self.episode_ends[-2] + 1]
            self.episode_ends = self.episode_ends[-1:]
        else:
            self.episode_ends = self.episode_ends[:-1]
            self.episode_starts = [0] + [self.episode_ends[i] for i in range(len(self.episode_ends) - 1)]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(2).sub(1)),
        ])

        self.root = zarr.open(self.zarr_path, mode='r')
        self.frames = self.root['frames']
        self.kinematics = self.root['kinematics']
        self.num_steps = len(self.episode_ends) * 8
        self.num_episodes = len(self.episode_ends)

        self.kinematics_indices = [0, 1, 2,  # Left MTM tool tip position (xyz)
                                   3, 4, 5, 6, 7, 8, 9, 10, 11,  # Left MTM rotation matrix
                                   19,  # Left MTM gripper angle velocity
                                   20, 21, 22,  # Right MTM tool tip position (xyz)
                                   23, 24, 25, 26, 27, 28, 29, 30, 31,  # Right MTM rotation matrix
                                   38]  # Right MTM gripper angle velocity
        self.kinematics_mean, self.kinematics_std = self.get_norm_stats()

    def __len__(self):
        return len(self.episode_ends) * 8

    def __getitem__(self, idx):
        idx = idx // 8
        kinematics = self.kinematics

        # Find the start and end of the current episode
        episode_start = self.episode_starts[idx]
        episode_end = self.episode_ends[idx]
        
        # Randomly sample an index from the chosen episode
        sampled_idx = random.randint(episode_start, episode_end)
        
        # Calculate the actual context start, allowing for zero-padding if necessary
        context_start = max(episode_start, sampled_idx - self.context_length + 1)
        
        # Load actual context data
        context_slice = slice(context_start, sampled_idx + 1)
        frames_data = self.frames[context_slice]
        kinematics_data = self.kinematics[context_slice][:, self.kinematics_indices]

        left_rotation_matrices = kinematics_data[:, 3:12].reshape(-1, 3, 3)
        right_rotation_matrices = kinematics_data[:, 15:24].reshape(-1, 3, 3)

        # Convert rotation matrices to quaternions
        left_quaternions = batch_rotation_matrix_to_quaternion(left_rotation_matrices)
        right_quaternions = batch_rotation_matrix_to_quaternion(right_rotation_matrices)

        # Combine position, quaternion, and gripper angle velocity
        kinematics_data = np.concatenate([
            kinematics_data[:, :3],  # Left MTM tool tip position
            left_quaternions,  # Left MTM quaternion
            kinematics_data[:, 12:13],  # Left MTM gripper angle velocity
            kinematics_data[:, 13:16],  # Right MTM tool tip position
            right_quaternions,  # Right MTM quaternion
            kinematics_data[:, 24:25]  # Right MTM gripper angle velocity
        ], axis=1)

        # Calculate padding lengths
        pad_length = max(0, self.context_length - (sampled_idx - context_start + 1))
        actual_data_length = sampled_idx - context_start + 1

        # Create mask_padding
        mask_padding = torch.cat([
            torch.zeros(pad_length, dtype=torch.bool),
            torch.ones(actual_data_length, dtype=torch.bool)
        ])

        # Transform frames
        context_frames = torch.stack([self.transform(frame) for frame in frames_data])

        # Normalize kinematics
        kinematics_tensor = torch.tensor(kinematics_data, dtype=torch.float32)
        normalized_kinematics = (kinematics_tensor - self.kinematics_mean) / self.kinematics_std

        # Combine with zero-padded data if necessary
        if sampled_idx - self.context_length + 1 < context_start:
            pad_length = context_start - (sampled_idx - self.context_length + 1)
            zero_frames = torch.zeros((pad_length, 3, self.image_size[0], self.image_size[1]))
            zero_kinematics = torch.zeros((pad_length, 16))
            
            pixel_values = torch.cat([zero_frames, context_frames])
            kinematics = torch.cat([zero_kinematics, normalized_kinematics])
        else:
            pixel_values = context_frames
            kinematics = normalized_kinematics

        segment_id = SegmentId(idx, context_start, sampled_idx + 1)
        segment = Segment(pixel_values, kinematics, mask_padding, info=None, id=segment_id)

        return segment
    

    def getOrderedSegment(self, idx, chunk_size):
        kinematics = self.kinematics

        # Find the start and end of the current episode
        episode_start = self.episode_starts[idx]
        episode_end = self.episode_ends[idx]

        for i in range(episode_start, episode_end, chunk_size):

            if i + chunk_size > episode_end:
                break  # Skip the last incomplete chunk
        
            # Calculate the actual context start, allowing for zero-padding if necessary
            context_start = i

        # Load actual context data
            context_slice = slice(context_start, i + chunk_size)
            frames_data = self.frames[context_slice]
            kinematics_data = self.kinematics[context_slice][:, self.kinematics_indices]

            # Convert rotation matrices to quaternions
            left_quaternions = np.apply_along_axis(rotation_matrix_to_quaternion, 1, kinematics_data[:, 3:12].reshape(-1, 3, 3))
            right_quaternions = np.apply_along_axis(rotation_matrix_to_quaternion, 1, kinematics_data[:, 15:24].reshape(-1, 3, 3))

            # Combine position, quaternion, and gripper angle velocity
            kinematics_data = np.concatenate([
                kinematics_data[:, :3],  # Left MTM tool tip position
                left_quaternions,  # Left MTM quaternion
                kinematics_data[:, 12:13],  # Left MTM gripper angle velocity
                kinematics_data[:, 13:16],  # Right MTM tool tip position
                right_quaternions,  # Right MTM quaternion
                kinematics_data[:, 24:25]  # Right MTM gripper angle velocity
            ], axis=1)

            # Calculate padding lengths
            pad_length = 0
            actual_data_length = (i + chunk_size) - context_start

            # Create mask_padding
            mask_padding = torch.cat([
                torch.ones(actual_data_length, dtype=torch.bool)
            ])

            # Transform frames
            context_frames = torch.stack([self.transform(frame) for frame in frames_data])

            # Normalize kinematics
            kinematics_tensor = torch.tensor(kinematics_data, dtype=torch.float32)
            normalized_kinematics = (kinematics_tensor - self.kinematics_mean) / self.kinematics_std

            # Combine with zero-padded data if necessary
            # if sampled_idx - self.context_length + 1 < context_start:
            #     pad_length = context_start - (sampled_idx - self.context_length + 1)
            #     zero_frames = torch.zeros((pad_length, 3, self.image_size[0], self.image_size[1]))
            #     zero_kinematics = torch.zeros((pad_length, 38))
                
            #     pixel_values = torch.cat([zero_frames, context_frames])
            #     kinematics = torch.cat([zero_kinematics, normalized_kinematics])
            # else:
            pixel_values = context_frames
            kinematics = normalized_kinematics

            segment_id = SegmentId(idx, context_start, context_start + 1)
            segment = Segment(pixel_values, kinematics, mask_padding, info=None, id=segment_id)

            yield segment

    def get_norm_stats(self):
        kinematics_sum = 0
        kinematics_sq_sum = 0
        total_frames = 0

        with zarr.open(self.zarr_path, mode='r') as root:
            kinematics = root['kinematics']
            
            for episode_start, episode_end in zip(self.episode_starts, self.episode_ends):
                episode_kinematics = kinematics[episode_start:episode_end][:, self.kinematics_indices]
                
                left_rotation_matrices = episode_kinematics[:, 3:12].reshape(-1, 3, 3)
                right_rotation_matrices = episode_kinematics[:, 15:24].reshape(-1, 3, 3)

                # Convert rotation matrices to quaternions
                left_quaternions = batch_rotation_matrix_to_quaternion(left_rotation_matrices)
                right_quaternions = batch_rotation_matrix_to_quaternion(right_rotation_matrices)

                # Combine position, quaternion, and gripper angle velocity
                episode_kinematics = np.concatenate([
                    episode_kinematics[:, :3],  # Left MTM tool tip position
                    left_quaternions,  # Left MTM quaternion
                    episode_kinematics[:, 12:13],  # Left MTM gripper angle velocity
                    episode_kinematics[:, 13:16],  # Right MTM tool tip position
                    right_quaternions,  # Right MTM quaternion
                    episode_kinematics[:, 24:25]  # Right MTM gripper angle velocity
                ], axis=1)

                kinematics_sum += np.sum(episode_kinematics, axis=0)
                kinematics_sq_sum += np.sum(np.square(episode_kinematics), axis=0)
                total_frames += episode_end - episode_start

        kinematics_mean = kinematics_sum / total_frames
        kinematics_std = np.sqrt(kinematics_sq_sum / total_frames - np.square(kinematics_mean))
        kinematics_std = np.clip(kinematics_std, 1e-2, np.inf)  # clipping to avoid very small std

        return torch.tensor(kinematics_mean, dtype=torch.float32), torch.tensor(kinematics_std, dtype=torch.float32)

# Usage example:
# dataset = GameDataset('path/to/your/suturing.zarr', context_length=5, image_size=224)


