"""Training datasets, sampling, augmentation, normalization, and metric helpers."""
import numpy as np
import torch
import os
from data_processing import episode_store
from data_processing.episode_data import open_episode, find_episodes, is_episode
import pickle
import cv2
from utils.masking import Masker
from contracts import standard_image_processing
from contracts.action_contract import require_absolute_actions, require_absolute_action_style as validate_action_command_style
from contracts.sequence_contract import contiguous_runs, training_runs, run_bounds
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T


def prepare_policy_frame(raw, compressed, width, height):
    """Shared deterministic decode/resize of one exact frame (live path and frame store)."""
    image = cv2.imdecode(raw, 1) if compressed else raw
    if width is not None and height is not None:
        image = standard_image_processing.resize_image(image, width, height)
    return np.array(image) if compressed else image



POLICY_TRAINING_MASK_ATTR = "policy_training_mask_path"


def load_policy_training_mask(root, expected_len, include_masked_segments=False):
    """Return the per-frame policy mask advertised by a route episode.

    Legacy episodes have no mask and remain fully trainable.  Grouped route
    episodes advertise the dataset path in a root attribute so unrelated Zarr
    schemas are not interpreted accidentally.
    """
    expected_len = int(expected_len)
    if include_masked_segments or POLICY_TRAINING_MASK_ATTR not in root.attrs:
        return np.ones(expected_len, dtype=bool)
    mask_path = root.attrs[POLICY_TRAINING_MASK_ATTR]
    if isinstance(mask_path, bytes):
        mask_path = mask_path.decode("utf-8")
    mask_path = str(mask_path)
    if mask_path not in root:
        raise ValueError(
            f"Dataset {root.filename} advertises missing policy mask {mask_path!r}"
        )
    mask = np.asarray(root[mask_path][()], dtype=bool).reshape(-1)
    if len(mask) != expected_len:
        raise ValueError(
            f"Policy mask length mismatch in {root.filename}: "
            f"expected {expected_len}, got {len(mask)}"
        )
    if not mask.any():
        raise ValueError(f"Policy mask excludes every frame in {root.filename}")
    return mask


def flatten_list(l):
    return [item for sublist in l for item in sublist]


class CachedEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, masking_strategy, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, 
                 chunk_size, policy_class,width=None, height=None, normalize_resnet=False,data_aug=False,
                 observation_name=[], outputs_name=[], feature_loss=False,grayscale=False,randomize_color=False,
                 randomize_index=None,randomize_data_degree=0,randomize_data=False, debug_image=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.transformations = None
        self.width = width
        self.height = height
        self.data_aug = data_aug
        self.feature_loss = feature_loss
        self.randomize_data = randomize_data
        self.randomize_data_radian = randomize_data_degree/180*np.pi
        self.randomize_index = randomize_index
        self.grayscale = grayscale
        self.randomize_color = randomize_color
        self.outputs_name = outputs_name if outputs_name else ['action']  # default to action only
        if self.data_aug:
            #has nothing to do with the deployment of the model 
            self.transformations = [
                T.ColorJitter(hue=0.5,saturation=0.5),
                T.RandomRotation(degrees=5, interpolation=T.InterpolationMode.BILINEAR),
                # T.Pad(padding=[int(self.width * 0.05), int(self.height * 0.05)], padding_mode='edge'),
                T.RandomCrop(size=[self.height,self.width])
            ]
        else:
            self.transformations = None
  
        self.normalize_resnet = normalize_resnet
        if self.normalize_resnet:
            #need to normalize the image to the same mean and std as the resnet model during depolyment
            self.normalize_resnet_tf = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        self.observation_name = observation_name
        self.masking_strategy = masking_strategy
        # Cache expensive operations for moving block mask
        self._cached_data = {}
        self._kdl_chain = None
        self._projection_matrices = {}
        
        self.masker = Masker(masking_strategy=self.masking_strategy)
        self.debug_image, _, _, _ = self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, masking_strategy, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len,
                 chunk_size, policy_class, width=None, height=None, normalize_resnet=False, data_aug=False,
                 observation_name=[], outputs_name=['action'], feature_loss=False, grayscale=False, randomize_color=False,
                 randomize_index=None, randomize_data_degree=0, randomize_data=False, debug_image=None, history_len=0, history_noise=0, noise_type='gaussian', aloha=False, image_history_len=0,
                 image_history_stride=1, use_raw_future_actions=False,
                 action_command_style='absolute', include_masked_segments=False, task_boundary_provider=None,
                 frame_store_dir=None, frame_store_workers=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.include_masked_segments = bool(include_masked_segments)
        self._policy_start_indices = {}
        effective_episode_len = []
        for episode_id in self.episode_ids:
            with open_episode(self.dataset_path_list[int(episode_id)]) as root:
                require_absolute_actions(root.attrs)
                full_len = len(root['/action'])
                mask = load_policy_training_mask(
                    root,
                    full_len,
                    include_masked_segments=self.include_masked_segments,
                )
            valid_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
            self._policy_start_indices[int(episode_id)] = valid_indices
            effective_episode_len.append(len(valid_indices))
        self.episode_len = effective_episode_len
        standard_image_processing.require_single_frame({'image_history_len': image_history_len})
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.transformations = None
        self.width = width
        self.height = height
        self.data_aug = data_aug
        self.feature_loss = feature_loss
        self.randomize_data = randomize_data
        self.randomize_data_radian = randomize_data_degree/180*np.pi
        self.randomize_index = randomize_index
        self.grayscale = grayscale
        self.randomize_color = randomize_color
        self.history_len = history_len
        self.noise_type = noise_type
        self.aloha = aloha
        self.task_boundary_provider = task_boundary_provider
        self.use_raw_future_actions = use_raw_future_actions
        self.action_command_style = validate_action_command_style(action_command_style)

        
        if self.data_aug:
            original_size = (self.height, self.width)
            ratio = 0.95
            self.transformations = T.Compose([
                T.ToImage(),
                T.RandomCrop(size=[int(ratio*self.height), int(ratio*self.width)]),
                T.Resize(size=[self.height, self.width])
            ])
        else:
            self.transformations = None
  
        # self.normalize_resnet = normalize_resnet
        # if self.normalize_resnet:
        #     #need to normalize the image to the same mean and std as the resnet model during depolyment
        #     self.normalize_resnet_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        self.observation_name = observation_name
        self.outputs_name = list(outputs_name)  # e.g., ['action'] or ['action', 'qvel'] for training to output velocity too
        if self.use_raw_future_actions and self.outputs_name != ['action']:
            raise ValueError(
                "Raw future action mode requires outputs_name=['action'] "
                f"but got {self.outputs_name}"
            )
        self.masking_strategy = masking_strategy
        # Cache expensive operations for moving block mask.
        self._cached_data = {}
        self._kdl_chain = None
        self._projection_matrices = {}

        self.history_noise = history_noise
        
        self.masker = Masker(masking_strategy=self.masking_strategy)
        self.frame_store = None
        if frame_store_dir is not None:
            if self.feature_loss:
                raise ValueError('The frame store does not cover feature-loss future frames')
            from policy.frame_store import build_frame_store
            self.frame_store = build_frame_store(self.dataset_path_list, self._policy_start_indices,
                                                 self.camera_names, self.width, self.height,
                                                 directory=frame_store_dir, workers=frame_store_workers)
        self.debug_image, _, _, _ = self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

        
    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        policy_offset = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = int(self.episode_ids[episode_index])
        start_ts = int(self._policy_start_indices[episode_id][policy_offset])
        return episode_id, start_ts

    def _load_mask_qpos(self, root, ts):
        observations = root.get('/observations')
        if observations is not None and 'qpos' in observations:
            return torch.from_numpy(observations['qpos'][ts]).float()
        return None

    def _load_raw_future_action_chunk(self, root, start_ts, action_dim):
        """Load target action chunk from raw high-rate action stream."""
        required = ('action_raw', 'action_raw_timestamps', 'action_timestamps')
        missing = [name for name in required if name not in root]
        if missing:
            raise ValueError(
                f"Dataset {root.filename} missing raw action fields {missing}. "
                "Use a raw episode or a legacy export containing its raw action stream."
            )

        action_raw = root['/action_raw'][()]
        action_raw_timestamps = root['/action_raw_timestamps'][()]
        action_timestamps = root['/action_timestamps'][()]

        if action_raw.ndim != 2:
            raise ValueError(f"/action_raw must be 2D in {root.filename}, got shape {action_raw.shape}")
        if action_raw.shape[1] != action_dim:
            raise ValueError(
                f"/action_raw dim mismatch in {root.filename}: expected {action_dim}, got {action_raw.shape[1]}"
            )
        if len(action_raw) != len(action_raw_timestamps):
            raise ValueError(
                f"/action_raw and /action_raw_timestamps length mismatch in {root.filename}: "
                f"{len(action_raw)} vs {len(action_raw_timestamps)}"
            )
        if start_ts >= len(action_timestamps):
            raise ValueError(
                f"start_ts={start_ts} out of range for /action_timestamps in {root.filename} "
                f"(len={len(action_timestamps)})"
            )

        padded_action = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
        is_pad = np.ones(self.chunk_size, dtype=np.float32)
        runs = training_runs(root, len(action_timestamps), boundary_provider=self.task_boundary_provider)
        _, run_stop = run_bounds(runs, start_ts)
        if start_ts >= run_stop:
            return padded_action, is_pad
        policy_mask = load_policy_training_mask(
            root,
            len(action_timestamps),
            include_masked_segments=self.include_masked_segments,
        )
        if not policy_mask[start_ts:start_ts + 1].all():
            return padded_action, is_pad

        anchor_ts = action_timestamps[start_ts]
        # Strictly after anchor timestamp.
        raw_start_idx = np.searchsorted(action_raw_timestamps, anchor_ts, side='right')

        if raw_start_idx < len(action_raw):
            raw_stop_idx = raw_start_idx + self.chunk_size
            raw_runs = training_runs(root, len(action_raw), raw=True, boundary_provider=self.task_boundary_provider)
            # The anchor belongs to the current run. A strictly-future target
            # after its last sample must not jump to a resumed recording.
            anchor_index = raw_start_idx - 1
            if anchor_index >= 0:
                _, raw_run_stop = run_bounds(raw_runs, anchor_index)
                raw_stop_idx = min(raw_stop_idx, raw_run_stop)
            if run_stop < len(action_timestamps):
                raw_stop_idx = min(raw_stop_idx, int(np.searchsorted(
                    action_raw_timestamps, action_timestamps[run_stop], side='left')))
            masked_after_start = np.flatnonzero(~policy_mask[start_ts:])
            if len(masked_after_start):
                boundary_index = start_ts + int(masked_after_start[0])
                boundary_timestamp = action_timestamps[boundary_index]
                raw_stop_idx = min(
                    raw_stop_idx,
                    int(np.searchsorted(action_raw_timestamps, boundary_timestamp, side='left')),
                )
            action_chunk = action_raw[raw_start_idx:raw_stop_idx]
            action_len = len(action_chunk)
            if action_len > 0:
                padded_action[:action_len] = action_chunk
                is_pad[:action_len] = 0
        return padded_action, is_pad

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        if "flipped" in dataset_path:
            flipped_data = True
        else:
            flipped_data = False
        with open_episode(dataset_path) as root:
            require_absolute_actions(root.attrs)
            is_sim = False
            compressed = root.attrs.get('compress', False)
            
            # Load outputs based on outputs_name config
            # outputs_name can be ['action'] (default), ['action', 'qvel'], etc.
            output_data_list = []
            for output_name in self.outputs_name:
                if output_name == 'action':
                    output_data_list.append(root['/action'][()])
                elif output_name in ['qpos', 'qvel']:
                    output_data_list.append(root[f'/observations/{output_name}'][()])
                else:
                    # Try to load from observations
                    output_data_list.append(root[f'/observations/{output_name}'][()])
            action = np.concatenate(output_data_list, axis=-1)
            
            original_action_shape = action.shape
            episode_len = original_action_shape[0]
            policy_mask = load_policy_training_mask(
                root,
                episode_len,
                include_masked_segments=self.include_masked_segments,
            )
            runs = contiguous_runs(np.arange(episode_len), training_runs(root, episode_len, boundary_provider=self.task_boundary_provider), policy_mask)
            run_start, run_stop = run_bounds(runs, start_ts)
            # get observation at start_ts only
            if len(self.observation_name) > 0:
                observation_data = []
                for name in self.observation_name:
                    observation_data.append(root[f'/observations/{name}'][start_ts])
                    # for now, going to make the history+the current observation as qpos
                    if self.history_len > 0:
                        actual_history_len = min(self.history_len, start_ts - run_start)
                        for i in range(actual_history_len):
                            observation_data.append(root[f'/observations/{name}'][start_ts - (i + 1)])
                        for j in range(self.history_len - actual_history_len):
                            observation_data.append(np.zeros_like(observation_data[-1]))
                qpos = np.concatenate(observation_data, axis=-1)
            else:
                qpos = np.zeros([original_action_shape[0],40])[start_ts]
                
            image_dict = dict()
            if self.feature_loss:
                image_dict_future = dict()
            
            stored = self.frame_store is not None
            current_bundle = (root.frame_bundle(start_ts, self.camera_names)
                              if not stored and hasattr(root, 'frame_bundle') else None)
            for cam_name in self.camera_names:
                if stored:
                    # Same exact frame after the same deterministic preprocessing, decoded once.
                    image_dict[cam_name] = self.frame_store.get(episode_id, cam_name, start_ts)
                    continue
                # Raw cameras decode concurrently and reuse the same caches.
                image_dict[cam_name] = (current_bundle[cam_name] if current_bundle is not None
                    else root[f'/observations/images/{cam_name}'][start_ts])
                
                
                if self.feature_loss:
                    masked_after_start = np.flatnonzero(~policy_mask[start_ts:])
                    execution_stop = (
                        start_ts + int(masked_after_start[0])
                        if len(masked_after_start)
                        else episode_len
                    )
                    dummy_index = min(start_ts + self.chunk_size, execution_stop - 1, run_stop - 1)
                    image_dict_future[cam_name] = root[f'/observations/images/{cam_name}'][dummy_index]
                    
            if stored:
                pass
            elif compressed:
                for cam_name in image_dict.keys():
                    image_dict[cam_name] = prepare_policy_frame(image_dict[cam_name], True, self.width, self.height)
                    
                            
                if self.feature_loss:
                    for cam_name in image_dict_future.keys():
                        decompressed_image = cv2.imdecode(image_dict_future[cam_name], 1)
                        if self.width is not None and self.height is not None: 
                            decompressed_image = standard_image_processing.resize_image(decompressed_image, self.width, self.height)
                        image_dict_future[cam_name] = np.array(decompressed_image)

            else:
                for cam_name in image_dict.keys():
                    image_dict[cam_name] = prepare_policy_frame(image_dict[cam_name], False, self.width, self.height)
                    
                    # Resize historical images

            if self.use_raw_future_actions:
                padded_action, is_pad = self._load_raw_future_action_chunk(
                    root, start_ts, original_action_shape[1]
                )
            else:
                if (
                    start_ts < run_stop
                    and policy_mask[start_ts:start_ts + 1].all()
                ):
                    action = action[start_ts:]
                    masked_after_start = np.flatnonzero(~policy_mask[start_ts:])
                    action_len = (
                        int(masked_after_start[0])
                        if len(masked_after_start)
                        else episode_len - start_ts
                    )
                else:
                    action = action[:0]
                    action_len = 0

                action_len = min(action_len, self.chunk_size, max(0, run_stop - start_ts))
                padded_action = np.zeros((self.chunk_size, original_action_shape[1]), dtype=np.float32)
                if action_len > 0:
                    padded_action[:action_len] = action[:action_len]
                is_pad = np.ones(self.chunk_size)
                is_pad[:action_len] = 0


        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()        
        
        # Apply the training mask to the policy-resolution images and the raw qpos, as the direct runner does.
        image_dict, qpos_data = self.masker.mask(image_dict, qpos_data)

        if self.randomize_data:
            #randomize the action data
            randomize_amount = torch.rand(len(self.randomize_index))*2*self.randomize_data_radian - self.randomize_data_radian
            action_data[:,self.randomize_index] += randomize_amount
            qpos_data[self.randomize_index] += randomize_amount
            
        # Normalize action and qpos data
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        if not self.masker.masks_proprioception:
            # Proprioceptive masks zero the qpos; it is then not normalized.
            qpos_data = normalize_with_history(qpos_data, 
                                             torch.from_numpy(self.norm_stats["qpos_mean"]).float(),
                                             torch.from_numpy(self.norm_stats["qpos_std"]).float(), 
                                             self.history_len)

        if self.history_noise > 0:
            base_dim = len(self.norm_stats["qpos_mean"])
            qpos_data[base_dim:] = add_history_noise(
                qpos_data[base_dim:],
                self.history_len,
                self.history_noise,
                self.noise_type,
            )

        # Deterministic model input processing is identical to deployment.
        # Shared deterministic resize has already run above, as on the direct runner.
        image_spec = standard_image_processing.ImageSpec(
            color="BGR" if self.aloha else "RGB", grayscale=self.grayscale)
        for cam_name in image_dict:
            current = standard_image_processing.image_to_chw(image_dict[cam_name], image_spec)
            # Training-only augmentation starts AFTER deterministic preprocessing.
            order = np.random.permutation(3) if self.randomize_color else None
            def augment(frame):
                if flipped_data:
                    frame = np.flip(frame, axis=-1).copy()
                if order is not None:
                    frame = frame[order]
                if self.data_aug:
                    frame = self.transformations(torch.from_numpy(frame).float())
                return frame
            image_dict[cam_name] = augment(current)


         # new axis for different cameras
        all_cam_images = []

        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
                
        all_cam_images = np.stack(all_cam_images, axis=0, dtype=np.float32)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        
        return image_data, qpos_data, action_data, is_pad
    
    
class RawFutureActionEpisodicDataset(EpisodicDataset):
    """Dataset variant that predicts chunked future actions from raw high-rate action stream."""

    def __init__(self, *args, **kwargs):
        kwargs['use_raw_future_actions'] = True
        super().__init__(*args, **kwargs)


def _load_action_array_for_outputs(root, outputs_name):
    output_data_list = []
    for output_name in outputs_name:
        if output_name == 'action':
            output_data_list.append(root['/action'][()])
        elif output_name in ['qpos', 'qvel']:
            output_data_list.append(root[f'/observations/{output_name}'][()])
        else:
            output_data_list.append(root[f'/observations/{output_name}'][()])
    return np.concatenate(output_data_list, axis=-1)


def get_norm_stats(dataset_path_list, observation_name=['qpos'], outputs_name=['action'],
                   use_raw_future_actions=False, chunk_size=None, action_command_style='absolute',
                   include_masked_segments=False, task_boundary_provider=None):
    action_command_style = validate_action_command_style(action_command_style)
    from contracts.robot_contract import dataset_robot_contract
    robot_contract = dataset_robot_contract(dataset_path_list, required=False)

    all_qpos_data = []
    all_action_data = []
    all_action_v_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with open_episode(dataset_path) as root:
                require_absolute_actions(root.attrs)
                if use_raw_future_actions:
                    required = ('action_raw', 'action_raw_timestamps', 'action_timestamps', 'action')
                    missing = [name for name in required if name not in root]
                    if missing:
                        raise ValueError(
                            f"Dataset {dataset_path} missing raw-action fields {missing}. "
                            "Use a raw episode or a legacy export containing its raw action stream."
                        )
                    action = root['/action_raw'][()]
                    action_len_for_episode = len(root['/action'][()])
                    raw_ts = root['/action_raw_timestamps'][()]
                    aligned_ts = root['/action_timestamps'][()]
                    if len(raw_ts) != len(action):
                        raise ValueError(
                            f"Raw action timestamp length mismatch in {dataset_path}: "
                            f"{len(raw_ts)} vs {len(action)}"
                        )
                    if len(aligned_ts) != action_len_for_episode:
                        raise ValueError(
                            f"Aligned action timestamp length mismatch in {dataset_path}: "
                            f"{len(aligned_ts)} vs {action_len_for_episode}"
                        )
                else:
                    action = _load_action_array_for_outputs(root, outputs_name)
                    action_len_for_episode = len(action)

                policy_mask = load_policy_training_mask(
                    root,
                    action_len_for_episode,
                    include_masked_segments=include_masked_segments,
                )

                if use_raw_future_actions:
                    aligned_index = np.searchsorted(aligned_ts, raw_ts, side='right') - 1
                    raw_policy_mask = np.zeros(len(raw_ts), dtype=bool)
                    in_range = (aligned_index >= 0) & (aligned_index < len(policy_mask))
                    raw_policy_mask[in_range] = policy_mask[aligned_index[in_range]]
                    velocity_mask = raw_policy_mask[1:] & raw_policy_mask[:-1]
                    runs = training_runs(root, len(action), raw=True, boundary_provider=task_boundary_provider)
                    velocity_mask &= runs[1:] == runs[:-1]
                    action_velocity = (action[1:] - action[:-1])[velocity_mask]
                    action = action[raw_policy_mask]
                else:
                    velocity_mask = policy_mask[1:] & policy_mask[:-1]
                    runs = training_runs(root, len(action), boundary_provider=task_boundary_provider)
                    velocity_mask &= runs[1:] == runs[:-1]
                    action_velocity = (action[1:] - action[:-1])[velocity_mask]
                    action = action[policy_mask]
                
                original_action_shape = action.shape
                try:
                    observation_data = []
                    for name in observation_name:
                        observation_data.append(root[f'/observations/{name}'][()])
                    qpos = np.concatenate(observation_data, axis=-1)
                except Exception:
                    if is_episode(dataset_path):
                        raise
                    qpos = np.zeros([action_len_for_episode,40])
                if len(qpos) != action_len_for_episode:
                    raise ValueError(
                        f"Observation length mismatch in {dataset_path}: "
                        f"expected {action_len_for_episode}, got {len(qpos)}"
                    )
                qpos = qpos[policy_mask]

      
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            raise
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        if action_velocity is not None and len(action_velocity) > 0:
            all_action_v_data.append(torch.from_numpy(action_velocity))
        else:
            all_action_v_data.append(torch.zeros((1, action.shape[-1]), dtype=torch.float32))
        all_episode_len.append(int(policy_mask.sum()))
    
    all_action_data = torch.cat(all_action_data, dim=0)
    all_action_v_data = torch.cat(all_action_v_data, dim=0)
    
    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping    

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    
    all_action_v_max = all_action_v_data.max(dim=0).values.float()
    all_action_v_min = all_action_v_data.min(dim=0).values.float()
    

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
        
    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
            "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
            "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
            "example_qpos": qpos,
            "all_action_v_max": all_action_v_max.numpy(),
            "all_action_v_min": all_action_v_min.numpy()}
  
    if robot_contract is not None:
        stats['robot_contract'] = robot_contract
    return stats, all_episode_len

def find_all_episodes(dataset_dir, skip_mirrored_data=False):
    return find_episodes(dataset_dir, skip_mirrored_data=skip_mirrored_data)


def BatchSampler(batch_size, episode_len_l, sample_weights):
    episode_len_l = episode_len_l[0] # hotfix for bug. it was only sampling the first element.... and stacking all the rest together....
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(num_dataloader_workers, masking_strategy, dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, 
              chunk_size, skip_mirrored_data=False, load_pretrain=False, 
              policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=1.0,
              width=None, height=None, normalize_resnet=False, data_aug=False, observation_name=[],
              outputs_name=['action'], feature_loss=False, grayscale=False, randomize_color=False,
              randomize_index=None,randomize_data_degree=0,randomize_data=False, history_len=0, history_noise=0, noise_type='gaussian', aloha=False, image_history_len=0,
              image_history_stride=1,
              use_raw_future_actions=False,
              action_command_style='absolute', include_masked_segments=False, task_boundary_provider=None,
              frame_store_dir=None):
    action_command_style = validate_action_command_style(action_command_style)
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = []
    for dataset_dir in dataset_dir_l:
        dataset_paths = find_episodes(dataset_dir, skip_mirrored_data)
        dataset_paths = [path for path in dataset_paths if name_filter(path)]
        dataset_path_list_list.append(sorted(dataset_paths))
    num_episodes_0 = len(dataset_path_list_list[0])
    if num_episodes_0 == 0:
        raise ValueError(
            f"No episodes found in primary dataset {dataset_dir_l[0]} after applying the current filters."
        )
    dataset_path_list = flatten_list(dataset_path_list_list)
    if frame_store_dir is not None:
        # Frames come from the store, so an open episode holds only its validated numeric
        # arrays. Keep every episode open in each worker instead of re-validating camera
        # timing on most random samples (~9x faster per sample). Spawned workers inherit this.
        os.environ.setdefault('FLASH_ALOHA_EPISODE_CACHE_SIZE', str(max(4, len(dataset_path_list))))
    # Preserve root grouping so episode IDs and train/validation splits agree.
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    shuffled_episode_ids_0 = np.arange(num_episodes_0)
    split_idx = int(train_ratio * num_episodes_0)
    if train_ratio >= 1:
        train_episode_ids_0 = shuffled_episode_ids_0.copy()
        val_episode_ids_0 = shuffled_episode_ids_0[:0]
        print("train_ratio=1: training on every episode; validation disabled.")
    elif num_episodes_0 == 1:
        train_episode_ids_0 = shuffled_episode_ids_0.copy()
        val_episode_ids_0 = shuffled_episode_ids_0.copy()
        print(
            "Warning: primary dataset has only 1 episode; reusing it for validation."
        )
    else:
        split_idx = min(max(split_idx, 1), num_episodes_0 - 1)
        train_episode_ids_0 = shuffled_episode_ids_0[:split_idx]
        val_episode_ids_0 = shuffled_episode_ids_0[split_idx:]
    print(f'train_episode_ids_0: {train_episode_ids_0}')
    print(f'val_episode_ids_0: {val_episode_ids_0}')
    
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    if len(val_episode_ids_l[0]) > 0:
        print("test on ", dataset_path_list[val_episode_ids_l[0][0]])
    else:
        print("test on <none>")

    dataset_stats, all_episode_len = get_norm_stats(
        dataset_path_list,
        observation_name=observation_name,
        outputs_name=outputs_name,
        use_raw_future_actions=use_raw_future_actions,
        chunk_size=chunk_size,
        action_command_style=action_command_style,
        task_boundary_provider=task_boundary_provider,
        include_masked_segments=include_masked_segments,
    )

    # Log episode length statistics (supports variable-length episodes)
    min_ep_len = min(all_episode_len)
    max_ep_len = max(all_episode_len)
    mean_ep_len = sum(all_episode_len) / len(all_episode_len)
    if min_ep_len != max_ep_len:
        print(f'Variable episode lengths detected: min={min_ep_len}, max={max_ep_len}, mean={mean_ep_len:.1f}')
    else:
        print(f'Fixed episode length: {min_ep_len}')

    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    stats_paths = flatten_list([find_episodes(path, skip_mirrored_data) for path in stats_dir_l])
    if stats_paths == dataset_path_list:
        norm_stats = dataset_stats
    else:
        norm_stats, _ = get_norm_stats(
            stats_paths,
            observation_name=observation_name,
            outputs_name=outputs_name,
            use_raw_future_actions=use_raw_future_actions,
            chunk_size=chunk_size,
            action_command_style=action_command_style,
            task_boundary_provider=task_boundary_provider,
            include_masked_segments=include_masked_segments,
        )
    if dataset_stats.get('robot_contract') != norm_stats.get('robot_contract'):
        raise ValueError('Normalization data robot model/arm roles differ from training data')
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    dataset_cls = RawFutureActionEpisodicDataset if use_raw_future_actions else EpisodicDataset
    if use_raw_future_actions:
        print("Using raw future action dataset mode")
    train_dataset = dataset_cls(masking_strategy, dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, 
                                policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=data_aug,
                                observation_name=observation_name, outputs_name=outputs_name, feature_loss=feature_loss,grayscale=grayscale,randomize_color=randomize_color,
                                randomize_index=randomize_index,randomize_data_degree=randomize_data_degree,randomize_data=randomize_data, history_len=history_len, history_noise=history_noise, noise_type=noise_type, aloha=aloha, image_history_len=image_history_len,
                                image_history_stride=image_history_stride,
                                action_command_style=action_command_style,
                                task_boundary_provider=task_boundary_provider,
                                include_masked_segments=include_masked_segments,
                                frame_store_dir=frame_store_dir, frame_store_workers=num_dataloader_workers)
    # train_ratio=1 leaves no validation episodes; the dataset cannot be built empty.
    val_dataset = None if len(val_episode_ids) == 0 else dataset_cls(masking_strategy, dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, 
                              policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=False,
                              observation_name=observation_name, outputs_name=outputs_name, feature_loss=False,grayscale=grayscale,randomize_color=False, history_len=history_len, history_noise=history_noise, noise_type=noise_type, aloha=aloha, image_history_len=image_history_len,
                              image_history_stride=image_history_stride,
                              action_command_style=action_command_style,
                              task_boundary_provider=task_boundary_provider,
                              include_masked_segments=include_masked_segments,
                              frame_store_dir=frame_store_dir, frame_store_workers=num_dataloader_workers)
    train_num_workers = num_dataloader_workers
    val_num_workers = num_dataloader_workers
    print(f'Augment images: {train_dataset.data_aug}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    worker_context = {'multiprocessing_context': 'spawn'} if any(is_episode(path) for path in dataset_path_list) else {}
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, **({'prefetch_factor': 4, 'persistent_workers': True, **worker_context} if train_num_workers else {}))
    if val_dataset is None:
        val_dataloader = None
    else:
        val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, **({'prefetch_factor': 2, 'persistent_workers': True, **worker_context} if val_num_workers else {}))

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim, getattr(train_dataset, 'debug_image', None)


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def add_history_noise(data, history_len, sigma, noise_type='gaussian'):
    # Replace the elements in the history with drop_probability chance and replace with noise.
    if history_len > 0:
        if noise_type == 'dropout':
            for i in range(history_len):
                if np.random.rand() < sigma:
                    data[i] = torch.normal(0, 1, size=data[i].shape)
        elif noise_type == 'gaussian':
            data += torch.normal(0, 1, size=data.shape) * sigma
        elif noise_type == 'decreasing_gaussian':
            # The history is such that data[0] is the most recent observation and data[history_len-1] is the oldest.
            for i in range(history_len):
                noise = torch.normal(0, 1, size=data[i].shape) * (sigma * (i / history_len))
                data[i] += noise
        else :
            raise ValueError(f"Unknown noise type: {noise_type}")
    return data

def normalize_with_history(data, mean, std, history_len=0):
    """
    Normalize data that contains historical observations.
    
    Args:
        data: Input data tensor of shape (history_len * action_dim + current_dim,)
        mean: Normalization mean of shape (action_dim,) 
        std: Normalization std of shape (action_dim,)
        history_len: Number of historical steps included in the data
        
    Returns:
        Normalized data tensor
    """
    if history_len == 0:
        # No history, use standard normalization
        return (data - mean) / std
    
    # For data with history, we need to tile the normalization stats
    # to match the expanded dimension
    base_dim = len(mean)
    total_dim = len(data)
    
    # Calculate how many times to repeat the normalization stats
    n_repeats = total_dim // base_dim
    remainder = total_dim % base_dim
    
    # Create expanded normalization stats
    expanded_mean = torch.cat([mean.repeat(n_repeats), mean[:remainder]])
    expanded_std = torch.cat([std.repeat(n_repeats), std[:remainder]])
    
    return (data - expanded_mean) / expanded_std
