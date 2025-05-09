import numpy as np 
from .data_base import MovieFileTreeDataset
from .transforms import (ChangeChannelsOrder, CutVideos,
                            ExpandChannels, NeuroNormalizer,
                            ScaleInputs, Subsequence, ToTensor,
                            AddBehaviorAsChannels, AddPupilCenterAsChannels)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

    
def mouse_video_loader_train(
    paths,
    batch_size,
    val_batch_size=8,
    num_workers=4,
    normalize=True,
    exclude: str = None,
    cuda: bool = False,
    max_frame=None,
    frames=50,
    offset=-1,
    inputs_mean=None,
    inputs_std=None,
    include_behavior=True,
    include_pupil_centers=True,
    scale=1,
    to_cut=True,
    dist=False
):

    data_keys = [
        "videos",
        "responses",
    ]
    dataloaders_combined = {}
    for path in paths:
        dat2 = MovieFileTreeDataset(path, *data_keys, output_dict=True)
        train_transforms = [
            NeuroNormalizer(dat2, exclude=exclude, in_name="videos"),
            CutVideos(
                max_frame=max_frame,
                frame_axis={data_key: -1 for data_key in data_keys},
                target_groups=data_keys,
            ),
            ScaleInputs(size=(32,64), in_name="videos", channel_axis=-1),
            ChangeChannelsOrder((2, 0, 1), in_name="videos"),
            ChangeChannelsOrder((1, 0), in_name="responses"),
            Subsequence(frames=frames, channel_first=(), offset=offset), 
            ExpandChannels("videos"),
            ToTensor(cuda)
        ]
        
        dat2.transforms.extend(train_transforms) 
        
        dataloaders = {}
        tier_array = dat2.trial_info.tiers
        
        train_indices = np.where(tier_array == 'train')[0]
        train_sampler = SubsetRandomSampler(train_indices)
        shuffle = None 

        dataloaders['train'] = DataLoader(
                dat2,
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
        )
        subset_idx_val = np.where(tier_array == 'oracle')[0]
        dataloaders['oracle'] = DataLoader(
                dat2,
                sampler=SubsetRandomSampler(subset_idx_val),
                batch_size=batch_size,
        )
        dataset_name = path.split("/")[-1]
        for k, v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][dataset_name] = v

    return dataloaders_combined



class TrainValidationDataset(MovieFileTreeDataset):
    def __getitem__(self, item):
        
        ret = []
        for data_key in self.data_keys:
            if self.use_cache and item in self._cache[data_key]:
                ret.append(self._cache[data_key][item])
            else:
                if data_key in self.trial_info.keys():
                    val = self.trial_info[data_key][item : item + 1]
                else:
                    datapath = self.resolve_data_path(data_key)
                    val = np.load(datapath / "{}.npy".format(item))
                if self.use_cache:
                    self._cache[data_key][item] = val
                ret.append(val)
        
        # create data point and transform
        x = self.data_point(*ret)

        for tr in self.transforms:
            x = tr(x)              

        # apply output rename if necessary
        if self.rename_output:
            x = self._output_point(*x)

        if self.output_dict:
            x = x._asdict()
        
        if "videos" in x.keys():
            stacked_input_video = [x['videos'][:,batch_index:batch_index+16, :,:] for batch_index in range(0, 299-16, 16)]
            stacked_input_video = torch.stack(stacked_input_video, dim=0)
            x['videos'] = stacked_input_video
        if "pupil_center" in x.keys():                      
            stacked_input_pupil_center = [x['pupil_center'][:, batch_index:batch_index+16] for batch_index in range(0, 299-16, 16)]
            stacked_input_pupil_center = torch.stack(stacked_input_pupil_center, dim=0).squeeze(1)
            x['pupil_center']=stacked_input_pupil_center
        if "behavior" in x.keys(): 
            stacked_input_behavior = [x['behavior'][:, batch_index:batch_index+16] for batch_index in range(0, 299-16, 16)]
            stacked_input_behavior = torch.stack(stacked_input_behavior, dim=0).squeeze(1)
            x['behavior'] = stacked_input_behavior
        if "responses" in x.keys(): 
            stacked_responses= [x['responses'][batch_index:batch_index+16, :] for batch_index in range(0, 299-16, 16)]
            stacked_responses = torch.stack(stacked_responses, dim=0).squeeze(1)
            x['responses'] = stacked_responses  
        return x 
    

    
def mouse_video_loader_val(
    paths,
    batch_size,
    val_batch_size=1,
    num_workers=4,
    normalize=True,
    exclude: str = None,
    cuda: bool = False,
    max_frame=None,
    frames=50,
    offset=-1,
    inputs_mean=None,
    inputs_std=None,
    include_behavior=True,
    include_pupil_centers=True,
    scale=1,
    to_cut=True,
    dist=False
):
    """
    Symplified version of the sensorium mouse_loaders.py
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        frames (int, optional): how many frames ot take per video
        max_frame (int, optional): which is the maximal frame that could be taken per video
        offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        include_pupil_centers (bool, optional): whether to include pupil center data
        include_pupil_centers_as_channels(bool, optional): whether to include pupil center data as channels
        scale(float, optional): scalar factor for the image resolution.
            scale = 1: full iamge resolution (144 x 256)
            scale = 0.25: resolution used for model training (36 x 64)
        float64 (bool, optional):  whether to use float64 in MovieFileTreeDataset
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    data_keys = [
        "videos",
        "responses",
    ]
    dataloaders_combined = {}
    for path in paths:
        dat2_val = TrainValidationDataset(path, *data_keys, output_dict=True)    
        
        val_transforms = [
            NeuroNormalizer(dat2_val, exclude=exclude, in_name="videos"),
            CutVideos(
                max_frame=max_frame,
                frame_axis={data_key: -1 for data_key in data_keys},
                target_groups=data_keys,
            ),
            ScaleInputs(size=(32,64), in_name="videos", channel_axis=-1),
            ChangeChannelsOrder((2, 0, 1), in_name="videos"),
            ChangeChannelsOrder((1, 0), in_name="responses"),
            ExpandChannels("videos"),
            ToTensor(cuda)
        ]

        dat2_val.transforms.extend(val_transforms)
        dataloaders = {}
        tier_array = dat2_val.trial_info.tiers

        val_indices = np.where(tier_array == 'train')[0]
        val_indices = np.append(val_indices, np.where(tier_array == 'oracle')[0])
        dataloaders['train'] = DataLoader(
                dat2_val,
                sampler=SubsetSequentialSampler(val_indices),
                batch_size=val_batch_size,
                num_workers=num_workers,
        )
        
        dataset_name = path.split("/")[-1]
        for k, v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][dataset_name] = v

    return dataloaders_combined



class ValidationDataset(MovieFileTreeDataset):
    def __getitem__(self, item):
        
        ret = []
        for data_key in self.data_keys:
            if self.use_cache and item in self._cache[data_key]:
                ret.append(self._cache[data_key][item])
            else:
                if data_key in self.trial_info.keys():
                    val = self.trial_info[data_key][item : item + 1]
                else:
                    datapath = self.resolve_data_path(data_key)
                    val = np.load(datapath / "{}.npy".format(item))
                if self.use_cache:
                    self._cache[data_key][item] = val
                ret.append(val)
        
        # create data point and transform
        x = self.data_point(*ret)

        for tr in self.transforms:
            x = tr(x)              

        # apply output rename if necessary
        if self.rename_output:
            x = self._output_point(*x)

        if self.output_dict:
            x = x._asdict()
         
        return x 
    

    
def data_loader_for_rep(
    paths,
    batch_size,
    val_batch_size=1,
    num_workers=4,
    normalize=True,
    exclude: str = None,
    cuda: bool = False,
    max_frame=None,
    frames=50,
    offset=-1,
    inputs_mean=None,
    inputs_std=None,
    include_behavior=True,
    include_pupil_centers=True,
    scale=1,
    to_cut=True,
    dist=False
):
    """
    Symplified version of the sensorium mouse_loaders.py
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        frames (int, optional): how many frames ot take per video
        max_frame (int, optional): which is the maximal frame that could be taken per video
        offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        include_pupil_centers (bool, optional): whether to include pupil center data
        include_pupil_centers_as_channels(bool, optional): whether to include pupil center data as channels
        scale(float, optional): scalar factor for the image resolution.
            scale = 1: full iamge resolution (144 x 256)
            scale = 0.25: resolution used for model training (36 x 64)
        float64 (bool, optional):  whether to use float64 in MovieFileTreeDataset
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    data_keys = [
        "videos",
        "responses",
    ]
    dataloaders_combined = {}
    for path in paths:
        dat2_val = ValidationDataset(path, *data_keys, output_dict=True)    
        
        val_transforms = [
            NeuroNormalizer(dat2_val, exclude=exclude, in_name="videos"),
            CutVideos(
                max_frame=max_frame,
                frame_axis={data_key: -1 for data_key in data_keys},
                target_groups=data_keys,
            ),
            ScaleInputs(size=(32,64), in_name="videos", channel_axis=-1),
            ChangeChannelsOrder((2, 0, 1), in_name="videos"),
            ChangeChannelsOrder((1, 0), in_name="responses"),
            ExpandChannels("videos"),
            ToTensor(cuda)
        ]

        dat2_val.transforms.extend(val_transforms)
        dataloaders = {}
        tier_array = dat2_val.trial_info.tiers

        val_indices = np.where(tier_array == 'train')[0]
        val_indices = np.append(val_indices, np.where(tier_array == 'oracle')[0])
        dataloaders['train'] = DataLoader(
                dat2_val,
                sampler=SubsetSequentialSampler(val_indices),
                batch_size=val_batch_size,
                num_workers=num_workers,
        )
        
        dataset_name = path.split("/")[-1]
        for k, v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][dataset_name] = v

    return dataloaders_combined
