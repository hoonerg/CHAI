import math
import zarr
import torch
import numpy as np
import xarray as xr
import pandas as pd
from ocf_blosc2 import Blosc2
from torch.utils.data import IterableDataset

class Dataset(IterableDataset):
    def __init__(self, hrv_paths, nonhrv_paths, pv_paths, weather_paths, 
                 site_locations, weather_features, sites=None, num_sites=None,
                 length=None, site_shuffle=True, aggregate_weather_norm=True):
        super(Dataset, self).__init__()
        # Site Locations
        self._site_locations = site_locations
        self._sites = sites if sites else list(site_locations["hrv"].keys())
        self.num_sites = num_sites if num_sites else len(self._sites)
        self.site_shuffle = site_shuffle
        
        self.hrv_paths = hrv_paths
        self.nonhrv_paths = nonhrv_paths
        self.pv_paths = pv_paths
        self.weather_paths = weather_paths
        self.weather_features = weather_features
        self.aggregate_weather_norm = aggregate_weather_norm
        self.end = None
        self.length = length
        self.load()
        
        self.counter = 0
        self.indexes = np.arange(self.end)
        np.random.shuffle(self.indexes)
        
        self.OFFSET_1_HR = 12
        self._load_weather_stat()
        
    def _load_weather_stat(self):
        # with open('t_2m_2021_mean.npy', 'rb') as f:
            # self.t_2m_mean = np.load(f).mean()
        # with open('t_2m_2021_std.npy', 'rb') as f:
            # self.t_2m_std = np.load(f).mean()
        self.t_2m_mean = 283.74854
        self.t_2m_std = 5.5142407
        
    def load(self):
        # HRV
        self.hrvs = [zarr.open(hrv_path, mode='r') for hrv_path in self.hrv_paths]
        self.time_index = [hrv.time.astype('datetime64[ns]') for hrv in self.hrvs]
        
        # Dataset Configuration
        self.time_mapping = np.cumsum([len(hrv.time) for hrv in self.hrvs])
        self.start = 0
        self.end = self.time_mapping[-1]
        self.length = min(self.length, self.end) if self.length else self.end
        
        # PV
        self.pvs = [self._load_parquet(pv_path) for pv_path in self.pv_paths]
        
        # Non-HRV
        self.nonhrvs = [zarr.open(nonhrv_paths, mode='r') for nonhrv_paths in self.nonhrv_paths]
        
        # Weather
        self.weathers = [self._load_zarr(weather_path) for weather_path in self.weather_paths]
        
    def _load_parquet(self, file_path):
        return pd.read_parquet(file_path, engine='pyarrow').drop("generation_wh", axis=1)
    
    def _load_zarr(self, file_path):
        return xr.open_dataset(file_path, engine="zarr", consolidated=True,)
        
    def _get_split(self, i):
        self.counter += 1
        if self.counter == self.length:
            self.counter = 0
            np.random.shuffle(self.indexes)
        i = self.indexes[i]
        s = np.argmax((i - self.time_mapping) < 0)
        return s, i - self.time_mapping[s]
        
    def __iter__(self):
        for i in range(self.start, self.length):
            split, offset = self._get_split(i)
            if offset + self.OFFSET_1_HR >= 0:
                continue
            cur_ts = pd.Timestamp(self.time_index[split][offset], tz="UTC")
            hrv_data = self.hrvs[split].data[offset:offset+self.OFFSET_1_HR]
            nonhrv_data = self.nonhrvs[split].data[offset:offset+self.OFFSET_1_HR]
            
            cur_ts_no_tz = pd.Timestamp(self.time_index[split][offset])
            six_hours = slice(cur_ts_no_tz - np.timedelta64(1, 'h'), cur_ts_no_tz + np.timedelta64(5, 'h'))
            weather_data = self.weathers[split].sel(time=six_hours)
            weather_data_arr = []
            for col in self.weather_features:
                _data = weather_data[col].to_numpy()
                if col in ["alb_rad", "clct", "relhum_2m"]:
                    _data = _data / 100.0
                elif col in ["t_2m"]:
                    _data = (_data - self.t_2m_mean) / self.t_2m_std
                weather_data_arr.append(_data)
                        
            pv = self.pvs[split]
            pv_features = pv.xs(slice(cur_ts, cur_ts + np.timedelta64(55, 'm')), drop_level=False)
            pv_targets = pv.xs(
                slice(
                    str(cur_ts + np.timedelta64(1, 'h')),
                    str(cur_ts + np.timedelta64(4, 'h') + np.timedelta64(55, 'm')),
                ),
                drop_level=False,
            )
            
            if self.site_shuffle:
                np.random.shuffle(self._sites)
            
            for site in self._sites[:self.num_sites]:
                
                try:
                
                    # PV features and targets
                    site_features = pv_features.xs(site, level=1).to_numpy().squeeze(-1)
                    site_targets = pv_targets.xs(site, level=1).to_numpy().squeeze(-1)
                    assert site_features.shape == (12,) and site_targets.shape == (48,)

                    # HRV
                    x, y = self._site_locations["hrv"][site]
                    hrv_features = hrv_data[:, y - 64 : y + 64, x - 64 : x + 64, 0]
                    assert hrv_features.shape == (12, 128, 128)
                    
                    # Non HRV
                    x, y = self._site_locations["nonhrv"][site]
                    nonhrv_features = nonhrv_data[:, y - 64 : y + 64, x - 64 : x + 64, :]
                    assert nonhrv_features.shape == (12, 128, 128, 11)
                    
                    # Weather
                    x, y = self._site_locations["weather"][site]
                    weather_features = np.array([weather_data[:, y - 64 : y + 64, x - 64 : x + 64] for weather_data in weather_data_arr])
                    if not self.aggregate_weather_norm:
                        assert weather_features.shape == (len(self.weather_features), 6, 128, 128)
                    else:
                        weather_features = weather_features.reshape(len(self.weather_features), 6, 128 * 128).mean(axis=-1)
                        assert weather_features.shape == (len(self.weather_features), 6)

                    yield site_features, hrv_features, nonhrv_features, weather_features, site_targets
                
                except Exception as e:
                    # print(e)
                    continue
            
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    # Critical! Open data file again in the sub-processes
    dataset.load()
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil(
            (dataset.length) / float(worker_info.num_workers)
        )
    )
    worker_id = worker_info.id
    dataset.start = dataset.start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, dataset.end)