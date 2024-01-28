import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from data_util import Dataset, worker_init_fn
from training import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
     
     data_dir = os.path.join("..", "climatehackai-2023")

     batch_size = 64
     num_sites = 50
     site_shuffle = True
     dropout = 0
     exp_name = f"SWeather_RNN_{num_sites}_shuffled"
     weather_features = ["alb_rad", "clct", "relhum_2m", "t_2m"]
                         # ["alb_rad", "clch", "clcl", "clcm", "relhum_2m", 
                         # "t_2m", "t_500", "t_850", "t_950", "t_g", "td_2m",
                         # "pmsl", "u_10m", "u_50", "u_500", "u_850", "u_950",
                         # "tot_prec", "ww", "aswdifd_s", "aswdir_s"]
                         #"aswdifd_s", "aswdir_s", #"cape_con", 
                         # "clch", "clcl", "clcm", "clct", "h_snow",
                         # "relhum_2m", "t_2m", "tot_prec"]
                         #"omega_1000", "omega_700", "omega_850", "omega_950", "pmsl", "relhum_2m"]
                         # runoff_g, runoff_s, t_2m, t_500, t_850, t_950, t_g, td_2m, 
                         # tot_prec, u_10m, u_50, u_500, u_850, u_950, 
                         # v_10m, v_50, v_500, v_850, v_950, vmax_10m, w_snow, ww, z0"]
     
     wandb_config = {
          "batch_size": batch_size,
          "lr": 1e-4,
          "weather_features": weather_features,
          "cnn_channel_out": None, #"ResNet18_32_2x",
          "train_data_period": "21_09_10",
          "dropout": dropout,
          "site_shuffle": site_shuffle,
          "sites": num_sites
     }
     

     with open("indices.json") as f:
          site_locations = {
               data_source: {
                    int(site): (int(location[0]), int(location[1]))
                    for site, location in locations.items()
               }
               for data_source, locations in json.load(f).items()
          }

     hrv_paths = [os.path.join(data_dir, "satellite-hrv", "2021", f"{i}.zarr.zip") for i in range(9, 11)]

     nonhrv_paths = [os.path.join(data_dir, "satellite-nonhrv", "2021", f"{i}.zarr.zip") for i in range(9, 11)]

     pv_paths = [os.path.join(data_dir, "pv", "2021", f"{i}.parquet") for i in range(9, 11)]

     weather_paths = [os.path.join(data_dir, "weather", "2021", f"{i}.zarr.zip") for i in range(9, 11)]

     train_dataset = Dataset(hrv_paths=hrv_paths, 
                             nonhrv_paths=nonhrv_paths,
                             pv_paths=pv_paths,
                             weather_paths=weather_paths,
                             site_locations=site_locations,
                             weather_features=weather_features,
                             num_sites=num_sites,
                             length=500,
                             site_shuffle=wandb_config["site_shuffle"])
     
     test_hrv_paths = [os.path.join(data_dir, "satellite-hrv", "2021", "11.zarr.zip")]

     test_nonhrv_paths = [os.path.join(data_dir, "satellite-nonhrv", "2021", "11.zarr.zip")]

     test_pv_paths = [os.path.join(data_dir, "pv", "2021", "11.parquet")]

     test_weather_paths = [os.path.join(data_dir, "weather", "2021", "11.zarr.zip")]
     
     test_dataset = Dataset(hrv_paths=test_hrv_paths, 
                            nonhrv_paths=test_nonhrv_paths,
                            pv_paths=test_pv_paths,
                            weather_paths=test_weather_paths,
                            site_locations=site_locations,
                            weather_features=weather_features,
                            num_sites=num_sites,
                            length=None,
                            site_shuffle=False)
     
     train_dataloader = DataLoader(train_dataset, 
                                   batch_size=wandb_config["batch_size"], 
                                   pin_memory=True,
                                   num_workers=0, 
                                   worker_init_fn=worker_init_fn)
     
     test_dataloader = DataLoader(test_dataset, 
                                  batch_size=wandb_config["batch_size"], 
                                  pin_memory=True,
                                  num_workers=0, 
                                  worker_init_fn=worker_init_fn)
     
     # test_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True, num_workers=1)
     
     # from models import NaiveModel as Model
     # from models import BasicModel as Model
     # from models import RecurrentModel as Model
     from models import PureWeatherModel as Model
    
     model = Model(num_weather_channel=len(weather_features), dropout=dropout, device=device).to(device)
     summary(model, input_size=[(wandb_config["batch_size"], 12), 
                                (wandb_config["batch_size"], 12, 128, 128), 
                                (wandb_config["batch_size"], 12, 128, 128, 11), 
                                (wandb_config["batch_size"], len(weather_features), 6)])
     
     criterion = nn.MSELoss()
     optimiser = optim.Adam(model.parameters(), lr=wandb_config["lr"])

     train(model, train_dataloader, test_dataloader, criterion, optimiser, device, 
           total_epochs=200,
           use_wandb=True, 
           wandb_proj="ClimatHackAI", 
           wandb_name=exp_name, 
           wandb_config=wandb_config)
     
     
     
