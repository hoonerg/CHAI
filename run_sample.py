import os
import json
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from data_util import Dataset, worker_init_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
     
    data_dir = os.path.join("..", "climatehack", "data")

    batch_size = 64
    num_sites = 50
    site_shuffle = True
    dropout = 0
    exp_name = f"T_PV_{num_sites}"
    
    USE_WANDB = False
    WANDB_PROJ = "CCAA"
    
    TOTAL_EPOCH = 200
     
    SRC_SEQ_LEN = 12
    TGT_SEQ_LEN = 48
    
    SRC_SIZE = 1
    TGT_SIZE = 1
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
     
    weather_features = ["alb_rad", "clct", "relhum_2m", "t_2m"]
     
    wandb_config = {
        "batch_size": batch_size,
        "lr": 1e-4,
        "weather_features": weather_features,
        "cnn_channel_out": None, #"ResNet18_32_2x",
        "train_data_period": "21_09_10",
        "dropout": dropout,
        "site_shuffle": site_shuffle,
        "sites": num_sites,
        "SRC_SIZE": SRC_SIZE,
        "TGT_SIZE": TGT_SIZE,
        "EMB_SIZE": EMB_SIZE,
        "NHEAD": NHEAD,
        "FFN_HID_DIM": FFN_HID_DIM,
        "NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS,
        "NUM_DECODER_LAYERS": NUM_DECODER_LAYERS,
        "TOTAL_EPOCH": TOTAL_EPOCH
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
                        length=500,
                        site_shuffle=False)
    
    from transformer_models import collate_fn
    
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=wandb_config["batch_size"], 
                                pin_memory=True,
                                num_workers=0, 
                                worker_init_fn=worker_init_fn,
                                collate_fn=collate_fn)
    
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=wandb_config["batch_size"], 
                                pin_memory=True,
                                num_workers=0, 
                                worker_init_fn=worker_init_fn,
                                collate_fn=collate_fn)
    
    from transformer_models import Seq2SeqTransformer as Model
    
    model = Model(num_encoder_layers=NUM_ENCODER_LAYERS, 
                  num_decoder_layers=NUM_DECODER_LAYERS, 
                  emb_size=EMB_SIZE,
                  nhead=NHEAD,
                  src_size=SRC_SIZE,
                  tgt_size=TGT_SIZE,
                  dim_feedforward=FFN_HID_DIM,
                  dropout=dropout).to(device)
    
    summary(model, input_size=[(SRC_SEQ_LEN, batch_size, SRC_SIZE),
                               (TGT_SEQ_LEN, batch_size, SRC_SIZE),
                               (SRC_SEQ_LEN, SRC_SEQ_LEN),
                               (TGT_SEQ_LEN, TGT_SEQ_LEN)])
    
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=wandb_config["lr"], betas=(0.9, 0.98), eps=1e-9)
    
    from transformer_models import train, validate

    chkpnt_dir = os.path.join(".", "ckpt")

    if USE_WANDB:
        wandb.init(project=WANDB_PROJ, config=wandb_config, name=exp_name)
        chkpnt_dir = wandb.run.dir
        
        # Define the custom x axis metric
        wandb.define_metric("epoch")
        
        # Define which metrics to plot against that x-axis
        wandb.define_metric("validation/loss", step_metric='epoch')
        
    for epoch in range(TOTAL_EPOCH):

        # -----------------TRAINING-----------------
        train_loss = train(model=model, 
                           data_loader=train_dataloader, 
                           criterion=criterion,
                           optimiser=optimiser,
                           device=device)
        
        chkp_pth = os.path.join(chkpnt_dir, f"{exp_name}_mdl_chkpnt_epoch_{epoch}.pt")
        torch.save(model.state_dict(), chkp_pth)
        print(f"Epoch {epoch + 1} [TRAIN]: {train_loss}")
        
        # -----------------VALIDATION-----------------
        validation_loss = validate(model=model,
                                   data_loader=test_dataloader,
                                   criterion=criterion,
                                   device=device)
        
        if USE_WANDB:
            wandb.log({"train/loss": train_loss, 
                    "validation/loss": validation_loss, 
                    "epoch": epoch})
            wandb.save(chkp_pth)
        print(f"Epoch {epoch + 1} [TEST]: {validation_loss}")
        
    if USE_WANDB:
        wandb.finish()
        
        
        
        
     
