import os
import wandb
import torch

def train(model, train_dataloader, test_dataloader, criterion, optimiser, device, total_epochs=200,
          use_wandb=False, wandb_proj="", wandb_name="", wandb_config=""):
    if use_wandb:
        wandb.init(project=wandb_proj, config=wandb_config, name=wandb_name)
        chkpnt_dir = wandb.run.dir
        
        # Define the custom x axis metric
        wandb.define_metric("epoch")

        # Define which metrics to plot against that x-axis
        wandb.define_metric("validation/loss", step_metric='epoch')
        
    for epoch in range(total_epochs):
        model.train()

        chkp_pth = os.path.join(chkpnt_dir, f"{wandb_name}_mdl_chkpnt_epoch_{epoch}.pt")

        running_loss = 0.0
        count = 0
        for i, (pv_features, hrv_features, non_hrv_features, weather_features, pv_targets) in enumerate(train_dataloader):
            
            optimiser.zero_grad()

            predictions = model(
                pv_features.to(device, dtype=torch.float32),
                hrv_features.to(device, dtype=torch.float32), 
                non_hrv_features.to(device, dtype=torch.float32),
                weather_features.to(device, dtype=torch.float32),
            )

            loss = criterion(predictions, pv_targets.to(device, dtype=torch.float32))
            
            if not torch.isnan(loss):
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimiser.step()

                size = int(pv_targets.size(0))
                running_loss += float(loss) * size
                count += size
            
            if i % 10 == 9:
                print(f"Epoch {epoch + 1}, {i + 1}: {running_loss / count}")
                torch.save(model.state_dict(), chkp_pth)
                if use_wandb:
                    wandb.save(chkp_pth)
                    wandb.log({"train/loss": running_loss / count})
                    
        torch.save(model.state_dict(), chkp_pth)
        print(f"Epoch {epoch + 1} [TRAIN]: {running_loss / count}")
        
        model.eval()
        with torch.no_grad():
            test_running_loss = 0.0
            test_count = 0
            
            for i, (pv_features, hrv_features, non_hrv_features, weather_features, pv_targets) in enumerate(test_dataloader):
                predictions = model(
                    pv_features.to(device, dtype=torch.float32),
                    hrv_features.to(device, dtype=torch.float32), 
                    non_hrv_features.to(device, dtype=torch.float32),
                    weather_features.to(device, dtype=torch.float32),
                )
                loss = criterion(predictions, pv_targets.to(device, dtype=torch.float32))
                
                if not torch.isnan(loss):
                    size = int(pv_targets.size(0))
                    test_running_loss += float(loss) * size
                    test_count += size
                else:
                    test_count += 1
                
            print(f"Epoch {epoch + 1} [TEST]: {test_running_loss / test_count}")
            
        if use_wandb:
            wandb.log({"train/loss": running_loss / count, "validation/loss": test_running_loss / test_count,
                       "epoch": epoch})
            wandb.save(chkp_pth)
            
    if use_wandb:
        wandb.finish()
        