import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from model import SAViModel
from params import SAViParams
from utils import Tensor
from utils import to_rgb_from_tensor

from evaluation import compute_ari, compute_mot_metrics

import numpy as np

params = SAViParams()

device = torch.device('cuda')

class SAViMethod(pl.LightningModule):
    def __init__(self, model: SAViModel, datamodule: pl.LightningDataModule, params: SAViParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params
        self.save_hyperparameters()
        
        self.epochs = 0

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        batch = next(iter(dl))[idx]

        if self.params.gpus > 0:
            batch = batch.to(self.device)
        recon_combined, recons, masks, slots_all = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recon_combined * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, num_slots, nFrames, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(nFrames * out.shape[1], C, H, W).cpu(), normalize=False, nrow=nFrames,
        ) # Assuming nSamples is 1, or do out - out[0] and out.shape[0]

        return images

    def validation_step(self, batch, batch_idx):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        
        if params.log_ari:
            if(self.epochs%2==0):
                ari = compute_ari(self.model)
                logs = {"mean_ari": ari,}
                self.log_dict(logs, sync_dist=True)
                
                
        if params.log_mot_metrics:
            if(self.epochs%5==0):
                pred_mot_path = "/kuacc/users/ashah20/SAVi/pred_mot_path.json"
                gt_mot_path = "/kuacc/users/ashah20/SAVi/gt_mot_path.json"
                metrics = compute_mot_metrics(self.model, pred_mot_path, gt_mot_path)
                self.log_dict(metrics, sync_dist=True)
        
        self.epochs += 1
        
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
    

