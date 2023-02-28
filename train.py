from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from data import CaterDataModule
from method import SAViMethod
from model import SAViModel
from params import SAViParams
from utils import ImageLogCallback
from utils import rescale

from torchvision.transforms import transforms
import torch

from pytorch_lightning.callbacks import ModelCheckpoint


torch.manual_seed(16409)

def main(params: Optional[SAViParams] = None):
    if params is None:
        params = SAViParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({params.num_slots - 1}) objects.")
        if params.num_train_videos:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_videos}")
        if params.num_val_videos:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_videos}")

    
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),
            transforms.Resize(params.resolution),
        ]
    )
        

    cater_datamodule = CaterDataModule(
        data_root=params.data_root,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        cater_transforms=img_transforms,
        num_train_videos=params.num_train_videos,
        num_val_videos=params.num_val_videos,
        num_workers=params.num_workers,
    )

    print(f"Training set size (images must have {params.num_slots - 1} objects):", len(cater_datamodule.train_dataset))

    model = SAViModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
    )
    
    

    method = SAViMethod(model=model, datamodule=cater_datamodule, params=params)

    logger_name = "savi-baseline"
    logger = pl_loggers.WandbLogger(project="savi-cater", name=logger_name)
    
    checkpoint_callback = ModelCheckpoint(dirpath="/kuacc/users/ashah20/SAVi/best_models/", save_top_k=2, monitor="avg_val_loss")

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="ddp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=100,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(), checkpoint_callback] if params.is_logger_enabled else [],
        gradient_clip_val=0.05,
    )
    trainer.fit(method)


if __name__ == "__main__":
    main()