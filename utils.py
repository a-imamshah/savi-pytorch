from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union
from torchvision import utils as vutils

import torch
from pytorch_lightning import Callback

import wandb

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def conv_transpose_out_shape(in_size, stride, padding, kernel_size, out_padding, dilation=1):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


def rescale(x: Tensor) -> Tensor:
    return x * 2 - 1


def compact(l: Any) -> Any:
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                trainer.logger.experiment.log({"images": [wandb.Image(images)]}, commit=False)


def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)

def sample_images_inference(model, datamodule, params, image_id):
    dl = datamodule.val_dataloader()
    
    idx = [image_id]
    batch = next(iter(dl))[idx]
    if params.gpus > 0:
        batch = batch.cuda()
    recon_combined, recons, masks, slots, attns = model.forward(batch)
    
    print(slots.shape)

    # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
    out = to_rgb_from_tensor(
        torch.cat(
            [
                batch.unsqueeze(1),  # original images
                recon_combined.unsqueeze(1),  # reconstructions
                recons * masks + (1 - masks),  # each slot
            ],
            dim=1,
        )
    )

    batch_size, num_slots, nFrames, C, H, W = recons.shape
    images = vutils.make_grid(
        out.view(nFrames * out.shape[1], C, H, W).cpu(), normalize=False, nrow=nFrames,
    ) # Assuming nSamples is 1, or do out - out[0] and out.shape[0]

    return images,attns, slots