## Slot Attention for Video (SAVi)

This repository contains an unofficial implementation of "Conditional Object-Centric
Learning from Video" (ICLR 2022)

Paper Link: https://arxiv.org/abs/2111.12594

## Instructions

Clone this repository
```sh
git clone https://github.com/a-imamshah/savi-pytorch
cd savi-pytorch
```

Create new environment
```sh
conda create --name savi python=3.8
source activate savi
```

```sh
pip install -r requirements.txt
```

## Usage

```bash
python train.py
```

Modify `SAViParams` in `params.py` to modify the hyperparameters.

### Logging

To log outputs to [wandb](https://wandb.ai/home), run `wandb login YOUR_API_KEY` and set `is_logging_enabled=True` in `SAViParams`.

## Credits
Credits to the original authors of the paper: Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone,Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy & Klaus Greff.

## Acknowledgements
I adapted this code from the unofficial implementation of "Object-Centric Learning with Slot Attention" by [Untitled AI](https://github.com/untitled-ai/slot_attention)
