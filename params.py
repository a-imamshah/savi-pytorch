from typing import Optional
from typing import Tuple

import attr


@attr.s(auto_attribs=True)
class SAViParams:
    lr: float = 0.0002
    batch_size: int = 64
    val_batch_size: int = 64
    resolution: Tuple[int, int] = (64, 64)
    num_slots: int = 11
    num_iterations: int = 2
    data_root: str = "/kuacc/users/ashah20/datasets"
    gpus: int = 2
    max_epochs: int = 65
    num_sanity_val_steps: int = 1
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0
    num_train_videos: Optional[int] = None
    num_val_videos: Optional[int] = 128
    empty_cache: bool = True
    is_logger_enabled: bool = True
    is_verbose: bool = True
    num_workers: int = 8
    n_samples: int = 1
    warmup_steps_pct: float = 0.025
    decay_steps_pct: float = 0.2
    video_mode: str = True
    hidden_dims: Tuple[int, int, int] = (32, 32, 32, 32)
    decoder_hidden_dims: Tuple[int, int, int] = (128, 64, 64, 64)
    log_ari: bool = True
