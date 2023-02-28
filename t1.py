from model import SAViModel
from params import SAViParams
from evaluation import compute_ari, compute_mot_metrics
params = SAViParams()

import torch



model = SAViModel(
    resolution=params.resolution,
    num_slots=params.num_slots,
    num_iterations=params.num_iterations,
    empty_cache=params.empty_cache,
)


device = torch.device('cuda')
model = model.to(device)

pred_mot_path = "/kuacc/users/ashah20/SAVi/pred_mot_path.json"
gt_mot_path = "/kuacc/users/ashah20/SAVi/gt_mot_path.json"
metrics = compute_mot_metrics(model, pred_mot_path, gt_mot_path)
print(metrics)