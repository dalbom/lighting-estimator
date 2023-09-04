import torch


def dot(x, y):
    return torch.sum(x * y, dim=-1)


def light_vector_generic(pred, gt):
    pred_norm = torch.linalg.norm(pred, dim=-1, keepdim=True).clamp(min=1e-8)
    normalized_pred = pred / pred_norm

    dot_prod = dot(normalized_pred, gt)

    loss_dot = 1 - dot_prod.mean()
    loss_norm = torch.square(1 - pred_norm).mean()
    loss_z = torch.clamp_min(-normalized_pred[..., 2], 0).mean()

    return loss_dot + loss_norm + loss_z


def light_vector_3d(pred, gt):
    subimage_49 = pred.shape[1]

    if len(gt.shape) == 2:
        gt = gt.unsqueeze(1)
        gt = gt.repeat(1, subimage_49, 1)

    return light_vector_generic(pred, gt)


def light_vector_2d(pred, gt):
    return light_vector_generic(pred, gt)
