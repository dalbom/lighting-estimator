import os
import torch
import numpy as np
import importlib
import models.lossfuns as lossfuns
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.alias = cfg.alias
        self.min_loss = 1e10

        # Initialization
        self._init_model()
        self._init_loss_functions()
        self._init_optimizers()
        self._init_tensorboard_writers()

    def _init_model(self):
        lib_lighting_estimator = importlib.import_module(
            self.cfg.models.lighting_estimator.type
        )
        self.module = lib_lighting_estimator.get_module(
            self.cfg.models.lighting_estimator
        ).cuda()

    def _init_loss_functions(self):
        self.lossfun_light = self._get_lossfun(self.cfg.trainer.loss_light)
        self.lossfun_light_vector = self._get_lossfun(
            self.cfg.trainer.loss_light_vector
        )
        self.lossfun_param = self._get_lossfun(self.cfg.trainer.loss_param)

    def _init_optimizers(self):
        self.optim_e2e = self._get_optim(
            self.module.parameters(), self.cfg.trainer.optimizer
        )

    def _init_tensorboard_writers(self):
        self.train_tf_writer = SummaryWriter(
            log_dir=os.path.join(self.cfg.log_dir, "train")
        )
        self.val_tf_writer = SummaryWriter(
            log_dir=os.path.join(self.cfg.log_dir, "val")
        )
        self.test_tf_writer = SummaryWriter(
            log_dir=os.path.join(self.cfg.log_dir, "test")
        )

    def _angle_from_loss(self, loss):
        # """Helper function to convert loss to angle in degrees."""
        # return np.degrees(np.arccos(1 - loss))
        """Helper function to convert loss to angle in degrees."""
        loss_val = np.array(loss).flatten()[0]  # Ensure scalar
        return np.degrees(np.arccos(1 - loss_val))

    def _log_scalars(self, tf_writer, step, loss_dict, prefix=""):
        """Common function to log scalar values."""
        angle_loss_light = self._angle_from_loss(loss_dict["loss_light"])
        angle_loss_light_vector = self._angle_from_loss(loss_dict["loss_light_vector"])

        tf_writer.add_scalar(
            f"{prefix}angular error", angle_loss_light, global_step=step
        )
        tf_writer.add_scalar(
            f"{prefix}light vector loss: mean of light loss values for all estimates",
            angle_loss_light_vector,
            global_step=step,
        )

        tf_writer.add_scalars(
            "Rmse of parameter estimates",
            {
                "all": np.mean(loss_dict["loss_param"]),
                "kappa": np.mean(loss_dict["loss_param_kappa"]),
                "beta": np.mean(loss_dict["loss_param_beta"]),
                "turbidity": np.mean(loss_dict["loss_param_turbidity"]),
                "sun": np.mean(loss_dict["loss_param_sun"]),
                "sky": np.mean(loss_dict["loss_param_sky"]),
            },
            global_step=step,
        )

    def log_train_step(self, step, train_loss_dict, preds, gt):
        """Log only scalars and optimizer state during summary ops"""
        self._log_scalars(self.train_tf_writer, step, train_loss_dict)
        self.train_tf_writer.add_text("gt vector", str(gt), global_step=step)

    def log_val(self, step, val_loss_dict):
        """Log validation scalars and input images during log ops"""
        self._log_scalars(self.val_tf_writer, step, val_loss_dict)
        self.val_loss = self._angle_from_loss(val_loss_dict["loss_light"])

    def log_test(self, step, test_loss_dict):
        """Log test scalars and input images during log ops"""
        self._log_scalars(self.test_tf_writer, step, test_loss_dict)

    def _mean_angle_from_loss(self, arr):
        return np.mean(np.degrees(np.arccos(1 - np.asarray(arr))))

    def _get_optim(self, parameters, cfg):
        if cfg.type.lower() == "adamw":
            return AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            raise NotImplementedError(f"Unknown optimizer: {cfg.type}")

    def _get_lossfun(self, cfg):
        loss_type = cfg.type.lower()

        if loss_type == "light_vector_3d":
            return lambda pred, gt: lossfuns.light_vector_3d(pred, gt)
        elif loss_type == "light_vector_2d":
            return lambda pred, gt: lossfuns.light_vector_2d(pred, gt)
        elif loss_type == "mse":
            return torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"Unknown loss function: {cfg.type}")

    def _np2cuda(self, nparray):
        if isinstance(nparray, torch.Tensor):
            return nparray.cuda()

        if isinstance(nparray, list):
            nparray = np.array(nparray, dtype=np.float32)
        elif isinstance(nparray, np.ndarray):
            nparray = nparray.astype(np.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(nparray)}")

        return torch.from_numpy(nparray).cuda()

    def step_train(self, data):
        self.module.train()
        self.optim_e2e.zero_grad()

        log_info, result_sunpos, loss = self.step(data)
        loss.backward()
        self.optim_e2e.step()

        return log_info, result_sunpos

    def step_val(self, data):
        with torch.no_grad():
            self.module.eval()
            return self.step(data)[:2]  # Only return log_info and result_sunpos

    def _loss_param_unnormalized(self, pred, gt, stat):
        return torch.sqrt(
            self.lossfun_param(pred * stat[1] + stat[0], gt * stat[1] + stat[0])
        )

    def _unnormalize_param(self, pred, stat):
        params = [
            "kappa",
            "beta",
            "turbidity",
            "sun",
            "sun",
            "sun",
            "sky",
            "sky",
            "sky",
        ]
        pred = pred.squeeze().cpu().numpy()
        return [
            pred[i] * stat[params[i]][1] + stat[params[i]][0]
            for i in range(len(params))
        ]

    def step_test(self, data):
        with torch.no_grad():
            self.module.eval()
            # Adjust the reshaping and permuting of batch_input.
            batch_input = self._np2cuda(data["input"]).permute(
                0, 1, 4, 2, 3
            )  # B, S, 3, 224, 224
            batch_pos_enc = self._np2cuda(data["pos_enc"])
            batch_gt = self._np2cuda(data["gt"])
            batch_gt_param = self._np2cuda(data["gt_param"])

            result_sunpos, result_param = self.module(batch_input, batch_pos_enc)
            b, s, _, _, _ = batch_input.shape
            result_sunpos = result_sunpos.reshape(b, s * 49, -1)  # B, S*49, 3
            result_param = result_param.reshape(b, s * 49, -1)

            mean_sunpos = result_sunpos.mean(dim=1)
            mean_param = result_param.mean(dim=1)

            # Calculate losses.
            loss_light = self.lossfun_light(mean_sunpos, batch_gt)
            loss_light_vector = self.lossfun_light_vector(result_sunpos, batch_gt)

            param_losses = [
                self._loss_param_unnormalized(
                    mean_param[:, i],
                    batch_gt_param[:, i],
                    self._np2cuda(data["statistics"][param_name]),
                )
                for i, param_name in enumerate(["kappa", "beta", "turbidity"])
            ]
            param_losses.append(
                self._loss_param_unnormalized(
                    mean_param[:, 3:6],
                    batch_gt_param[:, 3:6],
                    self._np2cuda(data["statistics"]["sun"]),
                )
                / 3
            )
            param_losses.append(
                self._loss_param_unnormalized(
                    mean_param[:, 6:9],
                    batch_gt_param[:, 6:9],
                    self._np2cuda(data["statistics"]["sky"]),
                )
                / 3
            )

            loss_param = sum(param_losses) / 5

            stdev_sunpos = result_sunpos.std(dim=1).mean()

            # Log information.
            log_info = {
                "loss_light": loss_light.item(),
                "loss_light_vector": loss_light_vector.item(),
                "loss_param_kappa": param_losses[0].item(),
                "loss_param_beta": param_losses[1].item(),
                "loss_param_turbidity": param_losses[2].item(),
                "loss_param_sun": param_losses[3].item(),
                "loss_param_sky": param_losses[4].item(),
                "loss_param": loss_param.item(),
                "loss": loss_light.item() + loss_param.item(),
                "stdev": stdev_sunpos.item(),
            }

            return log_info, {
                "sunpos": mean_sunpos.cpu().numpy(),
                "param": self._unnormalize_param(mean_param, data["statistics"]),
                "gt_sunpos": batch_gt.cpu().numpy(),
                "gt_param": self._unnormalize_param(batch_gt_param, data["statistics"]),
            }

    def step(self, data):
        # Batch, Subimages, 224, 224, 3 -> B, S, 3, 224, 224
        batch_input = self._np2cuda(data["input"]).permute(0, 1, 4, 2, 3)
        batch_pos_enc = self._np2cuda(data["pos_enc"])
        batch_gt = self._np2cuda(data["gt"])
        batch_gt_param = self._np2cuda(data["gt_param"])

        result_sunpos, result_param = self.module(batch_input, batch_pos_enc)
        b, s, _, _ = batch_pos_enc.shape
        result_sunpos = result_sunpos.reshape(b, s * 49, -1)
        result_param = result_param.reshape(b, s * 49, -1)

        mean_sunpos = result_sunpos.mean(dim=1)
        mean_param = result_param.mean(dim=1)

        loss_light = self.lossfun_light(mean_sunpos, batch_gt)
        loss_light_vector = self.lossfun_light_vector(result_sunpos, batch_gt)

        param_losses = [
            self.lossfun_param(mean_param[:, i], batch_gt_param[:, i]) for i in range(3)
        ]
        param_losses.append(
            self.lossfun_param(mean_param[:, 3:6], batch_gt_param[:, 3:6]) / 3
        )
        param_losses.append(
            self.lossfun_param(mean_param[:, 6:9], batch_gt_param[:, 6:9]) / 3
        )

        loss_param = sum(param_losses) / 5

        log_info = {
            "loss_light": loss_light.item(),
            "loss_light_vector": loss_light_vector.item(),
            "loss_param_kappa": param_losses[0].item(),
            "loss_param_beta": param_losses[1].item(),
            "loss_param_turbidity": param_losses[2].item(),
            "loss_param_sun": param_losses[3].item(),
            "loss_param_sky": param_losses[4].item(),
            "loss_param": loss_param.item(),
            "loss": loss_light.item() + loss_param.item(),
        }

        return (
            log_info,
            {
                "gt": batch_gt,
                "filtered": mean_sunpos,
            },
            loss_light + loss_light_vector + loss_param,
        )

    def save(self, epoch, loss, epoch_end=False):
        if loss < self.min_loss:
            self.min_loss = loss
            le_filename = os.path.join(
                self.cfg.save_dir, f"{self.alias}_{epoch:04}_loss_{loss:.4f}.pth"
            )
            torch.save(self.module.state_dict(), le_filename)
            return True
        return False
