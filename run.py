import os
import numpy as np
import argparse
import time
import yaml
from shutil import copy2
import sys
import json
import importlib
from torch.utils.data import DataLoader


# Finish this file and implement GT-sphere training code
# Need to load from a h5 file when it is deployed on the GPU cluster


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description="LightingEstimation Training")

    parser.add_argument("config", type=str, help="The configuration file.")
    # Resume:
    parser.add_argument("--resume", default=False, action="store_true")

    parser.add_argument(
        "--pretrained", default=None, type=str, help="pretrained model checkpoint"
    )

    # For easy debugging:
    parser.add_argument("--test_run", default=False, action="store_true")

    parser.add_argument("--special", default=None, type=str, help="Run special tasks")

    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = dict2namespace(config)

    #  Create log_name
    log_prefix = ""

    if args.test_run:
        log_prefix = "tmp_"

    if args.special is not None:
        log_prefix = log_prefix + "special_{}_".format(args.special)

    if config.alias is not None:
        log_prefix = config.alias + "_"

    run_time = time.strftime("%Y-%b-%d-%H-%M-%S")

    # Currently save dir and log_dir are the same
    config.log_name = "logs/{}{}/log.txt".format(log_prefix, run_time)
    config.save_dir = "logs/{}{}/checkpoints".format(log_prefix, run_time)
    config.log_dir = "logs/{}{}".format(log_prefix, run_time)

    os.makedirs(os.path.join(config.log_dir, "config"))
    os.makedirs(config.save_dir)

    copy2(args.config, os.path.join(config.log_dir, "config"))

    with open(os.path.join(config.log_dir, "config", "argv.json"), "w") as f:
        json.dump(sys.argv, f)

    return args, config


def get_loader(cfg):
    data_type = cfg.type

    dataset_lib = importlib.import_module(data_type)

    dataset_name = data_type.split(".")[1]
    dataset_cfg = cfg.__getattribute__(dataset_name)

    train_dataset = dataset_lib.get_data_manager(dataset_cfg.train)
    val_dataset = dataset_lib.get_data_manager(dataset_cfg.validation)
    test_dataset = dataset_lib.get_data_manager(dataset_cfg.test)

    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=dataset_cfg.train.batch_size, shuffle=True
        ),
        "validation": DataLoader(
            val_dataset, batch_size=dataset_cfg.validation.batch_size, shuffle=False
        ),
        "test": DataLoader(
            test_dataset, batch_size=dataset_cfg.test.batch_size, shuffle=False
        ),
    }

    return dataloaders


def record_log(log_dict, log_info):
    for k, v in log_info.items():
        if k not in log_dict:
            log_dict[k] = []

        log_dict[k].append(v)


def main(args, cfg):
    writer = open(cfg.log_name, "w")
    log_dict = {}

    # Data loader
    loaders = get_loader(cfg.data)

    # Trainer
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg)

    # Prepare for training
    start_epoch = 0
    global_step = 0

    # Main training loop
    for epoch in range(start_epoch, cfg.trainer.epochs):
        # Randomize batch indices
        log_dict["train"] = {}

        # train for one epoch
        for i, train_data in enumerate(loaders["train"]):
            train_logs, predictions = trainer.step_train(train_data)

            if (i + 1) % cfg.trainer.summary_freq == 0:
                record_log(log_dict["train"], train_logs)
                trainer.log_train_step(
                    global_step, train_logs, predictions, train_data["gt"]
                )

            if (i + 1) % cfg.trainer.img_freq == 0:
                trainer.train_tf_writer.flush()
                trainer.val_tf_writer.flush()
                trainer.test_tf_writer.flush()

            global_step += 1

        # Validation loop
        if cfg.trainer.end_epoch_validate:
            log_dict["validation"] = {}

            for val_data in loaders["validation"]:
                val_logs, predictions = trainer.step_val(val_data)
                record_log(log_dict["validation"], val_logs)

            loss = round(np.mean(log_dict["validation"]["loss"]), 4)
            trainer.log_val(global_step, log_dict["validation"])

        is_new_checkpoint = trainer.save(epoch=epoch + 1, loss=loss, epoch_end=True)

        # Test loop if new lowest validation loss is recorded
        if is_new_checkpoint:
            log_dict_key = "test"
            log_dict[log_dict_key] = {}

            for test_data in loaders["test"]:
                test_logs, predictions = trainer.step_test(test_data)
                record_log(log_dict[log_dict_key], test_logs)

            trainer.log_test(global_step, log_dict[log_dict_key])

    writer.close()


if __name__ == "__main__":
    # command line args
    args, cfg = get_args()

    main(args, cfg)
