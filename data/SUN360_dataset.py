import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Sun360Dataset(Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.batch_size = cfg.batch_size
        self.num_subimages = cfg.num_subimages

        if hasattr(cfg, "random_state"):
            random_state = cfg.random_state
        else:
            random_state = None

        self.rng = np.random.default_rng(random_state)

        self._load_data(cfg.split_file)

        # Rows and columns are hardcoded for the maximum possible image size
        # d_model should follow the transformer's out_dim parameter
        self.d_model = 512
        self.angle_rads = self._get_angles(
            np.arange(self.d_model // 2), self.d_model // 4
        )

        # The bins should be adjusted to be larger than the maximum width or height of all input images
        self.pos_enc_x = self._generate_positional_encoding_seeds(bins=2000, fov=90)
        self.pos_enc_y = self._generate_positional_encoding_seeds(
            bins=1000, fov=35, flip=True
        )

    def _normalize(self, arr, m1_to_p1=False):
        if m1_to_p1:
            arr_max, arr_min = np.max(arr), np.min(arr)

            return (arr - arr_min) / (arr_max - arr_min) * 2 - 1, arr_max, arr_min
        else:
            arr_mean, arr_std = np.mean(arr), np.std(arr)

            return (arr - arr_mean) / arr_std, arr_mean, arr_std

    def _load_data(self, datapath):
        original_data = pd.read_csv(datapath)

        self.gt_dict = {}
        self.gt_param_dict = {}
        self.path_arr = original_data["filepath"].to_numpy()

        # Compensate phi for elevation angle (pitch)
        elevation_arr = original_data["elev"].to_numpy()
        theta_arr = np.radians(original_data["theta"].to_numpy())
        phi_arr = np.radians(original_data["phi"].to_numpy() - elevation_arr)

        x_arr = np.cos(phi_arr) * np.sin(theta_arr)
        y_arr = np.cos(phi_arr) * np.cos(theta_arr)
        z_arr = np.sin(phi_arr)
        gt_arr = np.column_stack([x_arr, y_arr, z_arr])

        parameters = ["kappa", "beta", "t"]
        params_arr = [
            self._normalize(np.array(original_data[param])[:, None])
            for param in parameters
        ]
        params_normalized = [x[0] for x in params_arr]
        params_stats = [(x[1], x[2]) for x in params_arr]

        sun_arr, mean_sun, std_sun = self._normalize(
            np.array(
                [
                    list(map(float, record.split(" ")))
                    for record in original_data["wsun"]
                ]
            )
        )
        sky_arr, mean_sky, std_sky = self._normalize(
            np.array(
                [
                    list(map(float, record.split(" ")))
                    for record in original_data["wsky"]
                ]
            )
        )

        print("\nParameter Statistics")
        for param, (mean, std) in zip(parameters, params_stats):
            print(param.capitalize(), mean, std)
        print("Sun", mean_sun, std_sun)
        print("Sky", mean_sky, std_sky, "\n")

        self.gt_statistics = {
            "kappa": params_stats[0],
            "beta": params_stats[1],
            "turbidity": params_stats[2],
            "sun": [mean_sun, std_sun],
            "sky": [mean_sky, std_sky],
        }

        gt_param_arr = np.hstack([*params_normalized, sun_arr, sky_arr])

        for idx, filepath in enumerate(self.path_arr):
            self.gt_dict[filepath] = gt_arr[idx]
            self.gt_param_dict[filepath] = gt_param_arr[idx]

        self.num_images = len(self.path_arr)

    def _get_angles(self, i, out_dim):
        return 1 / np.power(10000, i / out_dim)

    def _generate_positional_encoding_seeds(self, bins, fov, flip=False):
        half_d_model = self.d_model // 2
        pos_encoding = np.zeros([bins, self.d_model])

        # Radian angle range will be [-0.5, 0.5] * radian_fov
        radian_fov = np.radians(fov)

        # Points to the center of the discrete space
        bin_array = (np.arange(bins) - (bins / 2) + 0.5) / bins * radian_fov

        if flip:
            bin_array = bin_array[::-1]

        # Broadcasting bin_array and angle_rads to compute sine values in one go
        sin_array = np.sin(bin_array[:, None] * 32 * self.angle_rads[:half_d_model])

        # Populating pos_encoding using advanced indexing
        pos_encoding[:, 0::2] = sin_array * np.sin(self.angle_rads[:half_d_model])
        pos_encoding[:, 1::2] = sin_array * np.cos(self.angle_rads[:half_d_model])

        return pos_encoding

    def _generate_subimg(self, filepath):
        img = self._imread(filepath)
        height, width, _ = img.shape

        # Pre-allocate memory for images and positional encodings
        imgs = np.empty((self.num_subimages, 224, 224, 3), dtype=img.dtype)
        pos_encs = np.empty((self.num_subimages, 49, self.d_model), dtype=float)

        crop_size = 224
        step_size = 32

        for i in range(self.num_subimages):
            crop_x = self.rng.integers(0, width - crop_size)
            crop_y = self.rng.integers(0, height - crop_size)

            imgs[i] = img[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size, :]

            x_arr = np.arange(crop_x + 16, crop_x + crop_size, step_size)
            y_arr = np.arange(crop_y + 16, crop_y + crop_size, step_size)

            pos_encs[i] = self._get_2D_positional_encoding(x_arr, y_arr)

        return list(imgs), pos_encs

    def _get_2D_positional_encoding(self, x_arr, y_arr):
        num_y, num_x = len(y_arr), len(x_arr)
        half_d_model = self.d_model // 2

        pos_encoding = np.zeros([num_y, num_x, self.d_model])

        # Using numpy broadcasting to eliminate the outer loop
        pos_encoding[:, :, :half_d_model] = self.pos_enc_x[x_arr, :half_d_model][
            None, :, :
        ]
        pos_encoding[:, :, half_d_model:] = self.pos_enc_y[y_arr, :half_d_model][
            :, None, :
        ]

        return pos_encoding.reshape((-1, self.d_model))

    def _imread(self, filepath):
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB) / 255

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        if index >= len(self) or index < 0:
            raise IndexError("Index out of bounds.")

        filepath = self.path_arr[index]
        subimages, pos_encs = self._generate_subimg(filepath)

        return {
            "input": np.asarray(subimages, dtype=np.float32),  # Shape: S, 224, 224, 3
            "gt": np.asarray(self.gt_dict[filepath], dtype=np.float32),  # Shape: 3
            "gt_param": np.asarray(self.gt_param_dict[filepath], dtype=np.float32),
            "pos_enc": np.asarray(pos_encs, dtype=np.float32),  # Shape: S, 49, 512
            "filepath": filepath,
            "statistics": self.gt_statistics,
        }


def get_data_manager(cfg):
    return Sun360Dataset(cfg)
