import logging
import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int, deterministic_cudnn: bool = False) -> None:
    """
    Setting multiple seeds to make runs reproducible.
    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA,
    but might slow down your training (see https://pytorch.org/docs/stable/notes/randomness.html#cudnn) !
    :param seed:number to use as seed
    :param deterministic_torch: Enable for full reproducibility when using CUDA. Caution: might slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def initialize_device_settings(
    use_cuda: Optional[bool] = None,
    local_rank: int = -1,
    multi_gpu: bool = True,
    devices: Optional[List[Union[str, torch.device]]] = None,
) -> Tuple[List[torch.device], int]:
    """
    Returns a list of available devices.
    :param use_cuda: Whether to make use of CUDA GPUs (if available).
    :param local_rank: Ordinal of device to be used. If -1 and `multi_gpu` is True, all devices will be used.
                       Unused if `devices` is set or `use_cuda` is False.
    :param multi_gpu: Whether to make use of all GPUs (if available).
                      Unused if `devices` is set or `use_cuda` is False.
    :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
    """
    if use_cuda is False:  # Note that it could be None, in which case we also want to just skip this step.
        devices_to_use = [torch.device("mps") if torch.has_mps else torch.device("cpu")]
        n_gpu = 0
    elif devices:
        if not isinstance(devices, list):
            raise ValueError(f"devices must be a list, but got {devices} of type {type(devices)}")
        if any(isinstance(device, str) for device in devices):
            torch_devices: List[torch.device] = [torch.device(device) for device in devices]
            devices_to_use = torch_devices
        else:
            devices_to_use = devices
        n_gpu = sum(1 for device in devices_to_use if "cpu" not in device.type)
    elif local_rank == -1:
        if torch.cuda.is_available():
            if multi_gpu:
                devices_to_use = [torch.device(device) for device in range(torch.cuda.device_count())]
                n_gpu = torch.cuda.device_count()
            else:
                devices_to_use = [torch.device("cuda:0")]
                n_gpu = 1
        else:
            devices_to_use = [torch.device("mps") if torch.has_mps else torch.device("cpu")]
            n_gpu = 0
    else:
        devices_to_use = [torch.device("cuda", local_rank)]
        torch.cuda.set_device(devices_to_use[0])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    # HF transformers v4.21.2 pipeline object doesn't accept torch.device("cuda"), it has to be an indexed cuda device
    # TODO eventually remove once the limitation is fixed in HF transformers
    device_to_replace = torch.device("cuda")
    devices_to_use = [torch.device("cuda:0") if device == device_to_replace else device for device in devices_to_use]

    logger.info(
        "Using devices: %s - Number of GPUs: %s",
        ", ".join([str(device) for device in devices_to_use]).upper(),
        n_gpu,
    )
    return devices_to_use, n_gpu


__all__ = ["set_all_seeds", "initialize_device_settings"]
