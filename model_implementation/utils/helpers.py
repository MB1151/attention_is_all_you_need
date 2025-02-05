# This file implements helper functions that are used in the model implementation.

import copy
import os
import torch

from model_implementation.utils.constants import DEVICE_CPU, DEVICE_GPU, MODEL_CHECK_POINT_PATH
from torch import nn


# Creates a copy (deepcopy) of the module and returns ModuleList containing the copies.
def clone_module(module: nn.Module, num_clones: int) -> nn.ModuleList:
    """Creates a deep copy of the module and returns a ModuleList containing the copies.

    Args:
        module (nn.Module): nn.Module to be copied.
        num_clones (int): Number of copies to be created.

    Returns:
        nn.ModuleList: Returns a ModuleList containing the copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_clones)])


# Returns the absolute path if relative path wrt to the repository is provided.
def get_absolute_path(relative_path: str) -> str:
    """Returns the absolute path if the relative path with respect to the repository is provided.

    Args:
        relative_path (str): Relative path with respect to the repository.

    Returns:
        str: Returns the absolute path on the machine.
    """
    # Please note that the implementation of this function depends on the placement of this file and
    # it might not work if this file is moved to a different location.
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_directory, _ = os.path.split(cur_path)
    repo_root, _ = os.path.split(model_directory)
    return os.path.join(repo_root, relative_path)


def get_full_model_path_from_name(model_name: str, checkpoint_prefix: str="") -> str:
    """Constructs the full path to the model from the given model name and a optional checkpoint prefix.

    Args:
        model_name (str): Name of the model for which the path has to be constructed.
        checkpoint_prefix (str, optional): prefix to be appended at the beginning of the model name. Defaults to "".

    Returns:
        str: Returns full path to the model on the machine.
    """
    if checkpoint_prefix == "":
        absolute_path_to_model_state = get_absolute_path(relative_path=f"{MODEL_CHECK_POINT_PATH}/{model_name}.pt")
    else:
        absolute_path_to_model_state = get_absolute_path(relative_path=f"{MODEL_CHECK_POINT_PATH}/{checkpoint_prefix}_{model_name}.pt") 
    return absolute_path_to_model_state


def save_model_to_disk(model: nn.Module, model_name: str, checkpoint_prefix: str=""):
    """Saves the model to disk.

    Args:
        model (nn.Module): The model for which the state has to be saved to the disk.
        model_name (str): Name of the model to be used to save on the disk.
    """
    absolute_path_to_save = get_full_model_path_from_name(model_name=model_name, checkpoint_prefix=checkpoint_prefix)
    torch.save(model.state_dict(), absolute_path_to_save)


def load_model_from_disk(model: nn.Module, model_name: str, checkpoint_prefix: str=""):
    """Loads the model from disk.

    Args:
        model (nn.Module): The model to load the state into. This is necessary because the model has to be initialized
                           before loading the state and we abstract these details from this function.
        model_name (str): Name of the model to be loaded from the disk.
    """
    absolute_path_to_load = get_full_model_path_from_name(model_name=model_name, checkpoint_prefix=checkpoint_prefix)
    model.load_state_dict(torch.load(absolute_path_to_load))


def get_device() -> str:
    """Returns the device on which the model has to be trained.

    Returns:
        str: Returns the device on which the model has to be trained.
    """
    return DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU


def save_lines_to_file(lines: list[str], relative_filepath: str):
    """Saves a list of strings to a file, one string per line.

    Args:
        lines (list[str]): List of strings to save.
        relative_filepath (str): Path to the file (relative to the repository root) where strings should be saved.
    """
    full_filepath = get_absolute_path(relative_path=relative_filepath)
    with open(full_filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


def load_lines_from_file(relative_filepath: str) -> list[str]:
    """Loads strings from a file into a list, one string per line.

    Args:
        relative_filepath (str): Path to the file (relative to the repository root) where strings should be saved.

    Returns:
        list[str]: List of strings read from the file
    """
    full_filepath = get_absolute_path(relative_path=relative_filepath)
    with open(full_filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]