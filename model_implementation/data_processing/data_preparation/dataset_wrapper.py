# This file implements a dataset wrapper. It takes in hugging face dataset and creates a wrapper around it
# to be uesd with DataLoaders. Pytorch requires the data to be 'torch.utils.data.Dataset' in order to 
# integrate it with the pytorch DataLoaders.

from typing import Dict, Optional

import datasets
import torch

class DatasetWrapper(torch.utils.data.Dataset): # type: ignore
    def __init__(self, hf_dataset: datasets.arrow_dataset.Dataset, dataset_name: Optional[str]=None):
        """Initializes the DatasetWrapper with the given dataset.

        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): The hugging face dataset to be wrapped.
        """
        self.dataset = hf_dataset
        # This is saved just for informational purposes since we have multiple datasets.
        self.dataset_name = dataset_name
    
    def __len__(self) -> int:
        """Extracts the number of examples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.dataset.num_rows
    
    def __getitem__(self, index: int) -> Dict[str, str]:
        """Extracts the data_point (example translation pair) at a particular index in the dataset.

        Args:
            index (int): Index of the data_point (example translation pair) to be extracted from the dataset.

        Returns:
            dict: Data_point at the given index in the dataset. This turns out to be a dictionary for our dataset but
                  it could be any type in general.
        """
        # Return the dataset at a particular index.
        # The index provided will always be less then length (64 in this case) returned by __len__ function.
        return self.dataset[index]