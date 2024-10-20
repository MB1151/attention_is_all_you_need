# This script is created just to show how to avoid the error (on Windows and Mac) we encountered 
# in the notebook 'step_3_datasets_and_dataloaders_pytorch.ipynb' while trying to load the data 
# using multiple workers. I created a python script since it seems like notebooks handle 
# multiprocessing differently from scripts and I couldn't resolve the issue within Jupyter 
# notebooks.

# %%
import torch
from torch.utils.data import Dataset, DataLoader

# %%
# Creating a SimpleDataset to be used in assocation with pytorch 'DataLoader'.
# 'DataLoader' uses 'Dataset' types to iterate on the dataset.
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# %%
# We define a very simple custom collate_fn that gets a batch of input tuples (features, target)
# and combines them into a batched tensor.
def custom_collate_fn(batch):
    # batch is a list of (feature, target) pairs.
    print(f"In custom_collate_fn:batch: {batch}, type(batch): {type(batch)}")
    # Refer the above 2 cells to understand this operation.
    features, labels = zip(*batch)
    features = torch.tensor(features, dtype=torch.float32)
    print(f"In custom_collate_fn:features: {features}")
    labels = torch.tensor(labels, dtype=torch.float32)
    print(f"In custom_collate_fn:labels: {labels}")
    return features, labels
    

# %%
if __name__ == "__main__":
    data = [(i, i + 1) for i in range(12)]
    my_dataset = SimpleDataset(data=data)
    my_dataloader = DataLoader(dataset=my_dataset, num_workers=2, batch_size=4, collate_fn=custom_collate_fn)
    for features, labels in my_dataloader:
        print(f"features: {features}, type(features): {type(features)}")
        print(f"labels: {labels}, type(labels): {type(labels)}")

# %%



