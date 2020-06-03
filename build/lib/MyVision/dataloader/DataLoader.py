from torch.utils.data import DataLoader, Dataset


class SimpleDataLoader:
    def __call__(self, dataset: Dataset, batch_size: int, shuffle: bool, phase: str):

        if phase == "train":
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        if phase == "valid":
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        if phase == "test":
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
