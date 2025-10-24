import numpy as np
from torch.utils.data import Dataset

class SimulationDataset(Dataset):
    def __init__(self, datasets, intervention_sets):
        super(SimulationDataset, self).__init__()
        self.datasets = datasets
        self.intervention_sets = intervention_sets 

        self.load_data()
    
    def load_data(self):
        self.data = np.vstack(self.datasets)
        masks_list = list()
        regimes_list = list()
        for i, dataset in enumerate(self.datasets):
            targets = self.intervention_sets[i]
            mask = np.ones_like(dataset)
            regimes_list += [i] * len(dataset)
            if targets[0] != None:
                mask[:, targets] = 0 # 0 - intervened nodes, 1 - purely observed nodes

            masks_list.append(mask)
        
        self.masks = np.vstack(masks_list)
        self.regimes = regimes_list
    
    def __getitem__(self, idx):
        return (
            self.data[idx].astype(np.float32),
            self.masks[idx].astype(np.float32),
            self.regimes[idx]
        )

    def __len__(self):
        return len(self.data)
    
class InterventionDataset(Dataset):
    def __init__(self, data, intervention_targets):
        super(InterventionDataset, self).__init__()
        self.data = data
        self.intervention_targets = intervention_targets 

        self.masks = np.ones_like(self.data)
        if self.intervention_targets[0] != None:    
            self.masks[:, self.intervention_targets] = 0

    def __getitem__(self, idx):
        return (
            self.data[idx].astype(np.float32),
            self.masks[idx].astype(np.float32)
        )
    
    def __len__(self):
        return len(self.data)