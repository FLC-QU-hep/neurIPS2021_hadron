import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PionsDataset(Dataset):

    def __init__(self, path_list, core=False):
        self.path_list = path_list
        self.core = core
        part = np.array([])
        index = np.array([])
        for i,path in enumerate(path_list):
            file = h5py.File(path, 'r')['hcal_only/energy']
            part = np.append(part, np.ones(len(file))*i)
            index = np.append(index, np.arange(len(file)))


        self.keys = pd.DataFrame({'part' : part.astype(int),
                                  'index' : index.astype(int)
                                 })

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        part = self.keys['part'][idx]
        index = self.keys['index'][idx]
        path = self.path_list[part]

        file = h5py.File(path, 'r')['hcal_only']
        energy = file['energy'][index]
        shower = file['layers'][index]
        
        if self.core:
            shower = shower[:, 13:38, 11:36]
            

        layer_energy = shower.sum(axis=1).sum(axis=1)
        try:
            free_path = 49 - np.where(np.diff(layer_energy) > 11)[0][0]
        except:
            free_path = 1

        energy = energy.reshape(1,1,1,1)
        shower = np.expand_dims(shower, 0)

        return {'energy' : energy,
                'shower' : shower,
                'free_path' : free_path}
