#!/usr/bin/env python
# coding: utf-8
import torch
from torch.utils import data
import h5py
import numpy as np
from torch.utils.data import Dataset



class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, train_size, transform=None):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self.hdf5file = h5py.File(self.file_path, 'r')
        
        if train_size > self.hdf5file['hcal_only']['layers'].shape[0]-1:
            self.train_size = self.hdf5file['hcal_only']['layers'].shape[0]-1
        else:
            self.train_size = train_size
        
        #print('layers shape')
        #print(self.hdf5file['30x30']['layers'].shape)
        #print('energy shape')
        #print(self.hdf5file['30x30']['energy'].shape)
        
        # Search for all h5 files
        
    def __len__(self):
        return self.train_size
        #return self.hdf5file['hcal_only']['layers'][0:self.train_size].shape[0]
             
    def __getitem__(self, index):
        # get data
        x = self.get_data(index)
        if self.transform:
            x = torch.from_numpy(self.transform(x)).float()
        else:
            x = torch.from_numpy(x).float()
        e = torch.from_numpy(self.get_energy(index))
        if torch.sum(x) != torch.sum(x): #checks for NANs
            return sefl.__getitem__(int(np.rand()*self.__len__()))
        else:
            return x, e

            
    def get_data(self, i):
        return self.hdf5file['hcal_only']['layers'][i]
    
    def get_energy(self, i):
        return self.hdf5file['hcal_only']['energy'][i]

    def get_data_range(self, i, j):
        return self.hdf5file['hcal_only']['layers'][i:j]
    
    def get_energy_range(self, i, j):
        return self.hdf5file['hcal_only']['energy'][i:j]

    def get_data_range_tf(self, i, j):
        x = self.get_data_range(i,j)
        print(x.shape)
        #x = np.transpose(x, (1, 2, 3, 0))
        x =self.transform(x)
        print(x.shape)
        return x #np.transpose(x, (2, 0, 1))




class HDF5DatasetManualBatching(data.Dataset):
    def __init__(self, file_path, train_size, batch_size, transform=None, shuffle=False):
        
        
        self.file_path = file_path
        self.transform = transform
        self.hdf5file = h5py.File(self.file_path, 'r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if train_size > self.hdf5file['hcal_only']['layers'].shape[0]-1:
            self.train_size = self.hdf5file['hcal_only']['layers'].shape[0]-1
        else:
            self.train_size = train_size
            
        self.number_batches = int(self.train_size//self.batch_size)
        self.train_size = self.number_batches*self.batch_size

        self.dataset = np.zeros((self.train_size, 1))

        self.batch_indices = np.arange(start=0, stop=train_size+1, step=batch_size)
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

            
    def get_data(self, i):
        return self.hdf5file['hcal_only']['layers'][i]
    
    def get_energy(self, i):
        return self.hdf5file['hcal_only']['energy'][i]

    def get_data_range(self, i, j):
        return self.hdf5file['hcal_only']['layers'][i:j]
    
    def get_energy_range(self, i, j):
        return self.hdf5file['hcal_only']['energy'][i:j]
    
        
    def __len__(self):
        return self.number_batches
        #return self.hdf5file['hcal_only']['layers'][0:self.train_size].shape[0]
             
    def __getitem__(self, index):
        # get data
        x = self.get_data_range(self.batch_indices[index], self.batch_indices[index]+self.batch_size)
        e = self.get_energy_range(self.batch_indices[index], self.batch_indices[index]+self.batch_size)
        
        if self.transform:
            x = torch.from_numpy(self.transform(x)).float()
        else:
            x = torch.from_numpy(x).float()
        e = torch.from_numpy(e)
            
        return x, e    
    
    
class BatchLoader:

    """Iterator that counts upward forever."""

    def __init__(self, file_path, train_size, batch_size, shuffle=True, transform=None):
        
        self.file_path = file_path
        self.transform = transform
        self.hdf5file = h5py.File(self.file_path, 'r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if train_size > self.hdf5file['hcal_only']['layers'].shape[0]-1:
            self.train_size = self.hdf5file['hcal_only']['layers'].shape[0]-1
        else:
            self.train_size = train_size
            
        self.train_size = (self.train_size//self.batch_size)*self.batch_size
        self.dataset = np.zeros((self.train_size, 1))

        self.batch_indices = np.arange(start=0, stop=train_size+1, step=batch_size)
        if self.shuffle:
            np.random.shuffle(self.batch_indices)

        self.index = 0
            
    def __len__(self):
        return self.train_size
            
    def get_data(self, i):
        return self.hdf5file['hcal_only']['layers'][i]
    
    def get_energy(self, i):
        return self.hdf5file['hcal_only']['energy'][i]

    def get_data_range(self, i, j):
        return self.hdf5file['hcal_only']['layers'][i:j]
    
    def get_energy_range(self, i, j):
        return self.hdf5file['hcal_only']['energy'][i:j]


    def __iter__(self):
        return self

    def __next__(self):
        # get data
        if self.index+1 >= len(self.batch_indices):
            if self.shuffle:
                np.random.shuffle(self.batch_indices)
            self.index = 0
            raise StopIteration
        
        x = self.get_data_range(self.batch_indices[self.index], self.batch_indices[self.index]+self.batch_size)
        e = self.get_energy_range(self.batch_indices[self.index], self.batch_indices[self.index]+self.batch_size)
        
        if self.transform:
            x = torch.from_numpy(self.transform(x)).float()
        else:
            x = torch.from_numpy(x).float()
        e = torch.from_numpy(e)
        
        self.index += 1


            
        return x, e
    
