import os
import sys
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from tqdm import tqdm
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def preprocess(images, labels, norm_param_file='mean_std.txt', is_train=True, for_estimate_only=False):
    images = np.array(images)
    labels = np.array(labels)
    
    if norm_param_file == None:
        print("# No normalization")
        return images, labels

    if is_train:
        with open(norm_param_file, 'w') as f:
            mean = np.mean(images)
            std = np.std(images)
            f.write(f"{mean} {std}\n")

            images = (images - mean) / std

            if not for_estimate_only:
                for i in range(labels.shape[1]):
                    min_val = np.min(labels[:,i])
                    max_val = np.max(labels[:,i])
                    f.write(f"{min_val} {max_val}\n")

                    #labels[:,i] = (labels[:,i] - min_val) / (max_val - min_val)

        print(f"# Save statistics to {norm_param_file}")

    else:
        with open(norm_param_file, 'r') as f:
            for i, line in enumerate(f):
                val1, val2 = line.split()
                val1 = float(val1)
                val2 = float(val2)
            
                if i == 0:
                    images = (images - val1) / val2
                else:
                    pass
                    #labels[:,i-1] = (labels[:,i-1] - val1) / (val2 - val1)           
                
        print(f"# Load statistics from {norm_param_file}")
    
    return images, labels


def convert_to_torch(images, labels, rtrain=0.9, device="cpu"):
    images = torch.from_numpy(np.array(images).astype(np.float32))
    labels = torch.from_numpy(np.array(labels).astype(np.float32))

    images_train, images_val = images[:int(rtrain*len(images)), :, :], images[int(rtrain*len(images)):, :, :]
    labels_train, labels_val = labels[:int(rtrain*len(images)), :], labels[int(rtrain*len(images)):, :]
    
    images_train = images_train.to(device)
    images_val = images_val.to(device)
    labels_train = labels_train.to(device)
    labels_val = labels_val.to(device)

    return images_train, labels_train, images_val, labels_val

def load_SDC3b_data(data_dir, n_feature=3, npix=10, norm_param_file=None, is_train=False, rtrain=1, istart=0, ndata=10000, device=None):
    data_list = []
    label_list = []
    for i in range(istart, istart+ndata):
        data_now = []
        label_now = []
        for j in range(n_feature):
            filename = f"{data_dir}/cylindrical_power_{i}_{j}.txt"
            data = np.loadtxt(filename, skiprows=1)
            x_HI = np.loadtxt(filename, max_rows=1)

            data = data[:npix, :npix]

            data_now.append(data)
            label_now.append(x_HI)

        data_list.append(data_now)
        label_list.append(label_now)

    data, label = preprocess(data_list, label_list, norm_param_file=norm_param_file, is_train=is_train)

    return convert_to_torch(data, label, rtrain=rtrain, device=device)

def load_jwst_data(ndata=2240, parent_dir="./COSMOS_web_galaxies", device=None):
    cosmos_cata=pd.read_csv(parent_dir+'/cd3_catalog.txt', " ")
    source_id=cosmos_cata['ID']
    source_ra=cosmos_cata['RA']
    source_dec=cosmos_cata['Dec']
    source_z=cosmos_cata['z']

    map_id = 1 #1: data, 3: noise
    data_all = []
    
    print("Loading data...")
    for ID in range(ndata):
        #img data
        import glob
        if ID == 27 or ID == 1049:
            data_now = np.zeros((4, 106, 106))
        else:
            fits115=glob.glob(parent_dir+"/cd3_cutout/{0}_NIRCAM_F115W_cutout*.fits".format(ID))[0]
            fits150=glob.glob(parent_dir+"/cd3_cutout/{0}_NIRCAM_F150W_cutout*.fits".format(ID))[0]
            fits277=glob.glob(parent_dir+"/cd3_cutout/{0}_NIRCAM_F277W_cutout*.fits".format(ID))[0]
            fits444=glob.glob(parent_dir+"/cd3_cutout/{0}_NIRCAM_F444W_cutout*.fits".format(ID))[0]
            fits115data=fits.open(fits115)[1].data
            fits150data=fits.open(fits150)[1].data
            fits277data=fits.open(fits277)[1].data
            fits444data=fits.open(fits444)[1].data
            data_now = np.array([fits115data, fits150data, fits277data, fits444data])
            data_now = np.reshape(data_now, (4, 106, 106))
        print(ID, np.shape(data_now))

        data_all.append(data_now)        
    data_all = np.array(data_all)
    print("aaa: ", np.shape(data_all))
    # data_all: (N, 4, Npix, Npix)

    data_all = torch.from_numpy( np.array(data_all).astype(np.float32) )
    source_z = torch.from_numpy( np.array(source_z) )
    ### send the data to device ###
    if device is not None:
        data_all = data_all.to(device)
        source_z = source_z.to(device)

    return data_all, source_z        

def calc_weight(pdf, values, xmin, xmax):
    dx = ( xmax - xmin ) / len(pdf)

    weights = torch.zeros(len(values))
    for i, x in enumerate(values):
        ix = int( ( x - xmin ) / dx )
        ix = np.clip(ix, 0, len(pdf)-1)
        weights[i] = 1. - pdf[ix]
    return torch.reshape(weights, (-1,1))

def print_pdf(pdf, xmin, xmax):
    dx = (xmax - xmin) / len(pdf)
    print("#### label distribution ####") 
    for i, p in enumerate(pdf):
        print("# {:e} {:e}".format(xmin+dx*(i+0.5), p))
    print("#### end of label distribution ####")


def load_fnames(data_dir, ndata, id_start=1, r_train = 0.9, shuffle=True):

    id_list = np.array(range(ndata))
    if shuffle == True:
        np.random.shuffle(id_list)

    ids_train = [ i for i in id_list[:int(ndata * r_train)] ]
    ids_val = [ i for i in id_list[int(ndata * r_train):] ]

    fnames_train = [ "{}/{:07d}.0.data".format(data_dir, i+id_start) for i in ids_train ]
    fnames_val = [ "{}/{:07d}.0.data".format(data_dir, i+id_start) for i in ids_val ]

    if len(ids_val) == 0:
        ids_val = [ids_train[-1]]
        fnames_val = [fnames_train[-1]]

    return fnames_train, fnames_val, ids_train, ids_val

def load_data(fnames, data_ids, fname_comb="./Combinations.txt", output_dim=100, output_id=[13], n_feature=1, seq_length=10, norm_params=None, loss="l1norm", data_aug=[], device="cpu"):

    if len(np.shape(norm_params)) == 1:
        norm_params = norm_params.reshape(1,-1)

    print(f"Reading files... (data_aug = {data_aug})", file=sys.stderr)

    ### read input data ###
    data = []
    for f in fnames:
        if os.path.exists(f) == False: 
            print(f"# Error: file not found {f}", file=sys.stderr) 
            sys.exit(1)

        for da in [0] + data_aug:
            if da == 0:
                d = read_data( f, norm_params=norm_params)
            else: 
                print("data augmentation {} is not defined")
                sys.exit(1)

            data.append(d)
    
    ### read label data ###
    label = np.loadtxt(fname_comb, skiprows=0, usecols=output_id)
    label = label[data_ids] # (ndata)
    xmin = 0.0
    xmax = 90.001
    if loss == "nllloss":
        dx = ( xmax - xmin ) / output_dim
        label = [ int( ( l - xmin ) / dx ) for l in label ]  ### this is for nllloss and doesn't work properly for other losses.
        if np.max( label ) >= output_dim:
            print(f"Error: label value {np.max(label)} is greater than output_dim {output_dim}")
            sys.exit(1)
    else:
        label = ( label - xmin ) / ( xmax - xmin )
        if len(np.shape(label)) == 1:
            label = label.reshape(-1, 1) #(ndata, 1) within [0,1]
        if np.shape(label)[1] != output_dim:
            print(f"Error: inconsistent output_dim {np.shape(label)[1]} != {output_dim}", file=sys.stderr)
            sys.exit(1)

    if np.shape(data)[1] != seq_length:
        print(f"Error: inconsistent seq_length {np.shape(data)[1]} != {seq_length}", file=sys.stderr)
        sys.exit(1)

    if np.shape(data)[2] != n_feature:
        print(f"Error: inconsistent n_feature {np.shape(data)[2]} != {n_feature}", file=sys.stderr)
        sys.exit(1)

    ### convert the data to torch.tensor ###
    data = torch.from_numpy( np.array(data).astype(np.float32) )
    label = torch.from_numpy( np.array(label) )
    if loss == "nllloss":
        label = label.to(torch.long)
    else:
        label = label.to(torch.float32)

    ### send the data to device ###
    if device is not None:
        data = data.to(device)
        label = label.to(device)

    print( f"# data size: {data.size()}" )
    print( f"# label size: {label.size()}" )

    return data, label

def read_data(path, norm_params=None):

    ## source data ## 
    data = np.loadtxt(path)
    input_data = data[:,1] ## read "spec"
    if len(np.shape(input_data)) == 1:
        input_data = input_data.reshape(-1,1) #(seq_length, n_feature=1)

    if norm_params is not None: 
        for i in range(np.shape(norm_params)[0]):
            input_data[:,i] -= norm_params[i,0]
            if norm_params[i,1] > 0: input_data[:,i] /= norm_params[i,1]

    return input_data

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label):
        self.transform = None #transforms.Compose([transforms.ToTensor()])
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            out_data = self.transform(self.data)[0][idx]
            out_label = self.label[idx]
        else:
            out_data = self.data[idx]
            out_label = self.label[idx]
        return out_data, out_label
            


