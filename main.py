import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torchinfo import summary

from model import MyModel, ConvNet

from tqdm import tqdm

from utils import *

parser = argparse.ArgumentParser(description=""
                                 )
parser.add_argument("--isTrain", dest="isTrain", action='store_true', help="train or test")
parser.add_argument("--model_dir", dest="model_dir", default="./Model", help="directory to save learned model parameters")

### Data parameters ###
parser.add_argument("--data_path", dest="data_path", type=str, default="./data", help="directory to data")
parser.add_argument("--file_id", dest="file_id", type=str, default="cylindrical_power", help="data file id")

parser.add_argument("--norm_param_file", dest="norm_param_file", type=str, default="./data/normalization_params.txt", help="normalization parameter file")
parser.add_argument("--source_id", dest="source_id", type=int, nargs='+', default=[0], help="source id")
parser.add_argument("--target_id", dest="target_id", type=int, nargs='+', default=[-1], help="target id")

parser.add_argument("--input_dim", dest="input_dim", type=int, default=10, help="the input dimension (used for CNN)")
parser.add_argument("--output_dim", dest="output_dim", type=int, default=1, help="the output dimension. Used for nllloss")
parser.add_argument("--idata_start", dest="idata_start", type=int, default=0, help="the start index of data")
parser.add_argument("--ndata", dest="ndata", type=int, default=1000, help="the number of data")
parser.add_argument("--n_noise", dest="n_noise", type=int, default=0, help="the number of noise")

### Model parameters ###
parser.add_argument("--model", dest="model", default="CNN", help="model")
parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=32, help="number of NN nodes")
parser.add_argument("--n_layer", dest="n_layer", type=int, default=3, help="number of NN layers")
parser.add_argument("--r_drop", dest="r_drop", type=float, default=0.0, help="dropout rate")

### Training parameters ###
parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", dest="epoch", type=int, default=100, help="training epoch")
parser.add_argument("--epoch_decay", dest="epoch_decay", type=int, default=0, help="training epoch")
parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--loss", dest="loss", default="l1norm", help="loss function: nllloss -> density estimation. others -> simple regression")
parser.add_argument("--use_sampler", dest="use_sampler", action='store_true', help="use sampler")

args = parser.parse_args()

def main():

    is_cuda = torch.cuda.is_available()
    random_seed = 1
    if is_cuda:
        device = torch.device("cuda:0")

        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

        print("# GPU is available")
    else:
        device = torch.device("cpu")
        print("# GPU not availabe, CPU used")

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    #torch.use_deterministic_algorithms(True)

    args.n_feature_in = len(args.source_id)
    args.n_feature_out = len(args.target_id)

    if args.isTrain:
        fname = f"{args.model_dir}/args.json"
        with open(fname, "w") as f:
            json.dump(vars(args), f)
        print(f"# Arguments saved to {fname}")

        train(device)
    else:
        test(device)


def update_learning_rate(optimizer, scheduler):
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('# learning rate %.7f -> %.7f' % (old_lr, lr))

####################################################
### training
####################################################

def train(device):
    ### Save arguments ###
    args_dict = vars(args)
    with open(f"{args.model_dir}/args.json", "w") as f:
        json.dump(args_dict, f)
    print(f"# Arguments saved to {args.model_dir}/args.json")

    ### define loss function ###

    if args.loss == "l1norm":
        loss_func = nn.L1Loss(reduction="mean")
        args.output_dim = 1
    elif args.loss == "l2norm":
        loss_func = nn.MSELoss(reduction="mean")
        args.output_dim = 1
    elif args.loss == "bce":
        loss_func = nn.BCELoss(reduction="mean")
        args.output_dim = 1
    elif args.loss == "nllloss":
        if args.output_dim < 2:
            raise ValueError("output_dim must be larger than 1 for nllloss")

        def loss_func(output, target):
            bins = torch.linspace(0, 1, args.output_dim+1, device=target.device)
            bin_indices = torch.bucketize(target, bins) - 1
            bin_indices = bin_indices.clamp(0, args.output_dim-1)
            log_probs = torch.gather(output, dim=2, index=bin_indices.unsqueeze(-1)).squeeze(-1)  # (batch, 3)
            return -log_probs.mean()

    else:
        print("Error: unknown loss", file=sys.stderr)
        sys.exit(1)

    print( f"# loss function: {args.loss}")

    ### define network and optimizer ###
    model = MyModel(args)

    print(model)
    #summary( model, input_size=(args.batch_size, args.n_feature_in, args.input_dim, args.input_dim), col_names=["input_size", "output_size", "num_params"])
    summary( model, input_size=(args.batch_size, args.n_feature_in), col_names=["input_size", "output_size", "num_params"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01) #default: lr=1e-3, betas=(0.9,0.999), eps=1e-8
    def lambda_rule(ee):
        lr_l = 1.0 - max(0, ee + 1 - args.epoch) / float( args.epoch_decay + 1 )
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    model.to(device)

    print( f"# hidden_dim: {args.hidden_dim}" )
    print( f"# n_layer: {args.n_layer}" )

    ### load training and validation data ###
    #data, label, val_data, val_label = load_SDC3b_data(args.data_path, file_id=args.file_id, npix=args.input_dim, norm_param_file=args.norm_param_file, rtrain=0.9, istart=args.idata_start, ndata=args.ndata, add_noise=True, n_noise=args.n_noise, is_train=True, device=device)
    data, label, val_data, val_label = load_AGN_LIM_data(args.data_path, norm_param_file=args.norm_param_file, source_id=args.source_id, target_id=args.target_id, rtrain=0.9, istart=args.idata_start, ndata=args.ndata, is_train=True, device=device)
    
    print( f"# data: {data.size()}")
    print( f"# label: {label.size()}")
    print( f"# val_data: {val_data.size()}")
    print( f"# val_label: {val_label.size()}")

    dataset = MyDataset(data, label)
    
    if args.use_sampler:
        bins = torch.linspace(0, 1, args.output_dim + 1).to(device)
        bin_indices = torch.bucketize(label[:, 0].contiguous(), bins, right=True) - 1
        bin_indices = bin_indices.clamp(0, args.output_dim - 1)
        bin_counts = torch.bincount(bin_indices, minlength=args.output_dim + 1)
        weights = 1.0 / bin_counts[bin_indices]
        weights = weights / weights.sum()
        weights = weights.clamp(0.1, 1.0) # avoid zero weights
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    
    ### training ###
    idx = 0
    print("Training...", file=sys.stderr)
    fout = "{}/log.txt".format(args.model_dir)
    with open(fout, "w") as f:
        print("#idx loss loss_val", file=f)
    for ee in tqdm(range(args.epoch + args.epoch_decay), file=sys.stderr):
        
        for i, (dd, ll) in enumerate(train_loader):
            
            dd = dd.to(device)
            ll = ll.to(device)
            output = model(dd)

            model.eval()
            with torch.no_grad():
                output_val = model(val_data)
            model.train()

            if "weighted" in args.loss:
                weights = w1 * ll + w0 * ( 1. - ll )
                weights_val = w1 * val_label + w0 * ( 1. - val_label )
                loss = torch.mean( nbin * weights / weights.sum() * loss_func(output, ll) )
                loss_val = torch.mean( nbin * weights_val / weights_val.sum() * loss_func(output_val, val_label) )
            else:
                loss = loss_func(output, ll)
                loss_val = loss_func(output_val, val_label)

            log = "{:d} {:f} {:f}".format(idx, loss.item(), loss_val.item())
            #print(log)
            with open(fout, "a") as f:
                print(log, file=f)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            idx += 1

        update_learning_rate(optimizer, scheduler)
 
    ### print validation result ###
    with torch.no_grad():
        output = model(val_data)

        if args.loss == "nllloss":
            for i, (true, pred) in enumerate(zip(val_label, output)):
                fname = "{}/val_{}.txt".format(args.model_dir, i)
                with open(fname, "w") as f:
                    print("# ", end="", file=f)
                    for k in range(args.n_feature_out):
                        print(f"{true[k].item()} ", end="", file=f)   
                    print("", file=f)
                    for j in range(args.output_dim):
                        for k in range(args.n_feature_out):
                            print(f"{pred[k, j].item()} ", end="", file=f)
                        print("", file=f)
                    print(f"# output {fname}", file=sys.stderr)
        else:
            fname = "{}/val.txt".format(args.model_dir)
            with open(fname, "w") as f:
                for i, (true, pred) in enumerate(zip(val_label, output)):
                    for j in range(args.n_feature_out):
                        print(f"{true[j].item()} {pred[j].item()} ", end="", file=f)
                    print("", file=f)
            print(f"# output {fname}", file=sys.stderr)

    ### save model ###
    fsave = "{}/model.pth".format(args.model_dir)
    torch.save(model.state_dict(), fsave)
    print( f"# save {fsave}" )

####################################################
### test
####################################################
def test(device):

    ### define network ###
    model = MyModel(args)
    model.to(device)

    fmodel = "{}/model.pth".format(args.model_dir)
    model.load_state_dict(torch.load(fmodel))
    model.eval()
    print("# load model from {}/model.pth".format(args.model_dir))

    ### load test data ###
    
    #data, label, _, _ = load_SDC3b_data(args.data_dir, file_id=args.file_id, source_id=args.source_id, npix=args.input_dim, norm_param_file=args.norm_param_file, istart=args.idata_start, ndata=args.ndata, add_noise=True, is_train=False, device=device)
    data, label, val_data, val_label = load_AGN_LIM_data(args.data_path, norm_param_file=args.norm_param_file, source_id=args.source_id, target_id=args.target_id, rtrain=0.9, istart=args.idata_start, ndata=args.ndata, is_train=True, device=device)
    
    ### output test result ###
    if args.loss == "nllloss":
        for i, (dd, ll) in enumerate(zip(data, label)):

            data_id = i + args.idata_start

            dd = torch.unsqueeze(dd, dim=0).to(device)
            ll = torch.unsqueeze(ll, dim=0).to(device)

            with torch.no_grad():
                output = model(dd)
            pred = output 
            true = ll 

            fname = "{}/test_{}_{:d}.txt".format(args.model_dir, args.file_id, data_id)
            with open(fname, "w") as f:
        
                print("# ", end="", file=f)
                for k in range(args.n_feature_out):
                    print(f"{true[0,k].item()} ", end="", file=f)   
                print("", file=f)
                for j in range(args.output_dim):
                    for k in range(args.n_feature_out):
                        print(f"{pred[0, k, j].item()} ", end="", file=f)
                    print("", file=f)
                print(f"# output {fname}", file=sys.stderr)
            print(f"# output {fname}", file=sys.stderr)
    else:

        fname = "{}/test_{}.txt".format(args.model_dir, args.file_id)
        with open(fname, "w") as f:
            for dd, ll in zip(data, label):

                dd = torch.unsqueeze(dd, dim=0).to(device)
                ll = torch.unsqueeze(ll, dim=0).to(device)

                with torch.no_grad():
                    output = model(dd)
                pred = output 
                true = ll 

                for j in range(args.n_feature_out):
                    print(f"{true[0,j].item()} {pred[0,j].item()} ", end="", file=f)
                print("", file=f)


        print(f"output {fname}", file=sys.stderr)

if __name__ == "__main__":
    main()
