import argparse
import os
import torch
import numpy as np
import pandas as pd
import datetime
import wandb

import lib

parser = argparse.ArgumentParser()
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
parser.add_argument('--wandb_project', default='GRU4Rec Project', type=str)
parser.add_argument('--wandb_on', default="True", type=str2bool)
parser.add_argument('--debug', default="False", type=str2bool)
# Model argument
parser.add_argument('--model_name', default='GRU4Rec', type=str)
parser.add_argument('--num_layers', default=1, type=int)        # 1 hidden layer
parser.add_argument('--batch_size', default=50, type=int)       # 50 in first paper and 32 in second paper
parser.add_argument('--hidden_size', default=100, type=int)     # Literature uses 100 / 1000 --> better is 100
parser.add_argument('--dropout_input', default=0.5, type=float)   # 0.5 for TOP and 0.3 for BPR
parser.add_argument('--dropout_hidden', default=0.5, type=float)# 0.5 for TOP and 0.3 for BPR
parser.add_argument('--embedding_dim', default=-1, type=int, help="using embedding") #TODO:Need to tune
# Optimizer arguments
parser.add_argument('--k_eval', default=20, type=int)           # value of K during Recall and MRR Evaluation
parser.add_argument('--n_epochs', default=5, type=int)          # number of epochs (10 in literature)
parser.add_argument('--optimizer_type', default='Adagrad', type=str)# Optimizer --> Adagrad is the best according to literature
parser.add_argument('--final_act', default='tanh', type=str)        # Final Activation Function
parser.add_argument('--lr', default=0.01, type=float)               # learning rate (Best according to literature 0.01 to 0.05)
parser.add_argument('--weight_decay', default=0, type=float)        # no weight decay
parser.add_argument('--momentum', default=0, type=float)            # no momentum
parser.add_argument('--eps', default=1e-6, type=float)              # not used
parser.add_argument('--seed', default=22, type=int, help="Seed for random initialization")          #Random seed setting
parser.add_argument('--sigma', default=None, type=float,             # weight initialization [-sigma sigma] in literature
                    help="init weight   -1: range [-sigma, sigma], -2: range [0, sigma]")
# Loss arguments
parser.add_argument('--loss_type', default='TOP1-max', type=str)    # type of loss function TOP1 / BPR / TOP1-max / BPR-max
# Data arguments
parser.add_argument('--data_folder', default='../_data/retailrocket-prep', type=str)
parser.add_argument('--train_data', default='retailrocket-train.csv.sample', type=str)
parser.add_argument('--valid_data', default='retailrocket-valid.csv.sample', type=str)
parser.add_argument('--item2idx_dict', default='item_idx_dict_filtered.pkl', type=str)
# Logging, environment, etc.
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--time_sort', default=False, type=bool)        # In case items are not sorted by time stamp
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--is_eval', action='store_true')               # should be used during testing and eliminated during training
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)


# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = torch.device('cuda' if args.cuda else 'cpu')
#use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

def make_checkpoint_dir():
    """ Write Checkpoints with arguments used in a text file for reproducibility
    """
    print("PARAMETER" + "-"*10)
    now = datetime.datetime.now()
    S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    save_dir = os.path.join(args.checkpoint_dir, S)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.checkpoint_dir = save_dir
    with open(os.path.join(args.checkpoint_dir, 'parameter.txt'), 'w') as f:
        for attr, value in sorted(args.__dict__.items()):
            print("{}={}".format(attr.upper(), value))
            f.write("{}={}\n".format(attr.upper(), value))
    print("---------" + "-"*10)

def init_model_weight(model):
    """ Weight initialization if it was defined
    """
    if args.sigma is not None:
        for p in model.parameters():
            if args.sigma != -1 and args.sigma != -2:
                sigma = args.sigma
                p.data.uniform_(-sigma, sigma)
            elif len(list(p.size())) > 1:
                sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
                if args.sigma == -1:
                    p.data.uniform_(-sigma, sigma)
                else:
                    p.data.uniform_(0, sigma)


def main():
    if args.wandb_on:
        wandb.init(project=args.wandb_project,
                   name=args.model_name + '-' + args.data_folder.split('/')[2] + '-' + args.loss_type)
        wandb.config.update({'hostname': os.popen('hostname').read().split('.')[0]})
        wandb.config.update(args)
    if args.item2idx_dict is not None:
        item2idx_dict = pd.read_pickle(os.path.join(args.data_folder, args.item2idx_dict))
    else:
        item2idx_dict = None

    print("Loading train data from {}".format(os.path.join(args.data_folder, args.train_data)))
    print("Loading valid data from {}".format(os.path.join(args.data_folder, args.valid_data)))

    train_data = lib.Dataset(os.path.join(args.data_folder, args.train_data))
    valid_data = lib.Dataset(os.path.join(args.data_folder, args.valid_data), itemmap=train_data.itemmap)

    if args.debug:
        train_data.df.to_csv(os.path.join(args.data_folder, 'GRU4Rec-train-data.csv'))
        valid_data.df.to_csv(os.path.join(args.data_folder, 'GRU4Rec-valid-data.csv'))
    make_checkpoint_dir()
        
    #set all the parameters according to the defined arguments
    args.input_size = len(train_data.items)
    args.output_size = args.input_size

    #loss function
    loss_function = lib.LossFunction(loss_type=args.loss_type, use_cuda=args.cuda) #cuda is used with cross entropy only
    if not args.is_eval: #training
        #Initialize the model
        model = lib.GRU4REC(args.input_size, args.hidden_size, args.output_size, final_act=args.final_act,
                            num_layers=args.num_layers, use_cuda=args.cuda, batch_size=args.batch_size,
                            dropout_input=args.dropout_input, dropout_hidden=args.dropout_hidden, embedding_dim=args.embedding_dim)
        #weights initialization
        init_model_weight(model)
        if args.wandb_on:
            wandb.watch(model, log="all")

        #optimizer
        optimizer = lib.Optimizer(model.parameters(), optimizer_type=args.optimizer_type, lr=args.lr,
                                  weight_decay=args.weight_decay, momentum=args.momentum, eps=args.eps)
        #trainer class
        trainer = lib.Trainer(model, train_data=train_data, eval_data=valid_data, optim=optimizer,
                              use_cuda=args.cuda, loss_func=loss_function, batch_size=args.batch_size, args=args)
        print('#### START TRAINING....')
        trainer.train(0, args.n_epochs - 1)
    else: #testing
        if args.load_model is not None:
            print("Loading pre-trained model from {}".format(args.load_model))
            try:
                checkpoint = torch.load(args.load_model)
            except:
                checkpoint = torch.load(args.load_model, map_location=lambda storage, loc: storage)
            model = lib.GRU4REC(args.input_size, args.hidden_size, args.output_size, final_act=args.final_act,
                                num_layers=args.num_layers, use_cuda=args.cuda, batch_size=args.batch_size,
                                dropout_input=args.dropout_input, dropout_hidden=args.dropout_hidden,
                                embedding_dim=args.embedding_dim)
            model.load_state_dict(checkpoint["state_dict"])

            model.gru.flatten_parameters()
            evaluation = lib.Evaluation(model, loss_function, use_cuda=args.cuda, k=args.k_eval)
            loss, recall, mrr = evaluation.eval(valid_data, args.batch_size)
            print("Final result: recall = {:.2f}, mrr = {:.2f}".format(recall, mrr))
        else:
            print("No Pretrained Model was found!")


if __name__ == '__main__':
    main()
