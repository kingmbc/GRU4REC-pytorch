import os
import lib
import time
import torch
import numpy as np
from tqdm import tqdm
import wandb

class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, batch_size, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = lib.Evaluation(self.model, self.loss_func, use_cuda, k = args.k_eval)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        self.args = args

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        self.best_result = [0, 0]
        self.best_epoch = [0, 0]
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            print('Start Epoch #', epoch)
            train_loss = self._train_epoch(epoch)
            valid_loss, valid_recall, valid_mrr = self.evaluation.eval(self.eval_data, self.batch_size)

            checkpoint = {
                'state_dict': self.model.state_dict(),
                'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
                'valid_loss': valid_loss,
                'valid_recall': valid_recall,
                'valid_mrr': valid_mrr
            }

            model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            torch.save(checkpoint, model_name)
            if self.args.wandb_on:
                wandb.log({'epoch': epoch,
                           'train_loss': train_loss, 'valid_loss': valid_loss,
                           'valid_recall': valid_recall, 'valid_mrr': valid_mrr,
                           'time': time.time() - st})
                self._log_best_result(epoch, valid_recall, valid_mrr)
                wandb.save(model_name)
            print("Save model as %s" % model_name)

    def _log_best_result(self, epoch, recall, mrr):
        if not self.args.wandb_on:
            return

        if recall >= self.best_result[0]:
            self.best_result[0] = recall
            self.best_epoch[0] = epoch
        if mrr >= self.best_result[1]:
            self.best_result[1] = mrr
            self.best_epoch[1] = epoch

        wandb.log({"best_recall": self.best_result[0],
                   "best_mrr": self.best_result[1],
                   "best_recall_epoch": self.best_epoch[0],
                   "best_mrr_epoch": self.best_epoch[1]})


    def _train_epoch(self, epoch):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()
        dataloader = lib.DataLoader(self.train_data, self.batch_size)
        #for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):
        for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            input = input.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            hidden = reset_hidden(hidden, mask).detach() #hidden=(num_layer, batch_size, hidden_size)
            logit, hidden = self.model(input, hidden)
            # output sampling
            logit_sampled = logit[:, target.view(-1)] #logit_sampled=(batch_size, batch_size)
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        if isinstance(losses, torch.Tensor):
            mean_losses = torch.mean(losses)
        else:
            mean_losses = np.mean(losses)
        return mean_losses