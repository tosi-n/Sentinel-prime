import logging
import math
import os
import sys
from time import strftime, localtime
import random
import numpy

from pytorch_transformers import BertModel
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset

from config.pre_process import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset, prepare_data

from aen import CrossEntropyLoss_LSR, AEN_BERT

from config.global_args import global_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


    
class tsc:
    def __init__(self, model_name):

        tokenizer = Tokenizer4Bert(global_args['max_seq_length'], global_args['pretrained_bert_name'])
        bert = BertModel.from_pretrained(global_args['pretrained_bert_name'])
        self.pretrained_bert_state_dict = bert.state_dict()
        self.model = model_name(bert).to(global_args['device'])

        self.trainset = ABSADataset(global_args['trainset'], tokenizer)
        self.devset = ABSADataset(global_args['devset'], tokenizer)

        # if global_args['device'] == 'cuda':
        #     logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=global_args['device'].index)))
        # self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for global_arg in vars(global_args):
            logger.info('>>> {0}: {1}'.format(global_args, getattr(global_args, global_arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
            else:
                self.model.bert.load_state_dict(self.pretrained_bert_state_dict)

    def train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(global_args['num_epoch']):
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched.get(key).to(global_args['device']) for key in ['text_raw_bert_indices', 'aspect_bert_indices']]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(global_args['device'])

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % global_args['log_step'] == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self.evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_temp'.format(global_args['model_name'], global_args['dataset'])
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched.get(key).to(global_args['device']) for key in ['text_raw_bert_indices', 'aspect_bert_indices']]
                t_targets = t_sample_batched['polarity'].to(global_args['device'])
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def train_model(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        # criterion = CrossEntropyLoss_LSR()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = global_args['optimizer'](_params, lr=global_args['learning_rate'], weight_decay=global_args['l2reg'])

        test_data_loader = DataLoader(dataset=self.devset, batch_size=global_args['batch_size'], shuffle=False)
        valset_len = len(self.trainset) // global_args['cross_val_fold']
        splitedsets = random_split(self.trainset, tuple([valset_len] * (global_args['cross_val_fold'] - 1) + [len(self.trainset) - valset_len * (global_args['cross_val_fold'] - 1)]))

        all_test_acc, all_test_f1 = [], []
        for fid in range(global_args['cross_val_fold']):
            logger.info('fold : {}'.format(fid))
            logger.info('>' * 100)
            trainset = ConcatDataset([x for i, x in enumerate(splitedsets) if i != fid])
            valset = splitedsets[fid]
            train_data_loader = DataLoader(dataset=trainset, batch_size=global_args['batch_size'], shuffle=True)
            val_data_loader = DataLoader(dataset=valset, batch_size=global_args['batch_size'], shuffle=False)

            self._reset_params()
            best_model_path = self.train(criterion, optimizer, train_data_loader, val_data_loader)

            self.model.load_state_dict(torch.load(best_model_path))
            test_acc, test_f1 = self.evaluate_acc_f1(test_data_loader)
            all_test_acc.append(test_acc)
            all_test_f1.append(test_f1)
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

        mean_test_acc, mean_test_f1 = numpy.mean(all_test_acc), numpy.mean(all_test_f1)
        logger.info('>' * 100)
        logger.info('>>> mean_test_acc: {:.4f}, mean_test_f1: {:.4f}'.format(mean_test_acc, mean_test_f1))




class load_n_predict():
    def __init__(self, model_name):
        self.tokenizer = Tokenizer4Bert(global_args['max_seq_length'], global_args['pretrained_bert_name'])
        bert = BertModel.from_pretrained(global_args['pretrained_bert_name'])
        self.model = model_name(bert).to(global_args['device'])

        print('loading model {0} ...'.format(global_args['model_name']))
        self.model.load_state_dict(torch.load(global_args['state_dict_path']))
        self.model.eval()
        torch.autograd.set_grad_enabled(False)


    def predict(self, text_left, aspect, text_right):
        text_raw_bert_indices, aspect_bert_indices = prepare_data(text_left, aspect, text_right, self.tokenizer)
        text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(global_args['device'])
        aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(global_args['device'])

        inputs = [text_raw_bert_indices, aspect_bert_indices]

        outputs = self.model(inputs)
        t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
        print('t_probs = ', t_probs)
        print('Entity sentiment = ', t_probs.argmax(axis=-1) - 1)