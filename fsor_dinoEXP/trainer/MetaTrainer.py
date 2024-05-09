from __future__ import print_function

import os
import numpy as np
import argparse
import socket
import time
import sys
from tqdm import tqdm
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from architectures.NetworkPre import FeatureNet
from trainer.FSEval import run_test_fsl
from util import adjust_learning_rate, accuracy, AverageMeter
from sklearn import metrics


class MetaTrainer(object):
    def __init__(self, args, dataset_trainer, eval_loader=None, hard_path=None):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        args.logroot = os.path.join(args.logroot, args.task + '_' + args.dataset)
        if not os.path.isdir(args.logroot):
            os.makedirs(args.logroot)

        try:
            iterations = args.lr_decay_epochs.split(',')
            args.lr_decay_epochs = list([])
            for it in iterations:
                args.lr_decay_epochs.append(int(it))
        except:
            pass
        
        args.model_name = '{}_{}_shot_{}'.format(args.dataset, args.n_train_runs, args.n_shots)

        
        self.save_path = os.path.join(args.logroot, args.model_name)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
        assert args.pretrained_model_path is not None, 'Missing Pretrained Model'
        params = torch.load(args.pretrained_model_path)['params']
        feat_params = {k: v for k, v in params.items() if 'feature' in k}
        cls_params = {k: v for k, v in params.items() if 'cls_classifier' in k}

        oslo_paras = (args.inference_steps, args.lambda_s, args.lambda_z, args.ema_weight, args.use_inlier_latent)
        self.args = args
        self.train_loader, self.val_loader, n_cls = dataset_trainer
        self.model = FeatureNet(args, args.restype, n_cls, (cls_params,self.train_loader.dataset.vector_array), oslo_paras)

        if args.restype == 'ResNet12':
            model_dict = self.model.state_dict()
            model_dict.update(feat_params)
            self.model.load_state_dict(model_dict)

        self.model.train()

        # optimizer
        if self.args.tunefeat == 0.0:
            optim_param = [{'params': self.model.cls_classifier.parameters()}]
        else:
            optim_param = [{'params': self.model.cls_classifier.parameters()},{'params': filter(lambda p: p.requires_grad, self.model.feature.parameters()),'lr': self.args.tunefeat}]

        self.optimizer = optim.SGD(optim_param, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            cudnn.benchmark = True

        # set cosine annealing scheduler
        if args.cosine:
            print("==> training with plateau scheduler ...")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max')
        else:
            print("==> training with MultiStep scheduler ... gamma {} step {}".format(args.lr_decay_rate, args.lr_decay_epochs))
    

    def train(self, eval_loader=None):

        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['maxmeta_acc'] = 0.0
        trlog['maxmeta_acc_epoch'] = 0
        trlog['maxmeta_auroc'] = 0.0
        trlog['maxmeta_auroc_epoch'] = 0
        
        writer = SummaryWriter(self.save_path)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()


        for epoch in range(1, self.args.epochs + 1):
            if self.args.cosine:
                self.scheduler.step(trlog['maxmeta_acc'])
            else:
                adjust_learning_rate(epoch, self.args, self.optimizer, 0.0001)
            
            train_acc, train_auroc, train_loss, train_msg = self.train_episode(epoch, self.train_loader, self.model, criterion, self.optimizer, self.args)

            writer.add_scalar('train/acc', float(train_acc), epoch)
            writer.add_scalar('train/auroc', float(train_auroc), epoch)
            writer.add_scalar('train/loss_ce', float(train_loss[0]), epoch)
            writer.add_scalar('train/loss_op', float(train_loss[1]), epoch)
            writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            self.model.eval()

            #evaluate
            if eval_loader is not None:
                start = time.time()
                assert self.args.task == 'FSOR'
                config = {'auroc_type':['prob']}
                result = run_test_fsl(self.model, eval_loader, config)
                meta_test_acc = result['data']['acc']
                open_score_auroc = result['data']['auroc_prob']

                test_time = time.time() - start
                writer.add_scalar('meta/close_acc', float(meta_test_acc[0]), epoch)
                writer.add_scalar('meta/close_std', float(meta_test_acc[1]), epoch)
                writer.add_scalar('meta/open_auroc', float(open_score_auroc[0]), epoch)
                writer.add_scalar('meta/open_std', float(open_score_auroc[1]), epoch)
                
                meta_msg = 'Meta Test Acc: {:.4f}, Test std: {:.4f}, AUROC: {:.4f}, Time: {:.1f}'.format(meta_test_acc[0], meta_test_acc[1], open_score_auroc[0], test_time)
                train_msg = train_msg + ' | ' + meta_msg
                
                if trlog['maxmeta_acc'] < meta_test_acc[0]:
                    trlog['maxmeta_acc'] = meta_test_acc[0]
                    trlog['maxmeta_acc_epoch'] = epoch
                    acc_auroc = (meta_test_acc[0], open_score_auroc[0])
                    self.save_model(epoch, 'max_acc', acc_auroc)
                if trlog['maxmeta_auroc'] < open_score_auroc[0]:
                    trlog['maxmeta_auroc'] = open_score_auroc[0]
                    trlog['maxmeta_auroc_epoch'] = epoch
                    acc_auroc = (meta_test_acc[0], open_score_auroc[0])
                    self.save_model(epoch, 'max_auroc', acc_auroc)
                
            print(train_msg)

            # regular saving
            if epoch % 5 == 0:
                self.save_model(epoch,'last')
                print('The Best Meta Acc {:.4f} in Epoch {}, Best Meta AUROC {:.4f} in Epoch {}'.format(trlog['maxmeta_acc'],trlog['maxmeta_acc_epoch'],trlog['maxmeta_auroc'],trlog['maxmeta_auroc_epoch']))


    def train_episode(self, epoch, train_loader, model, criterion, optimizer, args):
        """One epoch training"""
        model.train()
        if self.args.tunefeat==0:
            model.feature.eval()
        
        pp_loss_sche = 0# epoch // 10 + 1


        batch_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_op = AverageMeter()
        acc = AverageMeter()
        auroc = AverageMeter()
        end = time.time()

        with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
            for idx, data in enumerate(pbar):
                support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label, supp_idx, open_idx, base_ids = data

                # Data Conversion & Packaging
                support_data,support_label              = support_data.float().cuda(),support_label.cuda().long()
                query_data,query_label                  = query_data.float().cuda(),query_label.cuda().long()
                suppopen_data,suppopen_label            = suppopen_data.float().cuda(),suppopen_label.cuda().long()
                openset_data,openset_label              = openset_data.float().cuda(),openset_label.cuda().long()
                supp_idx, open_idx,base_ids = supp_idx.long(), open_idx.long(),base_ids.long()
                openset_label = self.args.n_ways * torch.ones_like(openset_label)
                the_img     = (support_data, query_data, suppopen_data, openset_data)
                the_label   = (support_label,query_label,suppopen_label,openset_label)
                the_conj    = (supp_idx, open_idx)

                _, _, probs, loss = model(the_img,the_label,the_conj,base_ids)
                query_cls_probs, openset_cls_probs = probs
                (loss_ce, loss_op) = loss
                loss_op = args.op_loss * loss_op

                loss = loss_op + loss_ce

                ### Closed Set Accuracy
                close_pred = np.argmax(probs[0][:,:,:self.args.n_ways].view(-1,self.args.n_ways).cpu().numpy(),-1)
                close_label = query_label.view(-1).cpu().numpy()
                acc.update(metrics.accuracy_score(close_label, close_pred),1)

                ### Open Set AUROC
                open_label_binary = np.concatenate((np.ones(close_pred.shape),np.zeros(close_pred.shape)))
                query_cls_probs = query_cls_probs.view(-1, self.args.n_ways+1)
                openset_cls_probs = openset_cls_probs.view(-1, self.args.n_ways+1)
                open_scores = torch.cat([query_cls_probs,openset_cls_probs], dim=0).cpu().numpy()[:,-1]
                auroc.update(metrics.roc_auc_score(open_label_binary,open_scores),1)
                
                
                losses_ce.update(loss_ce.item(), 1)
                losses_op.update(loss_op.item(), 1)

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # ===================meters=====================
                batch_time.update(time.time() - end)
                end = time.time()
                
                
                pbar.set_postfix({"Acc":'{0:.2f}'.format(acc.avg), 
                                "Auroc":'{0:.2f}'.format(auroc.avg), 
                                "ce_loss" :'{0:.2f}'.format(losses_ce.avg), 
                                "op_loss" :'{0:.2f}'.format(losses_op.avg), 
                                })

        message = 'Epoch {} Train_Acc {acc.avg:.3f} Train_Auroc {auroc.avg:.3f}'.format(epoch, acc=acc, auroc=auroc)

        return acc.avg, auroc.avg, (losses_ce.avg, losses_op.avg), message
    
               
    def save_model(self, epoch, name=None, acc_auroc=None):
        state = {
            'epoch': epoch,
            'cls_params': self.model.state_dict() if self.args.n_gpu==1 else self.model.module.state_dict(),
            'acc_auroc': acc_auroc
        }
        # 'optimizer': self.optimizer.state_dict()['param_groups'],
                 
        file_name = 'epoch_'+str(epoch)+'.pth' if name is None else name + '.pth'
        print('==> Saving', file_name)
        torch.save(state, os.path.join(self.save_path, file_name))
    
    
           