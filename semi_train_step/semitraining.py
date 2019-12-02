from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os 
import numpy as np
import sys
sys.path.append('.')
import pickle
import torch
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import TenCrop, Lambda, Resize
from reid import datasets
import models
#from models import mcc
from model import mgn
#from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss,FocalLoss, CenterLoss
from reid.trainers import Trainer, FinedTrainer, FinedTrainer2, JointTrainer2
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from sklearn.cluster import DBSCAN,AffinityPropagation
from reid.rerank import *
from reid.eug import *
# from reid.rerank_plain import *

from option import args
import reid.utils.utility as utility
import multiprocessing
multiprocessing.set_start_method('spawn',True)

ckpt = utility.checkpoint(args)

def get_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval # use all training image to train
    #num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        Resize((height,width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader, test_loader

def get_source_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval # use all training image to train
    #num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        Resize((height,width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, 

    ## get_source_data
    """ src_dataset, src_extfeat_loader = \
        get_source_data(args.src_dataset, args.data_dir, args.height,
                        args.width, args.batch_size, args.workers) """
    # get_target_data
    tgt_dataset, tgt_extfeat_loader, test_loader = \
        get_data(args.tgt_dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(2048) -> FC(args.features)
    num_class = 0
    model = mgn.MGN(args)
    checkpoint = torch.load(args.load_dir)
    for x in list(checkpoint.keys()):
        if 'fc' in x:
            checkpoint.pop(x, None)
    model.load_state_dict(checkpoint, strict=False)
    
    # Load from checkpoint
    start_epoch = best_top1 = 0
    """ if args.resume:
        print('Resuming checkpoints from finetuned model on another dataset...\n')
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise RuntimeWarning('Not using a pre-trained model') """
    model = nn.DataParallel(model).cuda()
   
    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, print_freq=args.print_freq)
    print("Test with the original model trained on source domain:")
    best_top1 = evaluator.evaluate_same_cams(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    if args.evaluate:
        return 

    # Criterion
    criterion = []
    criterion.append(TripletLoss(margin=args.margin,num_instances=args.num_instances).cuda())
    criterion.append(TripletLoss(margin=args.margin,num_instances=args.num_instances).cuda())
    criterion.append(nn.CrossEntropyLoss().cuda())
    criterion.append(CenterLoss(num_classes=args.num_classes, feat_dim=256, use_gpu=True))
    #multi lr
    param_groups = [{'params': model.module.parameters(), 'lr_mult': 1.0}]
    # Optimizer
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)    

    ##### adjust lr
    def adjust_lr(epoch):
        if epoch <= 7:
            lr = args.lr
        elif epoch <=14:
            lr = 0.3 * args.lr
        else:
            lr = 0.1 * args.lr
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    ##### training stage transformer on input images
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        Resize((args.height,args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])

    # Start training
    iter_nums = args.iteration
    start_epoch = args.start_epoch
    cluster_list = []
    top_percent = args.rho
    EF = 100 // iter_nums + 1
    eug = None
    for iter_n in range(start_epoch, iter_nums):
        
        #### generate new dataset
        if iter_n == 0:
            u_data, l_data = updata_lable(tgt_dataset, args.tgt_dataset, sample=args.sample)
            eug = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=num_class, 
            data_dir=args.data_dir, l_data=l_data, u_data=u_data, print_freq=args.print_freq, 
            save_path=args.logs_dir, pretrained_model=model, rerank=True)
            eug.model = model

        nums_to_select = int(min((iter_n + 1) * int(len(u_data) // (iter_nums)), len(u_data)))
        #nums_to_select = 500
        pred_y, pred_score = eug.estimate_label()
        
        print('This is running {} with EF= {}%, step {}:\t Nums_to_be_select {}, \t Logs-dir {}'.format(
            args.mode, EF, iter_n+1, nums_to_select, args.logs_dir
        ))
        selected_idx = eug.select_top_data(pred_score, nums_to_select)
        new_train_data = eug.generate_new_train_data(selected_idx, pred_y)
        eug_dataloader = eug.get_dataloader(new_train_data, training=True)

        top1 = iter_trainer(model, tgt_dataset, test_loader, eug_dataloader, optimizer, 
            criterion, args.epochs, args.logs_dir, args.print_freq, args.lr)
        eug.model = model
        #del train_loader
        # del eug_dataloader


        is_best = top1 >= best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': iter_n + 1,
            'best_top1': best_top1,
            # 'num_ids': num_ids,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.2%}  best: {:5.2%}{}\n'.
              format(iter_n+1, top1, best_top1, ' *' if is_best else ''))

def iter_trainer(model, dataset, test_loader, eug_dataloader, optimizer, 
    criterion, epochs, logs_dir, print_freq, lr):
    # Trainer
    best_top1 = 0
    # trainer = Trainer(model, criterion)
    trainer = JointTrainer2(model, criterion)
    evaluator = Evaluator(model, print_freq=print_freq)
    # Start training
    for epoch in range(0, epochs):
        adjust_lr(lr, epoch, optimizer)
        trainer.train(epoch, eug_dataloader, optimizer)
    #evaluate
    top1 = evaluator.evaluate_same_cams(test_loader, dataset.query, dataset.gallery)

    return top1


def adjust_lr(init_lr, epoch, optimizer, step_size=55):
    lr = init_lr / (10 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

    if epoch % step_size == 0:
        print("Epoch {}, current lr {}".format(epoch, lr))


def compute_dist(source_features, target_features, lambda_value, no_rerank, num_split=2):
    euclidean_dist_list = []
    rerank_dist_list = []
    if isinstance(source_features, list): 
        for (s, t) in zip(source_features, target_features):
            _, rerank_dist = re_ranking(
                s.numpy(), t.numpy(), 
                lambda_value=lambda_value, 
                no_rerank=no_rerank
            )
            rerank_dist_list.append(rerank_dist)
            euclidean_dist_list.append([])
            del rerank_dist
    else:
        _, rerank_dist = re_ranking(
            source_features.numpy(), 
            target_features.numpy(),
            lambda_value=lambda_value, no_rerank=no_rerank
        )
        rerank_dist_list.append(rerank_dist)
        euclidean_dist_list.append([])
        del rerank_dist
    return euclidean_dist_list, rerank_dist_list


def generate_selflabel(e_dist, r_dist, n_iter, args, cluster_list=[]):
    labels_list = []
    for s in range(len(r_dist)):
        if n_iter==args.start_epoch:
            if args.no_rerank:
                tmp_dist = e_dist[s]
            else:
                tmp_dist = r_dist[s]
            ####DBSCAN cluster
            tri_mat = np.triu(tmp_dist,1)       # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            top_num = np.round(args.rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps,min_samples=4, metric='precomputed', n_jobs=8)
            cluster_list.append(cluster)
        else:
            cluster = cluster_list[s]
        #### select & cluster images as training set of this epochs
        print('Clustering and labeling...', end='/r', flush=True)
        if args.no_rerank:
            #euclidean_dist = -1.0 * euclidean_dist #for similarity matrix
            labels = cluster.fit_predict(e_dist[s])
        else:
            #rerank_dist = -1.0 * rerank_dist  #for similarity matrix
            labels = cluster.fit_predict(r_dist[s])
        num_ids = len(set(labels)) - 1  ##for DBSCAN cluster
        #num_ids = len(set(labels)) ##for affinity_propagation cluster
        print('Iteration {} have {} training ids'.format(n_iter+1, num_ids))
        print('\r')
        labels_list.append(labels)
        del labels
        del cluster
    return labels_list, cluster_list

def generate_dataloader(tgt_dataset, train_transformer, iter_n, args):
    new_dataset = []
    for i, (fname, pid, _) in enumerate(tgt_dataset.trainval):
        if pid==0:
            continue 
        new_dataset.append((fname, pid, 0))
    print('Iteration {} have {} training images'.format(iter_n+1, len(new_dataset)))
    train_loader = DataLoader(
        Preprocessor(new_dataset, root=tgt_dataset.images_dir,
                        transform=train_transformer),
        batch_size=args.batch_size, num_workers=4,
        sampler=RandomIdentitySampler(new_dataset, args.num_instances),
        pin_memory=True, drop_last=True
    )
    return train_loader

if __name__ == '__main__':
    
    main(args)
