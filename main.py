import argparse
import os
import shutil
import time
import tqdm
import datetime
from shutil import copyfile

from tsnecuda import TSNE
import matplotlib.pyplot as plt
import faiss

#from tensorflow.python.keras.preprocessing import image
#from sklearn.manifold import TSNE
#from lap import lapjv
#from scipy.spatial.distance import cdist

from scipy.cluster.hierarchy import linkage, dendrogram

#from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models

import datasets
import models

from lib.utils import get_factors
from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN

from apex.fp16_utils import FP16_Optimizer

#from create_tsne_plot import compute_tsne
#from find_outlier_images import compute_distribution_of_samples
#from find_outlier_images import compute_knn_distances

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--balanced_sampling',  default=False, action='store_true', help='by default set false, oversamples less populated classes')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--fine_tune', default='', type=str, metavar='PATH', help='Fine tune a pre-trained model using new dataset')
parser.add_argument('--iter_size', default=1, type=int, help='caffe style iter size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--K',  default=200, type=int, help='Default number of neighbors for KNN classification')
parser.add_argument('--low-dim', default=128, type=int, metavar='D', help='feature dimension')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nce-k', default=4096, type=int, metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-m', default=0.5, type=float, help='momentum for non-parametric updates')
parser.add_argument('--nce-t', default=0.07, type=float, metavar='T', help='temperature parameter for softmax')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--recompute-memory', default=False, action='store_true', help='recompute memory on train dataset for evaluation stage')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--static_loss', default=25, type=float, help='set static loss for apex optimizer')
#parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--tsne', default=None, help='run tsne and set perplexity of tsne. Can be int or list of ints: 1,2,3,4')
parser.add_argument('--pca', dest='pca', action='store_true', help='run pca')
parser.add_argument('--graph_labels', default=50, type=int, help='Set number of labels to use ontop of pca/tsne plot')
parser.add_argument('--view_knn', dest='view_knn', action='store_true', help='move KNN to a folder to be downloaded')
parser.add_argument('--kmeans', default=0, type=int, help='run kmeans')
parser.add_argument('--tsne_grid', dest='tsne_grid', action='store_true', help='generate tsne grid of images')
parser.add_argument('--red_data', default=1, type=float, help='percentage of dataset to use from 0-1')
parser.add_argument('--color_jit', default=0.025, type=float, help='values for color jitter transform')
parser.add_argument('--h_cluster', action='store_true', help='run hierarichal clustering')
parser.add_argument('--select', action='store_true', help='run hierarichal clustering')
parser.add_argument('--mv_data', action='store_true', help='run hierarichal clustering')
parser.add_argument('--kmverbose', action='store_true', help='run hierarichal clustering')
parser.add_argument('--select_thresh', default = 0.95, type=float)
parser.add_argument('--select_size', default=0, type=float)
parser.add_argument('--select_save', action='store_true')
parser.add_argument('--select_load', action='store_true')
parser.add_argument('--select_num', default=50, type=int)
parser.add_argument('--kmeans_opt', action='store_true')

best_prec1 = 0

def main():


    global args, best_prec1
    args = parser.parse_args()

    # Initialize distributed processing
    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, low_dim=args.low_dim)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](low_dim=args.low_dim)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet stats
                                      std=[0.229, 0.224, 0.225])
#    normalize = transforms.Normalize(mean=[0.234, 0.191, 0.159],  # xView stats
#                                     std=[0.173, 0.143, 0.127])

    print("Creating datasets")
    cj = args.color_jit
    train_dataset = datasets.ImageFolderInstance(
        traindir,
        transforms.Compose([
            transforms.Resize((224, 224)),
#            transforms.Grayscale(3),
#            transforms.ColorJitter(cj, cj, cj, cj), #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
	    transforms.RandomRotation(45),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    elif args.balanced_sampling:

        print("Using balanced sampling")
        # Here's where we compute the weights for WeightedRandomSampler
        class_counts = {v: 0 for v in train_dataset.class_to_idx.values()}
        for path, ndx in train_dataset.samples:
            class_counts[ndx] += 1
        total = float(np.sum([v for v in class_counts.values()]))
        class_probs = [class_counts[ndx] / total for ndx in range(len(class_counts))]

        # make a list of class probabilities corresponding to the entries in train_dataset.samples
        reciprocal_weights = [class_probs[idx] for i, (_, idx) in enumerate(train_dataset.samples)]

        # weights are the reciprocal of the above
        weights = (1 / torch.Tensor(reciprocal_weights))

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    else:
        #if args.red_data is < 1, then the training is done with a subsamle of the total data. Otherwise it's the total data.
        data_size = len(train_dataset)
        sub_index = np.random.randint(0,data_size,round(args.red_data*data_size))
        sub_index.sort()
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sub_index)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    print("Training on", len(train_dataset.imgs), "images. Training batch size:", args.batch_size)

    if len(train_dataset.imgs) % args.batch_size != 0:
        print("Warning: batch size doesn't divide the # of training images so ",
              len(train_dataset.imgs) % args.batch_size, "images will be skipped per epoch.")
        print("If you don't want to skip images, use a batch size in:", get_factors(len(train_dataset.imgs)))

    val_dataset = datasets.ImageFolderInstance(
        valdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    val_bs = [factor for factor in get_factors(len(val_dataset)) if factor < 500][-1]
    val_bs = 100
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("Validating on", len(val_dataset),  "images. Validation batch size:", val_bs)

    # define lemniscate and loss function (criterion)
    ndata = train_dataset.__len__()
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
        criterion = NCECriterion(ndata).cuda()
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
#           best_prec1 = checkpoint['best_prec1']
            lemniscate = checkpoint['lemniscate']
            if args.select_load:
                pred = checkpoint['prediction']
            print("=> loaded checkpoint '{}' (epoch {}, best_prec1 )"
                  .format(args.resume, checkpoint['epoch'])) #, checkpoint['best_prec1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optionally fine-tune a model trained on a different dataset
    elif args.fine_tune:
        print("=> loading checkpoint '{}'".format(args.fine_tune))
        checkpoint = torch.load(args.fine_tune)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.fine_tune, checkpoint['epoch']))
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)

    # Optionally recompute memory. If fine-tuning, then we must recompute memory
    if args.recompute_memory or args.fine_tune:

        # Aaron - Experiments show that iterating over torch.utils.data.DataLoader will skip the last few
        # unless the batch size evenly divides size of the data set. This shouldn't be the case
        # according to documentation, there's even a flag for drop_last, but it's not working

        # compute a good batch size for re-computing memory
        memory_bs = [factor for factor in get_factors(len(train_loader.dataset)) if factor < 500][-1]
        print("Recomputing memory using", train_dataset.root, "with a batch size of", memory_bs)
        transform_bak = train_loader.dataset.transform
        train_loader.dataset.transform = val_loader.dataset.transform
        temploader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=memory_bs, shuffle=False,
            num_workers=train_loader.num_workers, pin_memory=True)
        lemniscate.memory = torch.zeros(len(train_loader.dataset), args.low_dim).cuda()
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, indexes) in enumerate(tqdm.tqdm(temploader)):
                batchSize = inputs.size(0)
                features = model(inputs)
                lemniscate.memory[batch_idx * batchSize:batch_idx * batchSize + batchSize, :] = features.data
        train_loader.dataset.transform = transform_bak
        model.train()
    
    cudnn.benchmark = True

    if args.evaluate:
        kNN(model, lemniscate, train_loader, val_loader, args.K, args.nce_t)
        return

    begin_train_time = datetime.datetime.now()

#    my_knn(model, lemniscate, train_loader, val_loader, args.K, args.nce_t, train_dataset, val_dataset)
    if args.tsne:
        labels = idx_to_name(train_dataset, args.graph_labels)
        tsne(lemniscate, args.tsne, labels)
    if args.pca:
        labels = idx_to_name(train_dataset, args.graph_labels)
        pca(lemniscate, labels)
    if args.view_knn:
        my_knn(model, lemniscate, train_loader, val_loader, args.K, args.nce_t, train_dataset, val_dataset)
    if args.kmeans:
        kmeans,yi = kmean(lemniscate, args.kmeans, 500, args.K, train_dataset)
        D, I = kmeans.index.search(lemniscate.memory.data.cpu().numpy(), 1)

        cent_group = {}
        data_cent = {}
        for n,i in enumerate(I):
            if i[0] not in cent_group.keys():
                cent_group[i[0]] = []
            cent_group[i[0]].append(n)
        data_cent[n] = i[0]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(cent_group[0])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        
#        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
#        criterion = NCECriterion(ndata).cuda()


#    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)

    if args.tsne_grid:
        tsne_grid(val_loader,model)
    if args.h_cluster:
        for size in range(2,3):
    #        size = 20
            kmeans,topk =  kmean(lemniscate, size, 500, 10, train_dataset)
            respred = torch.tensor([]).cuda()
            lab, idx = [[] for i in range(2)]
            num = 0
            '''
            for p,index,label in pred:
                respred = torch.cat((respred,p))
                if num == 0:
                    lab = label
                else:
                    lab += label
                idx.append(index)
                num+=1
            '''
            h_cluster(lemniscate, train_dataset, kmeans, topk, size) #, respred, lab, idx)

#    axis_explore(lemniscate, train_dataset)

#    kmeans_opt(lemniscate, 5)
    
    if args.select:
        if not args.select_load:
            pred = [] 

            if args.select_size:
                size = int(args.select_size*ndata)
            else:
                size = round(ndata/100.0)
            
            sub_sample = np.random.randint(0,ndata, size=size)
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sub_sample)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)

            pred = div_train(train_loader, model, 0, pred)

        pred_features = []
        pred_labels = []
        pred_idx = []

        for inst in pred:
            feat,idx,lab = list(inst)
            pred_features.append(feat)
            pred_labels.append(lab)
            pred_idx.append(idx.data.cpu())
        
        if args.select_save:

            save_checkpoint({
                'epoch': args.start_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'prediction':pred,
                'lemniscate': lemniscate,
                'optimizer' : optimizer.state_dict(),
            }, 'select.pth.tar')

        min_idx = selection(pred_features,pred_idx,train_dataset,args.select_num,args.select_thresh)

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(min_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        
        lemniscate = NCEAverage(args.low_dim, ndata, 20, args.nce_t, args.nce_m)

        optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.1, weight_decay=0.00001)

        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)

        for epoch in range(50):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch)


            if epoch % 1 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'lemniscate': lemniscate,
                    'optimizer' : optimizer.state_dict(),
                })

            train(train_loader, model, lemniscate, criterion, optimizer,epoch)
 
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sub_index)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)


        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)

    if args.kmeans_opt:
        kmeans_opt(lemniscate,500)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        if epoch % 1 == 0:
            # evaluate on validation set
            #prec1 = NN(epoch, model, lemniscate, train_loader, train_loader) # was evaluating on train
#            prec1 = kNN(model, lemniscate, train_loader, val_loader, args.K, args.nce_t)
            # prec1 really should be renamed to prec5 as kNN now returns top5 score, but
            # it won't be backward's compatible as earlier models were saved with "best_prec1"

            # remember best prec@1 and save checkpoint
#            is_best = prec1 > best_prec1
#            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'lemniscate': lemniscate,
#                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            })# , is_best)

        # train for one epoch
        train(train_loader, model, lemniscate, criterion, optimizer, epoch)

#        kmeans,cent = kmeans()
#        group_train(train_loader, model, lemniscate, criterion, optimizer, epoch, kmeans, cent)

    # print elapsed time
    end_train_time = datetime.datetime.now()
    d = end_train_time - begin_train_time
    print("Trained for %d epochs. Elapsed time: %s days, %.2dh: %.2dm: %.2ds" %
          (len(range(args.start_epoch, args.epochs)),
           d.days, d.seconds // 3600, (d.seconds // 60) % 60, d.seconds % 60))

    # evaluate KNN after last epoch
#    kNN(model, lemniscate, train_loader, val_loader, args.K, args.nce_t)

def axis_explore(lemniscate, train_dataset):
    data = lemniscate.memory.data.cpu().numpy()
    min_max = []
    dataT = data.T

    for i in range(len(dataT)):
        l = np.sort(dataT[i,:])
        h = list(range(0,len(data), int(round(len(data)/50.0))))
        h.append(len(data)-1)
        min_max.append(l[h])
    
    dim_exam = []

    for num, y in enumerate(min_max):
        dim = []
        for j in y:
            dim.append(np.abs(dataT[num]-j).argmin())
        dim_exam.append(dim)

    path = '/data/'
    for num, d in enumerate(dim_exam): 
        os.mkdir(path + 'dims/{}'.format(num))
        for n,x in enumerate(d):
            img = train_dataset.__getitem__(x)[3]
            copyfile(path + 'train/all/' + img, path + 'dims/{0}/{1}'.format(num,n))

def kmeans_opt(lemniscate, epoch):
    x = lemniscate.memory
    x = x.data.cpu().numpy()
    d = x.shape[1]


#    ncentroids = Variable(torch.randint(1,1000, size=(1,1)).type(torch.FloatTensor) , requires_grad=True)
#    niter = Variable(torch.randint(1,1000,size=(1,1)).type(torch.FloatTensor), requires_grad=True)

#    optimizer = torch.optim.Adam([ncentroids,niter], lr=0.01)
    low_loss = 0

    for i in range(2,epoch): 
        if i % 10 == 0:
            print (i)
        kmeans = faiss.Kmeans(d, i, 500, spherical=True)
        kmeans.train(x)

        loss = kmeans_loss(lemniscate,kmeans)
        if loss < low_loss:
            low_loss = loss
            print (i,loss)


'''
    for e in range(epoch):
        optimizer.zero_grad()


        kmeans = faiss.Kmeans(d, int(ncentroids.data.numpy()[0][0]), int(niter.data.numpy()[0][0]), spherical=True)
        kmeans.train(x)

        loss = kmeans_loss(lemniscate,kmeans)
        loss.backward()

        optimizer.step()

        print (loss)
'''


def kmeans_loss(lemniscate, kmeans):
    data = lemniscate.memory.data.cpu().numpy()
    cents = kmeans.centroids
    D, I = kmeans.index.search(data, 1)


    cent_group = {}
    data_cent = {}
    for n,i in enumerate(I):
        if i[0] not in cent_group.keys():
            cent_group[i[0]] = []
        cent_group[i[0]].append(n)
        data_cent[n] = i[0]

    inner_groupsim = []
    cos = nn.CosineSimilarity(dim=0)
    for c in cent_group.keys():
        centroid = torch.tensor(cents[c])
        dis = []
        for idx in cent_group[c]:
            x = torch.tensor(data[idx])
            out = cos(x,centroid)
            dis.append(out)
        dis = torch.tensor(dis)
        inner_groupsim.append(dis.mean())
    inner_groupsim = torch.tensor(inner_groupsim)


    outer_groupsim = []
    data_idxs = np.array(list(data_cent.keys()))
    r = np.random.randint(0,len(data_idxs), size=500)
    dat = data_idxs[r]
    for idx in dat:
        id_cent = data_cent[idx]
        sim_cent = list(cent_group.keys())
        sim_cent.remove(id_cent)
        x = torch.tensor(data[idx])
        dis = []
        for s in sim_cent:
            centroid = torch.tensor(cents[s])
            out = cos(x,centroid)
            dis.append(out)
        dis = torch.tensor(dis)
        outer_groupsim.append(torch.max(dis))
    outer_groupsim = torch.tensor(outer_groupsim)
    
    loss = -inner_groupsim.mean() + outer_groupsim.mean()
    return loss


def selection(pred,pred_idx,train_dataset,num_select,thresh):
    sub_data = pred
    cos = nn.CosineSimilarity(dim=0)
#    rand = np.random.randint(0,len(data), size=round(len(data)/100.0))
#    sub_data = data[rand,:]
    min_idx = []
    for ran in range(num_select):
        r = np.random.randint(0,len(sub_data))
        cos_list = []
        indx = []
        for idx,x in enumerate(sub_data):
            cos_list.append(cos(sub_data[r],x))
            indx.append([pred_idx[r],pred_idx[idx]])
        minn = min(cos_list, key=lambda x:abs(x-thresh))
#        minn = min(cos_list)
        min_idx.append(indx[cos_list.index(minn)])

        print (minn, train_dataset.__getitem__(indx[cos_list.index(minn)][0])[3], train_dataset.__getitem__(indx[cos_list.index(minn)][1])[3])
    
    final_min = []
    for x in min_idx:
        for y in x:
            final_min.append(y)

    return final_min

'''
    path = '/data/images_rgb/disc/'

    for num,inst in enumerate(min_idx):
        os.mkdir(path + 'seperate/{}'.format(num))
        for x in inst:
            print (x)
            img = train_dataset.__getitem__(x)[3]
            copyfile(path + 'train/all/' + img, path + 'seperate/{}/'.format(num) + img)
'''


def h_cluster(lemniscate,train_dataset, kmeans, topk, size):
    x = lemniscate.memory.data.cpu().numpy()
    d = x.shape[1]
#    size = round(len(data)/1000.0)
#    size = 1000
#    rand = np.random.randint(0,len(data), size=size)
#    data = data[rand,:].data.cpu().numpy()
#    lab = np.array(lab)
#    lab = lab[rand]
    data = kmeans.centroids
    hc = linkage(data,metric='cos',method='complete')
    plt.figure()
    den = dendrogram(hc)
    print (den['ivl'])
    plt.savefig('dend-{}.png'.format(size), dbi=600)

    for n,i in enumerate(topk):
        label = train_dataset.__getitem__(i[0])[3]
        print ('{0},{1}'.format(n,label))

def tsne_grid(val_loader, model):
        # Generate t-sne-based matrix of images
        features = []
        images = []
        for i, (input, _, index, names) in enumerate(val_loader):
            index = index.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            index_var = torch.autograd.Variable(index)
            
            # compute output
            feature = model(input_var)
            feature = feature.cpu()
            
            for i in range(feature.data.numpy().shape[0]):
                images.append(input.numpy()[i,...])
                features.append(feature.data.numpy()[i,:])
            
            print(len(features))
        
        print(np.array(images).shape)
        print(np.array(features).shape)
        img_collection = np.moveaxis(np.array(images),1,-1)
        print(img_collection.shape)
        size = 45
        perplexity = 20
        tsne_iter = 5000
        print("Running tsne...")
        tsne = TSNE(perplexity=perplexity, n_components=2, init='random', n_iter=tsne_iter)
        X_2d = tsne.fit_transform(np.array(features)[0:size*size,:])
        print("tsne complete.  Normalizing...")
        X_2d -= X_2d.min(axis=0)
        X_2d /= X_2d.max(axis=0)
        print("Normalization complete.  Creating plot...")
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))).reshape(-1, 2)
        cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        _, row_asses, col_asses = lapjv(cost_matrix)
        grid_jv = grid[col_asses]
        out = np.ones((size*224, size*224, 3))
        
        for pos, img in zip(grid_jv, img_collection[0:size*size]):
            h_range = int(np.floor(pos[0]* (size - 1) * 224))
            w_range = int(np.floor(pos[1]* (size - 1) * 224))
            out[h_range:h_range + 224, w_range:w_range + 224]  = image.img_to_array(img)
            
        print("plot complete.  Saving gridded plot...")
        im = image.array_to_img(out)
        im.save('UFL_TSNE_GRID.jpeg', quality=100)
        print("Gridded plot saved!")
        out = np.zeros((size*224, size*224, 3))
        for pos, img in zip(X_2d, img_collection[0:size*size]):
            h_range = int(pos[0]*(size-1)*224)
            w_range = int(pos[1]*(size-1)*224)
            out[h_range:h_range + 224, w_range:w_range + 224]  = image.img_to_array(img)
        
        print("plot complete.  Saving cloud plot...")
        im = image.array_to_img(out)
        im.save('UFL_TSNE_CLOUD.jpeg', quality=100)
        print("Cloud plot saved!")



def my_knn(net, lemniscate, trainloader, testloader, K, sigma, train_dataset, val_dataset): #, recompute_memory=0): # Changed to recompute_memory in main
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()

    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
    C = int(trainLabels.max() + 1)

    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        i = 0
        for batch_idx, (inputs, targets, indexes, path) in enumerate(tqdm.tqdm(testloader)):
            end = time.time()

            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
    
            path = '/data/images_rgb/disc/'

            for num,inst in enumerate(yi):
                os.mkdir(path + 'view/{}'.format(i))
                for x in yi[num]:
                    img = train_dataset.__getitem__(x)[3]
                    copyfile(path + 'train/all/' + img, path + 'view/{}/'.format(i) + img)
                i += 1
   

def kmean(lemniscate, ncentroids, niter, K, train_dataset):
    x = lemniscate.memory
    x = x.data.cpu().numpy()
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter, spherical=True)
    kmeans.train(x)
    cent = kmeans.centroids

    index = faiss.IndexFlatL2(d)
    index.add(x)
    D, I = index.search (kmeans.centroids, K)

    if args.kmverbose:
        #give cosine similarity between each kmeans centroid
        distances = []
        for c in cent:
            d = []
            for k in cent:
                dist = np.dot(c,k)
                cos = dist / (np.linalg.norm(c)*np.linalg.norm(k))
                d.append(cos)
            distances.append(d)

        abs_dis = [list(map(abs, inst)) for inst in distances ]
        print (np.mean(abs_dis), np.max(abs_dis), np.min(abs_dis))

        for i in range(20):
            print (i)
            for num,d in enumerate(distances[i]):
                if d >= 0.2:
                    print (d,num)

#get cosine similarity between the first KNN of each centroid with the rest of the KNN
    trainFeatures = lemniscate.memory.t()
    dist = torch.mm(torch.tensor(cent).cuda() , trainFeatures)
    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)

    if args.kmverbose:
        distances = []
        for num, centroid in enumerate(yi):
            if num >= 20:
                break
            d = []
            for n1, h in enumerate(centroid):
                for n1,j in enumerate(centroid):
                    xx = x[h]
                    yy = x[j]
                    dist = np.dot(xx,yy)
                    cos = dist / (np.linalg.norm(xx)*np.linalg.norm(yy))
                    d.append(cos)
                distances.append(d)
                break
        for i,d in enumerate(distances):
            print (i,d)

    #Gives the cluster and distance from each image if the cosine distance of th eimage and centroid is > 0.2
        distances = []
        for num, centroid in enumerate(yi):
            if num == 20:
                break
            dis = []
            for n1, h in enumerate(centroid):
                d = []
                for n,c in enumerate(cent):
                    xx = x[h]
                    dist = np.dot(xx,c)
                    cos = dist / (np.linalg.norm(xx)*np.linalg.norm(c))
                    if cos >= 0.2:
                        d.append([n,cos])
                dis.append(d)
            distances.append(dis)
        for i,d in enumerate(distances):
            print ('Cat {}'.format(i))
            for num,im in enumerate(d):
                print('image {}'.format(num))
                print (im)

#averages of knn to each centroid comparred to each image with cosine
#    av_knn_dist = []
#    for num, centroid in enumerate(yi):
#        img = x[centroid[:4].data.cpu()]
#        av_knn_dist.append(np.mean(img, axis=0))
#    distances = []
#    for num, centroid in enumerate(yi):
#        if num >= 10:
#            break
#        dis = []
#        for n1, h in enumerate(centroid):
#            xx = x[h]
#            d = []
#            for n,a in enumerate(av_knn_dist):
#                dist = np.dot(xx,a)
#                cos = dist / (np.linalg.norm(xx)*np.linalg.norm(a))
#                if cos >= .2:
#                    d.append([n,cos])
#            dis.append(d)
#        distances.append(dis)

#    for i,d in enumerate(distances):
#        print ('Cat {}'.format(i))
#        for num,im in enumerate(d):
#            print('image {}'.format(num))
#            print (im)

#    return kmeans, cent

    if args.mv_data:
        path = '/data/'
        for num,centroid in enumerate(yi):
            os.mkdir(path + 'kmeans/{}'.format(num))
            for n,x in enumerate(centroid):
                img = train_dataset.__getitem__(x)[3]
                copyfile(path + 'train/all/' + img, path + 'kmeans/{0}/{1}'.format(num,n))
    
    return kmeans,yi


def idx_to_name(train_dataset, size):
    name = {}
    from numpy.random import randint
    for i in range(size):
        r = randint(177000)
        name[r] = train_dataset.__getitem__(r)[3]
    return name

def pca(lemniscate, labels):
    x = lemniscate.memory
    niter=2
    k=2
    x = x.data.cpu()
    U,S,V = torch.svd(torch.t(x))
    C = torch.mm(x, U[:,:k])

    x, y = [[] for i in range(2)]
    for c in C.numpy():
        x.append(c[0])
        y.append(c[1])
    plt.scatter(x,y)
    for idx in labels.keys():
        label = labels[idx]
        lx = x[idx]
        ly = y[idx]
        plt.annotate(label, xy = (lx, ly), xytext = (-20, 20), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), textcoords = 'offset points')
    plt.savefig('pca.png')

def tsne(lemniscate, perp, labels):

    if isinstance(perp,int):
        perp = [perp]
    else:
        perp = perp.split(',')

    for p in perp:
        x = lemniscate.memory
        x = x.data.cpu()
        x_embedded = TSNE(n_components=2, perplexity=p, learning_rate=10).fit_transform(x)

        x,y = [[] for i in range(2)]
        for c in x_embedded:
            x.append(c[0])
            y.append(c[1])
        plt.scatter(x,y)

        for idx in labels.keys():
            label = labels[idx]
            lx = x[idx]
            ly = y[idx]
            plt.annotate(label, xy = (lx, ly), xytext = (-20, 20), arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), textcoords = 'offset points')
        plt.savefig('tsne-{}.png'.format(p))
        plt.clf()

def div_train(train_loader, model, epoch, pred):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
#    model.train()

    end = time.time()
#    optimizer.zero_grad()
    for i, (input, target, index, label) in enumerate(train_loader):
        #remove completely black images from training time

        # measure data loading time
        data_time.update(time.time() - end)

        index = index.cuda(async=True)                                                                          
        # compute output
        feature = model(input)
        for x in list(zip(feature.data,index,label)):
            pred.append(x)
#        output = lemniscate(feature, index)
#        loss = criterion(output, index) / args.iter_size
#       loss = 0.0
        #Backprop Apex optimizer loss
#        optimizer.backward(loss)

        # measure accuracy and record loss
#        losses.update(loss.item() * args.iter_size, input.size(0))

#        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
#            optimizer.step()
#            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
			epoch, i, len(train_loader), batch_time=batch_time,
			data_time=data_time))
    return pred


def train(train_loader, model, lemniscate, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (input, target, index, label) in enumerate(train_loader):
        #remove completely black images from training time

        # measure data loading time
        data_time.update(time.time() - end)

        index = index.cuda(async=True)

        # compute output
        feature = model(input)
#        y = torch.t(feature)
#        for num in range(len(y)):
#            var = torch.var(y[num])
#            if var > (1./(num+1.)):
#                y[num,:] = torch.zeros((1,len(feature)))
#        features = torch.t(y)

        output = lemniscate(feature, index)
        loss = criterion(output, index) / args.iter_size
        #Backprop Apex optimizer loss
        optimizer.backward(loss)

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input.size(0))

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss:.4f}\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss.item()))


#def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
#    if is_best:
#        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Decays the learning rate according to the default schedule, or more aggressively if fine-tuning"""

    if args.fine_tune:
        lr = args.lr * (0.5 ** (epoch // 2))
    else:
        lr = args.lr
        if epoch < 120:
            lr = args.lr
        elif epoch >= 120 and epoch < 160:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        #lr = args.lr * (0.1 ** (epoch // 100))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
