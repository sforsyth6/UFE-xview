import torch
import time
import datasets
from lib.utils import AverageMeter, get_factors
import torchvision.transforms as transforms
import numpy as np
import tabulate
import tqdm
import pandas as pd

def NN(net, lemniscate, trainloader, testloader, recompute_memory=0):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    end = time.time()
    print_freq=10
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()
            
            cls_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0:
                print('Test [{}/{}]\t'
                      'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                      'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                      'Top1: {:.2f}'.format(
                       batch_idx, len(testloader), correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def kNN(net, lemniscate, trainloader, testloader, K, sigma): #, recompute_memory=0): # Changed to recompute_memory in main
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

    # Changed to recompute_memory in main
    # if recompute_memory:
    #     transform_bak = trainloader.dataset.transform
    #     trainloader.dataset.transform = testloader.dataset.transform
    #     temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
    #     for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
    #         targets = targets.cuda(async=True)
    #         batchSize = inputs.size(0)
    #         features = net(inputs)
    #         trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
    #     #trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
    #     trainloader.dataset.transform = transform_bak
    
    top1 = np.zeros(C)
    top5 = np.zeros(C)
    total = np.zeros(C)
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(tqdm.tqdm(testloader)):
            end = time.time()
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            for i in targets.unique():
                idx = torch.nonzero(targets == i)
                top1[i] += torch.sum(correct[idx, 0]).cpu().numpy()
                top5[i] += torch.sum(correct[idx, :5]).cpu().numpy()
                total[i] += len(idx)
            
            cls_time.update(time.time() - end)
            # print('Test [{}/{}]\t'
            #      'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
            #      'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
            #      'Top1: {:.2f}  Top5: {:.2f}'.format(
            #      total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))

    # print(top1*100./total)

    table = pd.DataFrame({'classes': list(trainloader.dataset.class_to_idx.keys()),
                         'top1': np.round((top1*100./total), 2),
                         'top5': np.round((top5 * 100. / total), 2)})

    print(tabulate.tabulate(table.values, table.columns, tablefmt='pipe'))
    print('Top1 Avg (total samples): ', np.round(sum(top1) / sum(total) * 100., 1), '\nTop5 Avg (total samples): ',
          np.round(sum(top5) / sum(total) * 100., 1))
    print('Top1 Avg (by class): ', np.round(sum(top1 / total * 100.) / len(top1), 1), '\nTop5 Avg (by class): ',
          np.round(sum(top5 / total * 100.) / len(top5), 1))

    #return top1/total
    return sum(top5 / total * 100.) / len(top5)

