# This code is modified from https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py

import os
import time
import datetime
import logging
import os.path as osp
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.miscellaneous import load_configs, collect_env_info, mkdir, save_checkpoint
from utils.distributed import get_rank
from utils.logger import setup_logger
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import accuracy
from utils.lr_scheduler import adjust_learning_rate
import models.wideresnet as wideresnet

CONFIG_FILE = 'basic.yml'

best_acc1 = 0

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    global best_acc1
    
    run_time = str(datetime.datetime.now())
    cfgs = load_configs(CONFIG_FILE)

    # create log root dir and weight root dir
    mkdir(cfgs['weight_dir'])
    mkdir(cfgs['log_dir'])
    
    # create logger
    log_dir = osp.join(cfgs['log_dir'], cfgs['arch'])
    mkdir(log_dir)

    cfgs['log_name'] = cfgs['arch'] + '_' + cfgs['dataset']
    logger = setup_logger(cfgs['log_name'], log_dir, get_rank(), run_time + '.txt')

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(CONFIG_FILE))
    logger.info("Running with config:\n{}".format(cfgs))
    
    #  create model
    logger.info("=> creating model '{}'".format(cfgs['arch']))
    model = models.__dict__[cfgs['arch']]()
    
    if cfgs['arch'].lower().startswith('wideresnet'):
        # a customized resnet model with last feature map size as 14x14 for better class activation mapping
        model  = wideresnet.resnet50(num_classes=cfgs['num_classes'])
    else:
        model = models.__dict__[cfgs['arch']](num_classes=cfgs['num_classes'])

    if cfgs['arch'].lower().startswith('alexnet') or cfgs['arch'].lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    logger.info("=> created model '{}'".format(model.__class__.__name__))
    logger.info("model structure: {}".format(model))
    num_gpus = torch.cuda.device_count()
    logger.info("using {} GPUs".format(num_gpus))

    # optionally resume from a checkpoint
    if cfgs['resume']:
        if os.path.isfile(cfgs['resume']):
            logger.info("=> loading checkpoint '{}'".format(cfgs['resume']))
            checkpoint = torch.load(cfgs['resume'])
            cfgs['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfgs['resume'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(cfgs['resume']))

    torch.backends.cudnn.benchmark = True

    # Data loading code
    traindir = osp.join(cfgs['data_path'], 'train')
    valdir = osp.join(cfgs['data_path'], 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfgs['batch_size'], shuffle=True,
        num_workers=cfgs['workers'], pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfgs['batch_size'], shuffle=False,
        num_workers=cfgs['workers'], pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), cfgs['lr'],
                                momentum=cfgs['momentum'],
                                weight_decay=float(cfgs['weight_decay']))

    if cfgs['evaluate']:
        validate(val_loader, model, criterion, cfgs)
        return

    for epoch in range(cfgs['start_epoch'], cfgs['epochs']):
        adjust_learning_rate(optimizer, epoch, cfgs)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, cfgs)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, cfgs)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': cfgs['arch'],
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, cfgs['weight_dir'] + '/' + cfgs['arch'].lower())


def train(train_loader, model, criterion, optimizer, epoch, cfgs):
    logger = logging.getLogger('{}.train'.format(cfgs['log_name']))
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfgs['print_freq'] == 0:
            logger.info(progress.display(i))


def validate(val_loader, model, criterion, cfgs):
    logger = logging.getLogger('{}.validate'.format(cfgs['log_name']))

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfgs['print_freq'] == 0:
                logger.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg



if __name__ == '__main__':
    main()