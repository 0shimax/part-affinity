import torch
try:
    from .coco import CocoDataSet
except:
    pass
from data_process.test import TestDataSet


def create_data_loaders(opt, fname=None):
    tr_dataset, te_dataset = create_data_sets(opt, fname)
    train_loader = torch.utils.data.DataLoader(
        te_dataset if tr_dataset is None else tr_dataset,
        batch_size=opt.batchSize,
        shuffle=True if opt.DEBUG == 0 else False,
        drop_last=True,
        num_workers=opt.nThreads
    )
    test_loader = torch.utils.data.DataLoader(
        te_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.nThreads
    )
    return train_loader, test_loader


def create_data_sets(opt, fnames=None):
    if opt.dataset == 'coco':
        tr_dataset = CocoDataSet(opt.data, opt, 'train')
        te_dataset = CocoDataSet(opt.data, opt, 'val')
    elif opt.dataset=='mine':
        tr_dataset = None
        te_dataset = TestDataSet(opt.data, fnames, opt)
    else:
        raise ValueError('Data set ' + opt.dataset + ' not available.')
    return tr_dataset, te_dataset
