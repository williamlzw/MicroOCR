import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MicroMLPNet
from average_meter import AverageMeter
from metric import RecMetric
from loss import CTCLoss
from dataset import TextLineDataset
from collatefn import RecCollateFn
from label_converter import CTCLabelConverter
from logger import create_logger


if not os.path.exists('save_model'):
    os.makedirs('save_model')
if not os.path.exists('log'):
    os.makedirs('log')
logger = create_logger('log')


def test_model(model, device, data_loader, converter, metric, loss_func, show_str_size):
    model.eval()
    with torch.no_grad():
        running_char_corrects, running_word_corrects, running_all_word, running_all_char = 0, 0, 0, 0
        show_strs = []
        since = time.time()
        for batch_idx, batch_data in enumerate(data_loader):
            batch_data['targets'], batch_data['targets_lengths'] = converter.encode(
                batch_data['labels'])
            batch_data['images'] = batch_data['images'].to(device)
            batch_data['targets'] = batch_data['targets'].to(device)
            batch_data['targets_lengths'] = batch_data['targets_lengths'].to(
                device)
            predicted = model.forward(
                batch_data['images'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            acc_dict = metric(predicted, batch_data['labels'])
            show_strs.extend(acc_dict['show_str'])
            running_char_corrects += acc_dict['char_correct']
            running_word_corrects += acc_dict['word_correct']
            running_all_char += torch.sum(batch_data['targets_lengths']).item()
            running_all_word += len(batch_data['images'])
            if (batch_idx+1) == len(data_loader):
                logger.info('Eval:[step {}/{} ({:.0f}%)] Loss:{:.4f} Word Acc:{:.4f} '
                            'Char Acc:{:.4f} Cost time:{:5.0f}s'.format(
                                running_all_word,
                                len(data_loader.dataset),
                                100. * (batch_idx+1) / len(data_loader),
                                loss_dict['loss'].item(),
                                running_word_corrects / running_all_word,
                                running_char_corrects / running_all_char,
                                time.time()-since))
    for s in show_strs[:show_str_size]:
        logger.info(s)
    model.train()
    val_word_accu = running_word_corrects / \
        running_all_word if running_all_word != 0 else 0.
    val_char_accu = running_char_corrects / \
        running_all_char if running_all_char != 0 else 0.
    return val_word_accu, val_char_accu


def train_model(cfg):
    device = torch.device("cuda:{}".format(cfg.gpu_index)
                          if torch.cuda.is_available() else "cpu")
    with open(cfg.vocabulary_path, mode='r', encoding='utf-8') as fa:
        lines = fa.readlines()
        character = [line.strip() for line in lines]
    train_loader = build_dataloader(
        cfg.train_root, cfg.train_list, cfg.batch_size, cfg.workers, character, cfg.in_channels, is_train=True, aug=True)
    test_loader = build_dataloader(
        cfg.test_root, cfg.test_list, cfg.batch_size, cfg.workers, character, cfg.in_channels, is_train=True)
    converter = build_conveter(character)
    loss_func = build_loss().to(device)
    loss_average = build_average_meter()
    metric = build_metric(converter)
    model = build_model(
        cfg.in_channels, cfg.nh, cfg.depth, converter.num_of_classes).to(device)
    if cfg.model_path != '':
        load_model(cfg.model_path, model)
    optimizer = build_optimizer(model, cfg.lr)
    scheduler = build_scheduler(optimizer)
    val_word_accu, val_char_accu, best_word_accu = 0., 0., 0.
    for epoch in range(cfg.epochs):
        model.train()
        running_char_corrects, running_word_corrects, running_all_word, running_all_char = 0, 0, 0, 0
        since = time.time()
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data['targets'], batch_data['targets_lengths'] = converter.encode(
                batch_data['labels'])
            batch_data['images'] = batch_data['images'].to(device)
            batch_data['targets'] = batch_data['targets'].to(device)
            batch_data['targets_lengths'] = batch_data['targets_lengths'].to(
                device)
            predicted = model.forward(
                batch_data['images'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            loss_average.update(loss_dict['loss'].item())
            acc_dict = metric(predicted, batch_data['labels'])
            running_char_corrects += acc_dict['char_correct']
            running_word_corrects += acc_dict['word_correct']
            running_all_char += torch.sum(batch_data['targets_lengths']).item()
            running_all_word += len(batch_data['images'])
            cost_time = time.time()-since
            if (batch_idx+1) % cfg.display_step_interval == 0 or (batch_idx+1) == len(train_loader):
                logger.info('Train:[epoch {}/{}][step {}/{} ({:.0f}%)] lr:{:.5f} Loss:{:.4f} Word Acc:{:.4f} '
                            'Char Acc:{:.4f} Cost time:{:5.0f}s Estimated time:{:5.0f}s'.format(
                                epoch+1,
                                cfg.epochs,
                                running_all_word//len(batch_data['images']),
                                len(train_loader.dataset)//len(
                                    batch_data['images']),
                                100. * (batch_idx+1) / len(train_loader),
                                scheduler.get_last_lr()[0],
                                loss_average.avg,
                                running_word_corrects / running_all_word,
                                running_char_corrects / running_all_char,
                                cost_time,
                                cost_time*len(train_loader) / (batch_idx+1) - cost_time))
            if (batch_idx+1) % cfg.eval_step_interval == 0:
                val_word_accu, val_char_accu = test_model(
                    model, device, test_loader, converter, metric, loss_func, cfg.show_str_size)
                if val_word_accu > best_word_accu:
                    best_word_accu = val_word_accu
                    save_model(cfg.model_type, model, cfg.nh, cfg.depth, 'best',
                                   best_word_accu, val_char_accu)
        if (epoch+1) % cfg.save_epoch_interval == 0:
            val_word_accu, val_char_accu = test_model(
                model, device, test_loader, converter, metric, loss_func, cfg.show_str_size)
            if val_word_accu > best_word_accu:
                best_word_accu = val_word_accu
                save_epoch = 'best'
            else:
                save_epoch = epoch+1
            save_model(cfg.model_type, model, cfg.nh, cfg.depth, save_epoch,
                           val_word_accu, val_char_accu)
        loss_average.reset()
        scheduler.step()


def build_conveter(character):
    return CTCLabelConverter(character)


def build_average_meter():
    return AverageMeter()


def build_metric(converter):
    return RecMetric(converter)


def build_loss(blank_idx=0, reduction='sum'):
    return CTCLoss(blank_idx, reduction)


def build_optimizer(model, lr=0.0001):
    # return optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
    return optim.Adam(model.parameters(), lr, betas=(0.5, 0.999), weight_decay=0.0001)


def build_dataset(data_dir, label_file_list, character, in_channels, augmentation):
    return TextLineDataset(data_dir, label_file_list, character, in_channels, augmentation)


def build_collate_fn():
    return RecCollateFn(32)


def build_dataloader(data_dir, label_file_list, batch_size,
                         num_workers, character, in_channels, is_train=False, aug=False):
    dataset = build_dataset(
        data_dir, label_file_list, character, in_channels, aug)
    collate_fn = build_collate_fn()
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        collate_fn=collate_fn, shuffle=is_train,
                        num_workers=num_workers)
    return loader


def save_model(model_type, model, nh, depth, epoch, word_acc, char_acc):
    if epoch == 'best':
        save_path = './save_model/{}_nh{}_depth{}_best_rec.pth'.format(
            model_type, nh, depth)
        if os.path.exists(save_path):
            data = torch.load(save_path)
            if 'model' in data and data['wordAcc'] > word_acc:
                return
    else:
        save_path = './save_model/{}_nh{}_depth{}_epoch{}_wordAcc{:05f}_charAcc{:05f}.pth'.format(
            model_type, nh, depth, epoch, word_acc, char_acc)
    torch.save({
        'model': model.state_dict(),
        'nh': nh,
        'depth': depth,
        'wordAcc': word_acc,
        'charAcc': char_acc},
        save_path)
    logger.info('save model to:'+save_path)


def load_model(model_path, model):
    data = torch.load(model_path)
    if 'model' in data:
        model.load_state_dict(data['model'])
        logger.info('Model loaded nh {}, depth {}, wordAcc {} , charAcc {}'.format(
            data['nh'], data['depth'], data['wordAcc'], data['charAcc']))


def build_scheduler(optimizer):
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10], gamma=0.1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    return scheduler


def build_model(in_channels, nh, depth, nclass):
    model = MicroMLPNet(in_channels=in_channels, nh=nh, depth=depth, nclass=nclass, img_height=32)
    return model


def main():
    parser = argparse.ArgumentParser(description='MicroOCR')
    parser.add_argument('--train_root', default='D:/dataset/gen/',
                        help='path to train dataset dir')
    parser.add_argument('--test_root', default='D:/dataset/gen/',
                        help='path to test dataset dir')
    parser.add_argument(
        '--train_list', default='D:/dataset/gen/train.txt', help='path to train dataset label file')
    parser.add_argument(
        '--test_list', default='D:/dataset/gen/test.txt', help='path to test dataset label file')
    parser.add_argument('--vocabulary_path', default='english.txt',
                        help='vocabulary path')
    parser.add_argument('--model_path', default='',
                        help='model path')
    parser.add_argument('--model_type', default='micromlp',
                        help='model type', type=str)
    parser.add_argument(
        '--nh', default=256, help='feature width, the more complex the picture background, the greater this value', type=int)
    parser.add_argument(
        '--depth', default=2, help='depth, the greater the number of samples, the greater this value', type=int)
    parser.add_argument(
        '--in_channels', default=3, help='in channels', type=int)
    parser.add_argument('--lr', default=0.001,
                        help='initial learning rate', type=float)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--workers', default=0,
                        help='number of data loading workers', type=int)
    parser.add_argument('--epochs', default=20,
                        help='number of total epochs', type=int)
    parser.add_argument('--display_step_interval', default=50,
                        help='display step interval', type=int)
    parser.add_argument('--eval_step_interval', default=500,
                        help='eval step interval', type=int)
    parser.add_argument('--save_epoch_interval', default=1,
                        help='save checkpoint epoch interval', type=int)
    parser.add_argument('--show_str_size', default=10,
                        help='show str size', type=int)
    parser.add_argument('--gpu_index', default=0, type=int,
                        help='gpu index')
    cfg = parser.parse_args()

    train_model(cfg)


if __name__ == '__main__':
    main()
