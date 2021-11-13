import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MicroNet
from metric import RecMetric
from loss import CTCLoss
from dataset import RecTextLineDataset
from collatefn import RecCollateFn
from label_converter import CTCLabelConverter
from logger import create_logger
from keys import character

if not os.path.exists('save_model'):
    os.makedirs('save_model')
if not os.path.exists('log'):
    os.makedirs('log')
logger = create_logger('log')


def test_model(model, device, data_loader, converter, metric, loss_func, show_str_size):
    model.eval()
    with torch.no_grad():
        running_loss, running_char_corrects, running_all, running_all_char_correct, running_word_corrects = 0., 0., 0., 0., 0.
        word_correct, char_correct = 0, 0
        batch_idx = 0
        show_strs = []
        for batch_idx, batch_data in enumerate(data_loader):
            targets, targets_lengths = converter.encode(batch_data['labels'])
            batch_data['targets'] = targets
            batch_data['targets_lengths'] = targets_lengths
            batch_data['images'] = batch_data['images'].to(device)
            batch_data['targets'] = batch_data['targets'].to(device)
            batch_data['targets_lengths'] = batch_data['targets_lengths'].to(
                device)
            predicted = model.forward(
                batch_data['images'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            running_loss += loss_dict['loss'].item()
            acc_dict = metric(predicted, batch_data['labels'])
            word_correct = acc_dict['word_correct']
            char_correct = acc_dict['char_correct']
            show_strs.extend(acc_dict['show_str'])
            running_char_corrects += char_correct
            running_word_corrects += word_correct
            running_all_char_correct += torch.sum(targets_lengths).item()
            running_all += len(batch_data['images'])
            if batch_idx == 0:
                since = time.time()
            elif batch_idx == len(data_loader)-1:
                logger.info('Eval:[{:5.0f}/{:5.0f} ({:.0f}%)] Loss:{:.4f} Word Acc:{:.4f} Char Acc:{:.4f} Cost time:{:5.0f}s'.format(
                    running_all,
                    len(data_loader.dataset),
                    100. * batch_idx / (len(data_loader)-1),
                    running_loss / running_all,
                    running_word_corrects / running_all,
                    running_char_corrects / running_all_char_correct,
                    time.time()-since))
    for s in show_strs[:show_str_size]:
        logger.info(s)
    model.train()
    val_word_accu = running_word_corrects / \
        running_all if running_all != 0 else 0.
    val_char_accu = running_char_corrects / \
        running_all_char_correct if running_all_char_correct != 0 else 0.
    return val_word_accu, val_char_accu


def train_model(cfg):
    device = torch.device("cuda:{}".format(cfg.gpu_index)
                          if torch.cuda.is_available() else "cpu")
    train_loader = build_rec_dataloader(
        cfg.train_root, cfg.train_list, cfg.batch_size, cfg.workers, character, is_train=True)
    test_loader = build_rec_dataloader(
        cfg.test_root, cfg.test_list, cfg.batch_size, cfg.workers, character, is_train=False)
    converter = CTCLabelConverter(character)
    loss_func = build_rec_loss().to(device)
    metric = build_rec_metric(converter)
    model = build_rec_model(
        cfg.nh, cfg.depth, converter.num_of_classes).to(device)
    if cfg.model_path != '':
        load_rec_model(cfg.model_path, model)
    optimizer = build_optimizer(model, cfg.lr)
    scheduler = build_scheduler(optimizer)
    val_word_accu, val_char_accu, best_word_accu = 0., 0., 0.
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, running_char_corrects, running_all, running_all_char_correct, running_word_corrects = 0., 0., 0., 0., 0.
        word_correct, char_correct = 0, 0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data['targets'], batch_data['targets_lengths'] = converter.encode(
                batch_data['labels'])
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(device)
            predicted = model.forward(
                batch_data['images'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            running_loss += loss_dict['loss'].item()
            acc_dict = metric(predicted, batch_data['labels'])
            word_correct = acc_dict['word_correct']
            char_correct = acc_dict['char_correct']
            running_char_corrects += char_correct
            running_word_corrects += word_correct
            running_all_char_correct += torch.sum(
                batch_data['targets_lengths']).item()
            running_all += len(batch_data['images'])

            if batch_idx == 0:
                since = time.time()
            elif batch_idx % cfg.display_interval == 0 or (batch_idx == len(train_loader)-1):
                logger.info('Train:[epoch {}/{} {:5.0f}/{:5.0f} ({:.0f}%)] Loss:{:.4f} Word Acc:{:.4f} Char Acc:{:.4f} Cost time:{:5.0f}s Estimated time:{:5.0f}s'.format(
                    epoch+1,
                    cfg.epochs,
                    running_all,
                    len(train_loader.dataset),
                    100. * batch_idx / (len(train_loader)-1),
                    running_loss / running_all,
                    running_word_corrects/running_all,
                    running_char_corrects / running_all_char_correct,
                    time.time()-since,
                    (time.time()-since)*(len(train_loader)-1) / batch_idx - (time.time()-since)))
            if batch_idx != 0 and batch_idx % cfg.val_interval == 0:
                val_word_accu, val_char_accu = test_model(
                    model, device, test_loader, converter, metric, loss_func, cfg.show_str_size)
                if val_word_accu > best_word_accu:
                    best_word_accu = val_word_accu
                    save_rec_model(cfg.model_type, model, cfg.nh, cfg.depth, 'best',
                                   best_word_accu, val_char_accu)
        if (epoch+1) % cfg.save_epoch == 0:
            val_word_accu, val_char_accu = test_model(
                model, device, test_loader, converter, metric, loss_func, cfg.show_str_size)
            save_rec_model(cfg.model_type, model, cfg.nh, cfg.depth, epoch+1,
                           val_word_accu, val_char_accu)
        scheduler.step()


def build_rec_model(nh, depth, nclass):
    model = MicroNet(nh=nh, depth=depth, nclass=nclass)
    return model


def build_rec_metric(converter: CTCLabelConverter):
    return RecMetric(converter)


def build_rec_loss(blank_idx=0, reduction='sum'):
    return CTCLoss(blank_idx, reduction)


def build_optimizer(model, lr=0.0001):
    return optim.Adam(model.parameters(), lr,
                      betas=(0.5, 0.999), weight_decay=0.001)


def build_rec_dataset(data_dir, label_file_list, character):
    return RecTextLineDataset(data_dir, label_file_list, character)


def build_rec_collate_fn():
    return RecCollateFn()


def build_rec_dataloader(data_dir, label_file_list, batchsize,
                         num_workers, character,  is_train=False):
    dataset = build_rec_dataset(
        data_dir, label_file_list, character)

    collate_fn = build_rec_collate_fn()
    if is_train:
        loader = DataLoader(dataset=dataset, batch_size=batchsize,
                            collate_fn=collate_fn, shuffle=True,
                            num_workers=num_workers)
    else:
        loader = DataLoader(dataset=dataset, batch_size=batchsize,
                            collate_fn=collate_fn, shuffle=False,
                            num_workers=num_workers)
    return loader


def build_scheduler(optimizer, step_size=1000, gamma=0.8):
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
    return scheduler


def save_rec_model(model_type, model, nh, depth, epoch, word_acc, char_acc):
    if epoch == 'best':
        save_path = './save_model/{}_nh{}_depth{}_best_rec.pth'.format(model_type, nh, depth)
        if os.path.exists(save_path):
            data = torch.load(save_path)
            if 'model' in data and data['wordAcc'] > word_acc:
                return
        torch.save({
            'model': model.state_dict(),
            'nh': nh,
            'depth': depth,
            'wordAcc': word_acc,
            'charAcc': char_acc},
            save_path)
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


def load_rec_model(model_path, model):
    data = torch.load(model_path)
    if 'model' in data:
        model.load_state_dict(data['model'])
        logger.info('Model loaded nh {}, depth {}, wordAcc {} , charAcc {}'.format(
            data['nh'], data['depth'], data['wordAcc'], data['charAcc']))


def main():
    parser = argparse.ArgumentParser(description='MicroOCR')
    parser.add_argument('--train_root', default='./',
                        help='path to train dataset dir')
    parser.add_argument('--test_root', default='./',
                        help='path to test dataset dir')
    parser.add_argument(
        '--train_list', default='./train.txt', help='path to train dataset label file')
    parser.add_argument(
        '--test_list', default='./test.txt', help='path to test dataset label file')
    parser.add_argument('--model_path', default='',
                        help='model path')
    parser.add_argument('--model_type', default='micro',
                        help='model type', type=str)
    parser.add_argument(
        '--nh', default=64, help='feature width, the more complex the picture background, the greater this value', type=int)
    parser.add_argument(
        '--depth', default=2, help='depth, the greater the number of samples, the greater this value', type=int)
    parser.add_argument('--lr', default=0.0001,
                        help='initial learning rate', type=float)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--workers', default=0,
                        help='number of data loading workers', type=int)
    parser.add_argument('--epochs', default=100,
                        help='number of total epochs', type=int)
    parser.add_argument('--display_interval', default=200,
                        help='display interval', type=int)
    parser.add_argument('--val_interval', default=1000,
                        help='val interval', type=int)
    parser.add_argument('--save_epoch', default=1,
                        help='how many epochs to save the weight', type=int)
    parser.add_argument('--show_str_size', default=10,
                        help='show str size', type=int)
    parser.add_argument('--gpu_index', default=0, type=int,
                        help='gpu index')
    cfg = parser.parse_args()
    train_model(cfg)


if __name__ == '__main__':
    main()
