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
from keys import character


def test_model(model, device, data_loader, converter, metric, loss_func, show_str_size):
    model.eval()
    with torch.no_grad():
        running_loss, running_char_corrects, running_all, running_all_char_correct, running_word_corrects = 0., 0., 0., 0., 0.
        word_correct, char_correct = 0, 0
        batch_idx = 0
        show_strs = []
        for batch_idx, batch_data in enumerate(data_loader):
            targets, targets_lengths = converter.encode(batch_data['label'])
            batch_data['targets'] = targets
            batch_data['targets_lengths'] = targets_lengths
            batch_data['image'] = batch_data['image'].to(device)
            batch_data['targets'] = batch_data['targets'].to(device)
            batch_data['targets_lengths'] = batch_data['targets_lengths'].to(
                device)
            predicted = model.forward(
                batch_data['image'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            running_loss += loss_dict['loss'].item()
            acc_dict = metric(predicted, batch_data['label'])
            word_correct = acc_dict['word_correct']
            char_correct = acc_dict['char_correct']
            show_strs.extend(acc_dict['show_str'])
            running_char_corrects += char_correct
            running_word_corrects += word_correct
            running_all_char_correct += torch.sum(targets_lengths).item()
            running_all += len(batch_data['image'])
            if batch_idx == 0:
                since = time.time()
            elif batch_idx == len(data_loader)-1:
                print('Eval:[{:5.0f}/{:5.0f} ({:.0f}%)] Loss:{:.4f} Word Acc:{:.4f} Char Acc:{:.4f} Cost time:{:5.0f}s'.format(
                    running_all,
                    len(data_loader.dataset),
                    100. * batch_idx / (len(data_loader)-1),
                    running_loss / running_all,
                    running_word_corrects / running_all,
                    running_char_corrects / running_all_char_correct,
                    time.time()-since))
    for s in show_strs[:show_str_size]:
        print(s)
    model.train()
    val_word_accu = running_word_corrects / \
        running_all if running_all != 0 else 0.
    val_char_accu = running_char_corrects / \
        running_all_char_correct if running_all_char_correct != 0 else 0.
    return val_word_accu, val_char_accu


def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_rec_dataloader(
        cfg.train_root, cfg.train_list, cfg.batch_size, cfg.workers, character, is_train=True)
    test_loader = build_rec_dataloader(
        cfg.test_root, cfg.test_list, cfg.batch_size, cfg.workers, character, is_train=False)
    converter = CTCLabelConverter(character)
    loss_func = build_rec_loss().to(device)
    metric = build_rec_metric(converter)
    model = build_rec_model(cfg, converter.num_of_classes).to(device)
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
                batch_data['label'])
            for key, value in batch_data.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(device)
            predicted = model.forward(
                batch_data['image'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            running_loss += loss_dict['loss'].item()
            acc_dict = metric(predicted, batch_data['label'])
            word_correct = acc_dict['word_correct']
            char_correct = acc_dict['char_correct']
            running_char_corrects += char_correct
            running_word_corrects += word_correct
            running_all_char_correct += torch.sum(
                batch_data['targets_lengths']).item()
            running_all += len(batch_data['image'])

            if batch_idx == 0:
                since = time.time()
            elif batch_idx % cfg.display_interval == 0 or (batch_idx == len(train_loader)-1):
                print('Train:[epoch {}/{} {:5.0f}/{:5.0f} ({:.0f}%)] Loss:{:.4f} Word Acc:{:.4f} Char Acc:{:.4f} Cost time:{:5.0f}s Estimated time:{:5.0f}s'.format(
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
                    save_rec_model(cfg.model_type, model, 'best',
                                   best_word_accu, val_char_accu)
        if (epoch+1) % cfg.save_epoch == 0:
            val_word_accu, val_char_accu = test_model(
                model, device, test_loader, converter, metric, loss_func, cfg.show_str_size)
            save_rec_model(cfg.model_type, model, epoch+1,
                           val_word_accu, val_char_accu)
        scheduler.step()


def build_rec_model(cfg, nclass):
    model = MicroNet(nh=cfg.nh, depth=cfg.depth, nclass=nclass, use_lstm=cfg.use_lstm)
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


def save_rec_model(model_type, model,  epoch, word_acc, char_acc):
    if epoch == 'best':
        save_path = './save_model/{}_best_rec.pth'.format(model_type)
        if os.path.exists(save_path):
            data = torch.load(save_path)
            if 'model' in data and data['word_acc'] > word_acc:
                return
        torch.save({
            'model': model.state_dict(),
            'word_acc': word_acc,
            'char_acc': char_acc},
            save_path)
    else:
        save_path = './save_model/{}_epoch{}_word_acc{:05f}_char_acc{:05f}.pth'.format(
            model_type, epoch, word_acc, char_acc)
        torch.save({
            'model': model.state_dict(),
            'word_acc': word_acc,
            'char_acc': char_acc},
            save_path)
    print('save model to:'+save_path)


def load_rec_model(model_path, model):
    data = torch.load(model_path)
    if 'model' in data:
        model.load_state_dict(data['model'])
        print('Model loaded word_acc {} ,char_acc {}'.format(
            data['word_acc'], data['char_acc']))


def main():
    parser = argparse.ArgumentParser(description='MicroOCR')
    parser.add_argument('--train_root', default='F:/precode/',
                        help='path to train dataset dir')
    parser.add_argument('--test_root', default='F:/precode/',
                        help='path to test dataset dir')
    parser.add_argument(
        '--train_list', default='F:/precode/train.txt', help='path to train dataset label file')
    parser.add_argument(
        '--test_list', default='F:/precode/test.txt', help='path to test dataset label file')
    parser.add_argument('--model_path', default='',
                        help='model path')
    parser.add_argument('--model_type', default='micro',
                        help='model type', type=str)
    parser.add_argument('--nh', default=64, help='nh', type=int)
    parser.add_argument('--depth', default=2, help='depth', type=int)
    parser.add_argument('--use_lstm', default=True, help='use lstm', type=bool)
    parser.add_argument('--lr', default=0.0001,
                        help='initial learning rate', type=float)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--workers', default=0,
                        help='number of data loading workers', type=int)
    parser.add_argument('--epochs', default=300,
                        help='number of total epochs', type=int)
    parser.add_argument('--display_interval', default=200,
                        help='display interval', type=int)
    parser.add_argument('--val_interval', default=1000,
                        help='val interval', type=int)
    parser.add_argument('--save_epoch', default=1,
                        help='save epoch', type=int)
    parser.add_argument('--show_str_size', default=10,
                        help='show str size', type=int)
    cfg = parser.parse_args()
    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    train_model(cfg)


if __name__ == '__main__':
    main()
