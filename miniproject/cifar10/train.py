import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
import argparse
from utils import *
from model import model_generator
from    torchvision import datasets
from    torchvision import transforms

parser = argparse.ArgumentParser(description="Cat&Dog Clf_Luo")
parser.add_argument('--method', type=str, default='ERROR')
parser.add_argument('--name', type=str, default='empty')
parser.add_argument("--batch_size", type=int, default=2000)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--outf", type=str, default='checkpoint/')
parser.add_argument("--data_root", type=str, default='dataset/')
parser.add_argument("--seed", type=str, default='0')
parser.add_argument("--data_aug", type=bool, default=False)
parser.add_argument("--gpus", type=str, default='0')
opt = parser.parse_args()

opt.outf = opt.outf + opt.name

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

logger = initialize_logger(opt.outf+'/'+opt.name+'_seed'+opt.seed+'.log')

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
print('GPU {} is used!'.format(opt.gpus))

set_seed(opt.seed)


train_dataset = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]), download=True)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

val_dataset = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]), download=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)


print("train_len = ", len(train_dataset))
print("val_len = ", len(val_dataset))


model = model_generator(opt.method).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
criterion = nn.CrossEntropyLoss()

best_acc = 0
train_acc_list = []
val_acc_list = []

for epoch in range(opt.epoch):
    for step, (data, label) in enumerate(train_loader):
        data = data.squeeze()
        data, label = data.float().cuda(), label.cuda()
        
        model.train()
        logits = model(data)

        loss = criterion(logits, label) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = test(model, train_loader)
    train_acc_list.append(train_acc.item())
    acc = test(model, val_loader)
    val_acc_list.append(acc.item())

    if acc > best_acc:
        best_epoch = epoch
        best_acc = acc
        torch.save(model.state_dict(), opt.outf+'/'+opt.name+'_seed'+opt.seed+'_best.mdl')

    info = "train epoch[{}/{}] current acc:{:3f}, best acc:{:.3f}".format(epoch + 1, opt.epoch, acc,max(val_acc_list))
    print(info) 
    logger.info(info)
    logger.info(f'Train accuracy: {train_acc.item()}, validation accuracy: {acc.item()}\n')

print('best test acc:', best_acc.item(), 'best epoch:', best_epoch + 1)
logger.info('best test acc:', best_acc.item(), 'best epoch:', best_epoch + 1)

model.load_state_dict(torch.load(opt.outf+'/'+opt.name+'_seed'+opt.seed+'_best.mdl'))
print('--best model loaded')

test_acc = test(model, val_loader)

print(f'\nFinal test\nTest accuracy = {test_acc}')
logger.info(f'\nFinal test\nTest accuracy = {test_acc}')

acc_plt(train_acc_list, val_acc_list, opt)
show_confusion_matrix(model, val_loader, opt)
# test_submit(model, opt)



