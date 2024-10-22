import torch
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv

def set_seed(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enable =True


def test(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)

    with torch.no_grad():
        for data, label in loader:
            data = data.squeeze()
            data, label = data.float().cuda(), label.squeeze().long().cuda()

            logits = model(data)  # [b]
            # pred = logits >= 0.5
            pred = logits.argmax(dim=1)

            correct += torch.sum(pred == label)

    acc = 100. * correct / total

    return acc


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def acc_plt(train_acc_list, val_acc_list, opt):
    plt.figure()
    plt.title("Accuracy")
    plt.ylim((50,100))
    plt.plot(train_acc_list, label='train_acc')
    plt.plot(val_acc_list, label='val_acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(opt.outf+'/acc_plt_'+opt.name+'_seed'+opt.seed+'.png')


def show_confusion_matrix(model, loader, opt):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    cm = np.zeros([10, 10])
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, label in loader:
            batch_length = data.shape[0]
            data = data.squeeze()
            data, label = data.float().cuda(), label.squeeze().long().cuda()

            logits = model(data)  # [b]
            # pred = logits >= 0.5
            pred = logits.argmax(dim=1)
            correct += torch.sum(pred == label)

            for i in range(batch_length):
                true_index = int(label[i])
                pred_index = int(pred[i].int())
                cm[true_index][pred_index] += 1
                all_labels.append(true_index)
                all_predictions.append(pred_index)

    # Plot the confusion matrix using seaborn and matplotlib
    plt.figure()
    labels = ['0','1','2','3','4','5','6','7','8','9']  # Replace with your actual class labels
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(opt.outf+'/Confusion Matrix_'+opt.name+'_seed'+opt.seed+'.png')


def test_submit(model, opt):
    test_pack = np.load(opt.data_root+'test.npz')
    data = test_pack['arr_0']
    data = torch.from_numpy(data).cuda()
    img_id = test_pack['arr_1']

    model.load_state_dict(torch.load(opt.outf+'/'+opt.name+'_seed'+opt.seed+'_best.mdl'))
    print('--best model loaded')

    with open(opt.outf+'/'+opt.name+'_seed'+opt.seed+'_sampleSubmission.csv', mode='w', newline='', encoding='utf8') as csv_file:
        wf = csv.writer(csv_file)
        wf.writerow(['id', 'label'])
        
        model.eval()
        
        for i in range(data.shape[0]):
            logits = model(data[i].unsqueeze(0))  # [b]
            pred = logits >= 0.5
            pred = pred.int().item()
            wf.writerow([img_id[i], pred])

    #     wf.writerow([])
        csv_file.flush()
        csv_file.close()

    print('Finish')

