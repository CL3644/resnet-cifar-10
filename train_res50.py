# load data

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import pandas as pd

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0")

criterion = torch.nn.CrossEntropyLoss()
net = torchvision.models.resnet50()
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

losses = []
time_used = []
acc = []

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

top1 = AverageMeter()

for epoch in range(350):  # loop over the dataset multiple times
    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        start = time.time()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        end = time.time()
        # print statistics
        running_loss += loss.item()
        prec1 = accuracy(outputs.float().data, labels)[0]
        top1.update(prec1.item(), inputs.size(0))

        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f, acc: %.3f, avg-acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100, prec1 / 100, top1.avg))
            running_loss = 0.0

        losses += [loss.item()]
        time_used += [end - start]
        acc += [prec1]

data = {'train loss':losses,
        'time to train':time_used,
        'train_acc':acc}
# Convert the dictionary into DataFrame
df = pd.DataFrame(data)
df.to_csv('./resnet18_v100.csv')
plt.plot(losses)
print('Finished Training')
