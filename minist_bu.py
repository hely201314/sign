import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import prettytable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#获取数据集
trans = [transforms.ToTensor()]
trans.insert(0, transforms.Resize(224))
trans = transforms.Compose(trans)
batch_size = 256

training_data = datasets.MNIST(
    root="./minist",
    train=True,
    download=True,
    transform=trans
)

test_data = datasets.MNIST(
    root="./minist",
    train=False,
    download=True,
    transform=trans
)

train_iter = data.DataLoader(training_data, batch_size, shuffle=True,
                        num_workers=2)
test_iter = data.DataLoader(test_data, batch_size, shuffle=False,
                        num_workers=2)

train_features, train_labels = next(iter(train_iter))


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(9216, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, 10)

    def forward(self, x):
        out_conv1 = F.relu(self.conv1(x))
        out_pool1 = self.maxpool1(out_conv1)
        out_conv2 = F.relu(self.conv2(out_pool1))
        out_pool2 = self.maxpool2(out_conv2)
        out_conv3 = F.relu(self.conv3(out_pool2))
        out_conv4 = F.relu(self.conv4(out_conv3))
        out_conv5 = F.relu(self.conv5(out_conv4))
        out_pool3 = self.maxpool3(out_conv5)

        flatten_x = self.flatten(out_pool3)
        out_linear1 = F.relu(self.linear1(flatten_x))
        out_dropout1 = self.dropout1(out_linear1)
        out_linear2 = F.relu(self.linear2(out_dropout1))
        out_dropout2 = F.relu(out_linear2)
        out_linear3 = F.relu(self.linear3(out_dropout2))
        return out_linear3

model=AlexNet().to(device)
learning_rate = 0.001
batch_size = 256
# 自行设定optimizer和loss_fn
###
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
###

model.to(device)
loss_list = []
acc_list = []
epoch_num = []


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# 定义训练循环和测试循环
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    for t in range(epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        running_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            running_loss += loss
            # Backpropagation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 49:
                writer.add_scalar('training loss',
                                  running_loss / 50,
                                  epoch * len(dataloader) + batch + 1)

                loss, current = loss.item(), (batch + 1) * len(X)
                loss_list.append(loss), epoch_num.append(t + current / size)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                running_loss = 0

        test_loop(test_iter, model, loss_fn)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    acc_list.append(correct)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

#开始训练(epoch=30)
model.apply(init_weights)
writer = SummaryWriter()
train_loop(train_iter, model, loss_fn, optimizer, 10)

torch.save(model.state_dict(), 'MnistOnAlexNet_epoch30.pkl')

model.eval()
model.to('cpu')
pred_list = torch.tensor([])
with torch.no_grad():
    for X, y in test_iter:
        pred = model(X)
        pred_list = torch.cat([pred_list, pred])

test_iter1 = data.DataLoader(test_data, batch_size=10000, shuffle=False,
                        num_workers=2)
features, labels = next(iter(test_iter1))

#输出每个类别的精确率和召回率
train_result = np.zeros((10, 10), dtype=int)
for i in range(len(test_data)):
    train_result[labels[i]][np.argmax(pred_list[i])] += 1
result_table = prettytable.PrettyTable()
result_table.field_names = ['Type', 'Accuracy(精确率)', 'Recall(召回率)', 'F1_Score']
class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
for i in range(10):
    accuracy = train_result[i][i] / train_result.sum(axis=0)[i]
    recall = train_result[i][i] / train_result.sum(axis=1)[i]
    result_table.add_row([class_names[i], np.round(accuracy, 3), np.round(recall, 3),
                          np.round(accuracy * recall * 2 / (accuracy + recall), 3)])
print(result_table)