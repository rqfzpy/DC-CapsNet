from __future__ import print_function

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import lr_scheduler
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class LeNet(nn.Module):
    def __init__(self, input_channels=1, n_classes=10, img_size=28):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Calculate the number of input features for the fully connected layer
        if img_size == 28:
            fc_input = 128 * 5 * 5
        elif img_size == 32:
            fc_input = 128 * 6 * 6
        else:
            raise ValueError("Unsupported img_size: {}".format(img_size))
        self.fc1 = nn.Linear(fc_input, 1280)
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = F.softmax(x, dim=1)
        return x, probs
    

class CrossEntropyLossWrapper(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWrapper, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets, size_average=True):
        return self.loss_fn(logits, targets)

if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    parser = argparse.ArgumentParser(description='LeNet with various datasets')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='SVHN', choices=['MNIST', 'F-MNIST', 'Kuzushiji-MNIST', 'SVHN', 'CIFAR10', 'SMALLNORB', 'AFFNIST', 'IDRID', 'ISIC'],
                        help='dataset to use')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Dataset selection and transform
    if args.dataset == 'MNIST':
        input_channels = 1
        n_classes = 10
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Pad(2), transforms.RandomCrop(28),
                                           transforms.ToTensor()
                                       ]))
        test_dataset = datasets.MNIST('../data', train=False, download=True,
                                      transform=transforms.ToTensor())
    elif args.dataset == 'F-MNIST':
        input_channels = 1
        n_classes = 10
        train_dataset = datasets.FashionMNIST('../data', train=True, download=True,
                                              transform=transforms.Compose([
                                                  transforms.Pad(2), transforms.RandomCrop(28),
                                                  transforms.ToTensor()
                                              ]))
        test_dataset = datasets.FashionMNIST('../data', train=False, download=True,
                                             transform=transforms.ToTensor())
    elif args.dataset == 'Kuzushiji-MNIST':
        input_channels = 1
        n_classes = 10
        train_dataset = datasets.KMNIST('../data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Pad(2), transforms.RandomCrop(28),
                                            transforms.ToTensor()
                                        ]))
        test_dataset = datasets.KMNIST('../data', train=False, download=True,
                                       transform=transforms.ToTensor())
    elif args.dataset == 'SVHN':
        input_channels = 3
        n_classes = 10
        train_dataset = datasets.SVHN('/mnt/data/dataset', split='train', download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize(32),
                                          transforms.ToTensor()
                                      ]))
        test_dataset = datasets.SVHN('/mnt/data/dataset', split='test', download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor()
                                     ]))
    elif args.dataset == 'CIFAR10':
        input_channels = 3
        n_classes = 10
        train_dataset = datasets.CIFAR10('/mnt/data/dataset', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        test_dataset = datasets.CIFAR10('/mnt/data/dataset', train=False, download=True,
                                        transform=transforms.ToTensor())
    elif args.dataset == 'IDRID':
        input_channels = 3
        n_classes = 3
        train_dataset = datasets.ImageFolder('/mnt/data2/meddata/IDRID/data_cop/Images/train', transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]))
        test_dataset = datasets.ImageFolder('/mnt/data2/meddata/IDRID/data_cop/Images/test', transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]))
    elif args.dataset == 'ISIC':
        input_channels = 3
        n_classes = 3
        train_dataset = datasets.ImageFolder('/mnt/data2/meddata/ISIC/train', transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]))
        test_dataset = datasets.ImageFolder('/mnt/data2/meddata/ISIC/test', transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]))
    elif args.dataset == 'SMALLNORB':
        input_channels = 1
        n_classes = 5
        from smallnorb import SmallNORBDataset
        train_dataset = SmallNORBDataset('../data/smallnorb', train=True)
        test_dataset = SmallNORBDataset('../data/smallnorb', train=False)
    elif args.dataset == 'AFFNIST':
        input_channels = 1
        n_classes = 10
        from affnist import AffNISTDataset
        train_dataset = AffNISTDataset('../data/affnist', train=True)
        test_dataset = AffNISTDataset('../data/affnist', train=False)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    if args.dataset in ['MNIST', 'F-MNIST', 'Kuzushiji-MNIST', 'SMALLNORB', 'AFFNIST']:
        img_size = 28
    elif args.dataset in ['SVHN', 'CIFAR10', 'IDRID', 'ISIC']:
        img_size = 32

    model = LeNet(input_channels=input_channels, n_classes=n_classes, img_size=img_size)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
    loss_fn = CrossEntropyLossWrapper()

    log_dir = 'lenetres/{}/logs'.format(args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train_log.txt')

    params = sum(p.numel() for p in model.parameters())
    with open(log_path, 'w') as f:
        f.write(f"Total Params: {params}\n")
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,epoch_time\n")

    def train(epoch):
        model.train()
        correct = 0
        total = 0
        train_loss = 0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            logits, probs = model(data)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            pred = probs.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += data.size(0)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        avg_loss = train_loss / total
        acc = correct / total
        epoch_time = time.time() - start_time
        return avg_loss, acc, epoch_time

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                logits, probs = model(data)
                test_loss += loss_fn(logits, target).item() * data.size(0)
                pred = probs.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total += data.size(0)
        avg_loss = test_loss / total
        acc = correct / total
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, total, 100. * acc))
        return avg_loss, acc

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, epoch_time = train(epoch)
        val_loss, val_acc = test()
        scheduler.step(val_loss)
        if epoch % 20 == 0:
            torch.save(model.state_dict(),
                    'lenetres/{}/{:03d}_model_dict.pth'.format(
                        args.dataset, epoch))
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f},{epoch_time:.2f}\n")