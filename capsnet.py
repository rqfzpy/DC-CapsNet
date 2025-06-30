from __future__ import print_function

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import lr_scheduler
from torch.autograd import Variable

try:
    from thop import profile
except ImportError:
    profile = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=10,input_channels=1, 
                 img_size=28, primary_kernel=9, primary_stride=2, conv1_kernel=9, conv1_stride=1):
        super(CapsNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        # self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        # self.num_primaryCaps = 32 * 6 * 6
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=conv1_kernel, stride=conv1_stride)
        # Calculate conv1 output size
        conv1_out = (img_size - conv1_kernel) // conv1_stride + 1
        # PrimaryCaps parameters
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=primary_kernel, stride=primary_stride)
        # Calculate PrimaryCaps output size
        primary_out = (conv1_out - primary_kernel) // primary_stride + 1
        self.num_primaryCaps = 32 * primary_out * primary_out
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
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
    parser.add_argument('--routing_iterations', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=False)
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
        # You need to implement SmallNORBDataset in smallnorb.py
        from smallnorb import SmallNORBDataset
        train_dataset = SmallNORBDataset('../data/smallnorb', train=True)
        test_dataset = SmallNORBDataset('../data/smallnorb', train=False)
    elif args.dataset == 'AFFNIST':
        input_channels = 1
        n_classes = 10
        # You need to implement AffNISTDataset in affnist.py
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
    elif args.dataset in ['SVHN', 'CIFAR10', 'IDRID', 'ISIC']: # Add IDRID and ISIC to 32x32 image list
        img_size = 32
    # Build model
    # model = CapsNet(args.routing_iterations, n_classes=n_classes)
    model = CapsNet(args.routing_iterations, n_classes=n_classes, input_channels=input_channels,img_size=img_size)

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(16, n_classes)
        reconstruction_alpha = 2
        model = CapsNetWithReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = MarginLoss(0.9, 0.1, 0.5)

    # ==== Logging ====
    log_dir = 'capsres/{}/logs'.format(args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train_log.txt')
    # Count parameters and FLOPs
    if profile is not None:
        # Set input shape according to dataset
        if args.dataset in ['MNIST', 'F-MNIST', 'Kuzushiji-MNIST', 'SMALLNORB', 'AFFNIST']:
            dummy_input = torch.randn(1, 1, 28, 28)
        elif args.dataset in ['SVHN', 'CIFAR10', 'IDRID', 'ISIC']:
            dummy_input = torch.randn(1, 3, 32, 32)
        else:
            # Default grayscale 28x28
            dummy_input = torch.randn(1, 1, 28, 28)
        if args.cuda:
            dummy_input = dummy_input.cuda()
        if args.with_reconstruction:
            dummy_target = torch.zeros(1, dtype=torch.long)
            if args.cuda:
                dummy_target = dummy_target.cuda()
            macs, params = profile(model, inputs=(dummy_input, dummy_target), verbose=False)
        else:
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    else:
        macs, params = 0, sum(p.numel() for p in model.parameters())
    with open(log_path, 'w') as f:
        f.write(f"Total Params: {params}\n")
        f.write(f"FLOPs (MACs): {macs}\n")
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
            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(data.size(0), -1))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
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
                if args.with_reconstruction:
                    output, probs = model(data, target)
                    reconstruction_loss = F.mse_loss(output, data.view(data.size(0), -1), reduction='sum').item()
                    test_loss += loss_fn(probs, target, size_average=False).item()
                    test_loss += reconstruction_alpha * reconstruction_loss
                else:
                    output, probs = model(data)
                    test_loss += loss_fn(probs, target, size_average=False).item()
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
                    'capsres/{}/{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(
                        args.dataset, epoch, args.routing_iterations, args.with_reconstruction))
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f},{epoch_time:.2f}\n")

