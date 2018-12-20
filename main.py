import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import Generator_UNet, Discriminator
import numpy as np
import matplotlib.pyplot as plt

plot_dis = []
plot_gen = []
    
def train(args, D, G, batch_size, device, train_loader, optimizer_gen, optimizer_dis, epoch, test_loader, fig, ax):

    running_loss_dis = 0
    count_dis = 0
    count_gen = 0
    running_loss_gen = 0
    dis_accuracy = 0

    for no, (data, target) in enumerate(train_loader):

        for param in G.parameters():
            param.requires_grad = False

        for param in D.parameters():
            param.requires_grad = True

        G.eval()
        D.train()

        optimizer_dis.zero_grad()

        data, target = data.to(device).view(-1, 28*28), torch.FloatTensor(np.ones([data.size()[0], 1])).to(device)
        
        output = D(data)
        dis_accuracy = (dis_accuracy*count_dis + np.mean((output.data.cpu().numpy() >= 0.5).astype(np.float32)))/(count_dis + 1)
        loss = F.binary_cross_entropy(output, target)
        running_loss_dis = (running_loss_dis*count_dis + loss.data.cpu().numpy())/(count_dis + 1)
        # running_loss_dis = loss.data.cpu().numpy()/2
        count_dis += 1
        loss.backward()
        optimizer_dis.step()


        optimizer_dis.zero_grad()
        input_g, target = torch.FloatTensor(np.random.normal(size=batch_size*100).reshape(batch_size, 100)).to(device), torch.FloatTensor(np.zeros([batch_size, 1])).to(device)
        
        G_output = G(input_g)
        output = D(G_output)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        dis_accuracy = (dis_accuracy*count_dis + np.mean((output.data.cpu().numpy() < 0.5).astype(np.float32)))/(count_dis + 1)
        running_loss_dis = (running_loss_dis*count_dis + loss.data.cpu().numpy())/(count_dis + 1)
        # running_loss_dis += loss.data.cpu().numpy()/2
        count_dis += 1
        optimizer_dis.step()

        for param in G.parameters():
            param.requires_grad = True

        for param in D.parameters():
            param.requires_grad = False

        

        G.train()
        D.eval()

        optimizer_gen.zero_grad()

        
        input_g, target = torch.FloatTensor(np.random.normal(size=batch_size*100).reshape(batch_size, 100)).to(device), torch.FloatTensor(np.ones([batch_size, 1])).to(device)
        G_output = G(input_g)
        output = D(G_output)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        running_loss_gen = (running_loss_gen*count_gen + loss.data.cpu().numpy())/(count_gen + 1)
        count_gen += 1
        optimizer_gen.step()

        if no % 100 == 0:
            test(args, D, G, args.test_batch_size, device, test_loader, fig, ax)

            
    plot_dis.append(running_loss_dis)
    plot_gen.append(running_loss_gen)

    plt.figure(2)
    plt.clf()
    plt.plot(plot_gen)
    plt.plot(plot_dis)
    plt.pause(1)
    plt.figure(1)



def test(args, D, G, batch_size, device, test_loader, fig, ax):

    G.eval()
    D.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data = torch.FloatTensor(np.random.normal(size=batch_size*100).reshape(batch_size, 100)).to(device)
        data = data.to(device)
        output = G(data).view(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), 28, 28).data.cpu().numpy()

        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                col.imshow(output[i, j])
        plt.pause(1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    G = Generator_UNet().to(device)
    D = Discriminator().to(device)

    optimizer_gen = optim.Adam(G.parameters(), lr=args.lr)
    optimizer_dis = optim.Adam(D.parameters(), lr=args.lr)

    fig, ax = plt.subplots(nrows=int(np.sqrt(args.test_batch_size)), ncols=int(np.sqrt(args.test_batch_size)))

    for epoch in range(1, args.epochs + 1):
        train(args, D, G, args.batch_size, device, train_loader, optimizer_gen, optimizer_dis, epoch, test_loader, fig, ax)
        

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()