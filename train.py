import argparse
from model import resnet18
from torch import optim
import torchvision
from torch.backends import cudnn
import torchvision.transforms as transforms
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from torch.autograd import Variable

def main(config):
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.result_path = os.path.join(config.result_path, config.aug)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    writer = SummaryWriter(log_dir=config.result_path)

    if config.aug == 'cutout':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            Cutout(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    lr = config.lr

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    net = resnet18()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(net.parameters()), lr=lr)

    epochs = config.num_epochs
    best_acc = 0.0

    sava_path = os.path.join(config.result_path, 'model.pkl')
    if os.path.isfile(sava_path):
        print('Have finished')
        return

    train_steps = len(trainloader)
    val_num = len(testset)
    test_steps = len(testloader)

    for epoch in range(epochs):
        #train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(trainloader, file=sys.stdout)
        for images, labels in train_bar:
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)

            if config.aug == 'mixup':
                inputs, labels_a, labels_b, lam = mixup_data(images, labels)
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))
                output = net(inputs)
                loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
                predict = torch.max(output, dim=1)[1]
                train_acc += (lam*predict.eq(labels_a.data).cpu().sum().float()+ \
                              (1-lam)*predict.eq(labels_b.data).cpu().sum().float())
            elif config.aug == 'cutmix':
                inputs, labels_a, labels_b, lam = cutmix((images, labels))
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))
                output = net(inputs)
                loss = cutmix_criterion(criterion, output, labels_a, labels_b, lam)
                predict = torch.max(output, dim=1)[1]
                train_acc += (lam * predict.eq(labels_a.data).cpu().sum().float() + \
                              (1 - lam) * predict.eq(labels_b.data).cpu().sum().float())
            else:
                output = net(images)
                loss = criterion(output, labels)
                predict = torch.max(output, dim=1)[1]
                train_acc += torch.eq(predict, labels).sum().item()

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        mean_loss = running_loss / train_steps
        writer.add_scalar('Train/loss', mean_loss, epoch+1)

        #valid
        net.eval()
        acc = 0.0
        running_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(testloader, file=sys.stdout)
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                output = net(images)
                loss = criterion(output, labels)

                predict = torch.max(output, dim=1)[1]
                acc += torch.eq(predict, labels).sum().item()
                running_loss += loss.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_acc = acc / val_num
        mean_loss = running_loss / test_steps
        writer.add_scalar('Test/accuracy', val_acc, epoch+1)
        writer.add_scalar('Test/loss', mean_loss, epoch+1)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, mean_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), sava_path)

    print('Finished Training')

if __name__ == '__main__':
    os.chdir(sys.path[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--aug', type=str, default='baseline')
    config = parser.parse_args()

    main(config)
