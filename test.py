import argparse
from model import resnet18
from torch import optim, nn
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_path = config.net_path
    # writer = SummaryWriter(log_dir=config.result_path)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    net = resnet18()
    net.to(device)
    net.load_state_dict(torch.load(net_path))

    criterion = nn.CrossEntropyLoss()
    val_num = len(testset)
    test_steps = len(testloader)
    epochs = config.num_epochs
    best_acc = 0.0
    for epoch in range(epochs):
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
        # writer.add_scalar('Test/accuracy', val_acc, epoch + 1)
        # writer.add_scalar('Test/loss', mean_loss, epoch + 1)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, mean_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy is: {}'.format(best_acc))


if __name__ == '__main__':
    os.chdir(sys.path[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--aug', type=str, default='baseline')
    parser.add_argument('--net_path', type=str, default='')
    config = parser.parse_args()

    main(config)