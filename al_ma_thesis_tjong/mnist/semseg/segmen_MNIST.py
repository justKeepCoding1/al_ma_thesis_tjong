import torch
from torchvision import transforms, models
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np

from active_learning_prototypes.presets.datasets_preset import MNIST_JPG_Dataset


'''
Prototype for semantic segmentation using MNIST
'''

# Background color persist ist original color (0, 0, 0)
label_colors = np.array([
    (128, 192, 192),    # 0
    (0, 128, 0),        # 1
    (128, 128, 0),      # 2
    (0, 0, 128),        # 3
    (128, 0, 128),      # 4
    (0, 128, 128),      # 5
    (128, 128, 128),    # 6
    (64, 0, 0),         # 7
    (64, 0, 128),        # 8
    (64, 128, 0)])      # 9


def decode_segmap(image, nc=21):
    labels = label_colors
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = labels[l, 0]
        g[idx] = labels[l, 1]
        b[idx] = labels[l, 2]
        rgb = np.stack([r, g, b], axis=2)

    return rgb

# train here
def train(model, train_loader, optimizer, log_step, epoch):
    model.train()
    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.to("cpu"), target.to("cpu")
        optimizer.zero_grad()  # reset param gradient
        output = model(data)
        output_label = output["out"]
        loss = torch.nn.CrossEntropyLoss()(output_label, target)  # or nll_loss
        loss.backward()
        optimizer.step()
        if batch_i % log_step == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_i * len(data), len(train_loader.dataset),
                       100. * batch_i / len(train_loader), loss.item()))

# test here
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    torch.no_grad()
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # training param
    batch_train = 64
    batch_test = 100
    epochs = 1
    lr = 0.9
    momentum = 0.9
    log_n_batch = 1
    torch.manual_seed(1)  # reset random for reproducibility
    root = "../../../data/MNIST_jpg"

    # About datasets
    trf = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    train_set = MNIST_JPG_Dataset(root=root, train_flag=True, n_data=10000)
    val_set = MNIST_JPG_Dataset(root=root, train_flag=False, n_data=1000)
    train_loader = DataLoader(train_set, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_test, shuffle=False)

    # About model
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # train and eval
    for epoch_i in range(1, epochs + 1):
        train(model, train_loader, optimizer=optimizer, log_step=log_n_batch, epoch=epoch_i)
        test(model, val_loader)
        scheduler.step()


if __name__ == '__main__':
    main()