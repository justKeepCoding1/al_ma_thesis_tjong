import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
import numpy as np

'''
convert MNIST into images for semseg
provide color coded image as ground truth
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
    (64, 0, 128),       # 8
    (64, 128, 0)])      # 9


def train(model, train_loader, optimizer, log_step, epoch):
    model.train()
    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.to("cpu"), target.to("cpu")
        optimizer.zero_grad()  # reset param gradient
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_i % log_step == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_i * len(data), len(train_loader.dataset),
                       100. * batch_i / len(train_loader), loss.item()))


# convert train data of MNIST to Semseg ready data
def train_data():

    # about datasets
    mnist_train = datasets.MNIST(root='data', train=True, download=True)
    mnist_trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=False)

    it = iter(mnist_train)

    for epoch_i in range(mnist_trainloader.__len__()):
        img, gt = next(it)

        # process and save image
        img_jpg = img.resize((300,300), Image.BICUBIC)
        img_tensor = transforms.ToTensor()(img_jpg)
        img_tensor = torch.where(img_tensor > 0.5, torch.ones(img_jpg.size), torch.zeros(img_jpg.size))
        img_jpg = transforms.ToPILImage()(img_tensor).convert("RGB")
        img_jpg.save(fp="data/MNIST_jpg/train/img/img"+str(epoch_i)+".jpg")

        # create GT image
        gt_r = torch.ones(img_tensor.shape) * label_colors[gt][0]
        gt_g = torch.ones(img_tensor.shape) * label_colors[gt][1]
        gt_b = torch.ones(img_tensor.shape) * label_colors[gt][2]
        
        # assign correct rgb value
        gt_r = torch.where(img_tensor > 0.001, gt_r, img_tensor)
        gt_g = torch.where(img_tensor > 0.001, gt_g, img_tensor)
        gt_b = torch.where(img_tensor > 0.001, gt_b, img_tensor)
        gt = torch.cat((gt_r, gt_g, gt_b), 0)

        # save gt.jpg
        gt_png = transforms.ToPILImage()(gt).convert("RGB")
        gt_png.save(fp="data/MNIST_jpg/train/gt/img"+str(epoch_i)+".png")

        if epoch_i % 100 == 0:
            print("Image nr: " + str(epoch_i) + "\n")


# convert test data of MNIST to Semseg ready data
def test_data():

    # about datasets
    mnist_test = datasets.MNIST(root='data', train=False, download=True)
    mnist_testloader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=False)

    it = iter(mnist_test)

    for epoch_i in range(mnist_testloader.__len__()):
        img, gt = next(it)

        # process and save image
        img_jpg = img.resize((300, 300), Image.BICUBIC)
        img_tensor = transforms.ToTensor()(img_jpg)
        img_tensor = torch.where(img_tensor > 0.5, torch.ones(img_jpg.size), torch.zeros(img_jpg.size))
        img_jpg = transforms.ToPILImage()(img_tensor).convert("RGB")
        img_jpg.save(fp="data/MNIST_jpg/test/img/img" + str(epoch_i) + ".jpg")

        # create GT image
        gt_r = torch.ones(img_tensor.shape) * label_colors[gt][0]
        gt_g = torch.ones(img_tensor.shape) * label_colors[gt][1]
        gt_b = torch.ones(img_tensor.shape) * label_colors[gt][2]

        # assign correct rgb value
        gt_r = torch.where(img_tensor > 0.001, gt_r, img_tensor)
        gt_g = torch.where(img_tensor > 0.001, gt_g, img_tensor)
        gt_b = torch.where(img_tensor > 0.001, gt_b, img_tensor)
        gt = torch.cat((gt_r, gt_g, gt_b), 0)

        # save gt.jpg
        gt_png = transforms.ToPILImage()(gt).convert("RGB")
        gt_png.save(fp="data/MNIST_jpg/test/gt/img" + str(epoch_i) + ".png")

        if epoch_i % 100 == 0:
            print("Image nr: " + str(epoch_i) + "\n")


if __name__ == '__main__':
    train_data()
    test_data()