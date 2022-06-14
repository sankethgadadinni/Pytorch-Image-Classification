import torch
import torch.nn as nn
from data import ImageDataset
from model import CNNNet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def train(model, train_dataloader, valid_dataloader, optimizer, criterion, epochs):

    train_losses = []
    valid_losses = []
    for epoch in range(1, epochs + 1):

        train_loss = 0
        valid_loss = 0
        total = 0
        correct = 0

        for image, label in tqdm(train_dataloader):

            
            output = model(image)

            optimizer.zero_grad()

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()* image.size(0)


        
        model.eval()

        for image, label in tqdm(valid_dataloader):

            output = model(image)

            loss = criterion(output, label)

            valid_loss += loss.item() * image.size(0)

            _, predicted = torch.max(output.data, 1)

            total += label.size(0)
            
            correct += (predicted == label).sum().item()



        train_loss = train_loss/len(train_dataloader.sampler)
        train_losses.append(train_loss)

        valid_loss = valid_loss/len(valid_dataloader.sampler)
        valid_losses.append(valid_loss)

        print("Epoch: {}\tTraining Loss: {}\t Valid Loss: {}\t Valid Acc: {}".format(epoch, train_loss, valid_loss, 100*(correct/total)))

        torch.save(model.state_dict(), 'models/model.ckpt')





if __name__ == '__main__':


    base_dir = '/Sanketh/Pytorch-Image-Classification/train'
    path = '/Sanketh/Pytorch-Image-Classification/train.csv'

    epochs = 50
    batch_size = 256
    lr = 0.001
    image_size = 255

    model = CNNNet()

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((image_size,image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    dataset = ImageDataset(base_dir, path, train_transform)

    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


    train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    train(model, train_dataloader, valid_dataloader, optimizer, criterion, epochs)

