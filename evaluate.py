import torch
import torch.nn as nn
import pandas as pd
from data import ImageDataset
from model import CNNNet
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

            
category_to_label = {"no_cactus":0,"has_cactus":1}
label_to_category = dict([(value,key) for key, value in category_to_label.items()])



def evaluate(model_path, dataloader):

    model = CNNNet()

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    model.eval()

    predicted_labels = []
    predicted_categories = []
    for image in tqdm(dataloader):

        output = model(image)
        _, predicted = torch.max(output.data, 1)

        predicted = [pred.item() for pred in predicted]
        category = [label_to_category[pred] for pred in predicted]
        
        predicted_labels.extend(predicted)
        predicted_categories.extend(category)

    return predicted_labels, predicted_categories



if __name__ == '__main__':

    test_transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((255,255)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_dataset = ImageDataset('/Sanketh/Pytorch-Image-Classification/test','/Sanketh/Pytorch-Image-Classification/sample_submission.csv', test_transform, False)

    test_dataloader = DataLoader(test_dataset, batch_size = 64)

    model_path = '/Sanketh/Pytorch-Image-Classification/model.ckpt'

    labels, categories = evaluate(model_path, test_dataloader)


    df = pd.read_csv("/Sanketh/Pytorch-Image-Classification/sample_submission.csv")

    df['has_cactus'] = labels
    df['category'] = categories

    df.to_csv("submit.csv", index=False)










