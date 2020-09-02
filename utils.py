import argparse
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils

def get_train_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='dir', default='flowers', help='Image folder')
    parser.add_argument('--save_dir', dest='save_dir', default='', help='Save directory for model checkpoint')
    parser.add_argument('--arch', dest='arch', default='squeezenet1_1', help='Pretrained model architecture [squeezenet1_1 / resnet18]')
    parser.add_argument('--learning_rate', dest='learn_rate', default=0.001, type=float, help='Learning rate for training')
    parser.add_argument('--hidden_units', dest='hidden_units', default=256, type=int, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, help='Number of epochs to train model')
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False, help='Train model using GPU (if available)')

    return parser.parse_args()

def get_predict_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='dir', help='Path to image (e.g. flowers/test/1/image_06743.jpg)')
    parser.add_argument(dest='checkpoint', help='Name of model checkpoint to load [checkpoint_squeezenet1_1.pth / checkpoint_resnet18.pth]')
    parser.add_argument('--top_k', dest='top_k', default=5, type=int, help='Number of matching classes to return')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='File containing category to name mapping')
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False, help='Train model using GPU (if available)')

    return parser.parse_args()

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if ('squeezenet1_1' in filepath):
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, checkpoint['output_size'], kernel_size=(1,1), stride=(1,1))
        model.num_classes = checkpoint['output_size']
    elif ('resnet18' in filepath):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(model.fc.in_features, checkpoint['hidden_units'])),
                      ('dropout1', nn.Dropout(0.5)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size']))]))
        
    model.load_state_dict(checkpoint['state_dict'])
    mappings = checkpoint['class_mapping']
    
    return model, mappings

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image).resize((256, 256))

    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    img = np.array(img).transpose((2, 0, 1)) / 256
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img -= means
    img /= stds

    img_tensor = torch.Tensor(img)

    return img_tensor

def predict(image_path, model, class_mappings, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img_tensor = process_image(image_path).to(device).view(1, 3, 224, 224)

    model.to(device)

    with torch.no_grad():
        model.eval()
        out = model(img_tensor)
        ps = F.softmax(out, dim=1)

        topk, topclass = ps.topk(topk, dim=1)
        top_classes = [list(class_mappings)[class_] for class_ in topclass.cpu().numpy()[0]]
        top_p = topk.cpu().numpy()[0]

        return top_p, top_classes