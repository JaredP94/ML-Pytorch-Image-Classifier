import utils

from collections import OrderedDict
import json
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Main program function defined below
def main():
    in_args = utils.get_train_input_args()

    data_dir = in_args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])                                 

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=6)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True, num_workers=6)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if (in_args.arch.lower() == 'squeezenet1_1'):
        model = models.squeezenet1_1(pretrained=True)

    elif (in_args.arch.lower() == 'resnet18'):
        model = models.resnet18(pretrained=True)

    elif (in_args.arch.lower() == 'alexnet'):
        model = models.alexnet(pretrained=True)

    else:
        print("Invalid arch")
        exit()

    for param in model.parameters():
        param.requires_grad = False

    if (in_args.arch.lower() == 'squeezenet1_1'):
        # classifier is already composed of dropout and ReLU activation, so let's resize to our required outputs
        model.classifier[1] = nn.Conv2d(512, len(cat_to_name), kernel_size=(1,1), stride=(1,1))
        model.num_classes = len(cat_to_name)

    elif (in_args.arch.lower() == 'resnet18'):
        model.fc = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(model.fc.in_features, in_args.hidden_units)),
                      ('dropout1', nn.Dropout(0.5)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(in_args.hidden_units, len(cat_to_name)))]))

    elif (in_args.arch.lower() == 'alexnet'):
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(cat_to_name))
        model.num_classes = len(cat_to_name)

    params_to_update = []
    for param in model.parameters():
        if (param.requires_grad == True):
            params_to_update.append(param)

    # Config and train
    device = torch.device("cuda:0" if (in_args.gpu and torch.cuda.is_available()) else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=in_args.learn_rate)

    model.to(device)

    epochs = in_args.epochs
    total_time = time.time()

    for e in range(epochs):
        model.train()
        running_loss = 0
        start = time.time()

        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # training phase
            with torch.set_grad_enabled(True):
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        # validation phase
        model.eval()

        with torch.no_grad():
            valid_loss, accuracy = utils.validation(model, valid_loader, criterion, device)
        
        print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/index),
                "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                "Validation Accuracy: {:.3f} % ".format(100 * accuracy/len(valid_loader)),
                "Train time: {:.3f} seconds".format(time.time() - start))
        
        running_loss = 0
        model.train()

    print(f"Training complete. Total train time = {(time.time() - total_time):.3f} seconds")

    model.eval()

    with torch.no_grad():
        test_loss, accuracy = utils.validation(model, test_loader, criterion, device)
            
    print("Average Test Set Accuracy: {:.3f} %".format(100 * accuracy/len(test_loader)))

    with torch.no_grad():
        valid_loss, accuracy = utils.validation(model, valid_loader, criterion, device)
            
    print("Average Validation Set Accuracy: {:.3f} %".format(100 * accuracy/len(valid_loader)))

    checkpoint = {'hidden_units': in_args.hidden_units,
                'output_size': len(cat_to_name),
                'state_dict': model.state_dict(),
                'class_mapping': train_data.class_to_idx}
    
    save_path = ''

    if in_args.save_dir == '':
        save_path = 'checkpoint_{}.pth'.format(in_args.arch)
    else:
        save_path = in_args.save_dir + '/checkpoint_{}.pth'.format(in_args.arch)

    torch.save(checkpoint, save_path)

    print("Model save path: {}".format(save_path))

if __name__ == "__main__":
    main()

# Demo: python train.py flowers --arch squeezenet --learning_rate 0.001 --epochs 1 --gpu --save_dir checkpoints