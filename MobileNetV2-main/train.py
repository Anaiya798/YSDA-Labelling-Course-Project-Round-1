import torch
import torchsummary


class CarNumbersDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.to('cuda')
    torchsummary.summary(model, (3, 224, 224))
    print(model)
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2, bias=True).to('cuda')
    torchsummary.summary(model, (3, 224, 224))
    print(model)
