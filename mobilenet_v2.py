from tqdm import tqdm
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision


transform_dict = {
    "train": transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

data_train = torchvision.datasets.ImageFolder(
    root="./data", transform=transform_dict["train"]
)
data_test = torchvision.datasets.ImageFolder(
    root="./data", transform=transform_dict["test"]
)

train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=32, shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    data_test, batch_size=32, shuffle=False, num_workers=4
)

model = models.mobilenet_v2(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2, inplace=False), nn.Linear(in_features, 5)
)

# 訓練に際して、可能であればGPU（cuda）を設定します。GPUが搭載されていない場合はCPUを使用します
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(dataloader, model, loss_fn, optimizer):
    for (X, y) in tqdm(dataloader, total=len(dataloader)):
        X, y = X.to(device), y.to(device)

        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model)
print("Done!")
