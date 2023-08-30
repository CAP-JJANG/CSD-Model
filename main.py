import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from sklearn.model_selection import KFold

EPOCH = 10

# CUDA 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.init()
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 데이터셋
train_dataset = datasets.ImageFolder(root='trainImages', transform=data_transforms)
test_dataset = datasets.ImageFolder(root='testImages', transform=data_transforms)
# 모델 생성 후 L2 정규화 적용
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34')
resnet.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # 드롭아웃 추가
    nn.Linear(512, 26)  # 출력층의 뉴런 수는 26
)

# L2 정규화 적용할 가중치 파라미터를 모아둘 리스트
weight_decay_params = []
bias_params = []

# 정규화 비율 설정
weight_decay = 0.001

# 모든 가중치 파라미터를 추출하여 정규화 적용
for name, param in resnet.named_parameters():
    if 'bias' in name:
        bias_params.append(param)
    else:
        weight_decay_params.append(param)

# 교차 검증을 위한 KFold 객체 생성
kfold = KFold(n_splits=5, shuffle=True)

# 각 폴드에 대해 반복
for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset)):

    # 데이터셋 분할
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                              sampler=train_sampler, num_workers=0)
    valloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                            sampler=val_sampler, num_workers=0)

    # 모델 학습을 위한 하이퍼파라미터 설정
    criterion = nn.CrossEntropyLoss()

    # 정규화를 위한 optimizer 생성
    optimizer = optim.SGD([
        {'params': weight_decay_params, 'weight_decay': weight_decay},
        {'params': bias_params, 'weight_decay': 0.0}
    ], lr=0.001, momentum=0.9)

    # 학습률 스케줄러 생성
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 모델 학습
    resnet.to(device)
    best_accuracy = 0.0  # 최고 정확도를 저장하기 위한 변수
    train_losses = []
    val_losses = []
    accuracies = []
    for epoch in range(EPOCH):
        running_loss = 0.0
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1500 == 1499:
                train_loss = running_loss / 100
                train_losses.append(train_loss)
                running_loss = 0.0
        # 학습률 스케줄링
        scheduler.step()
        # 검증 데이터셋을 이용하여 모델 성능 평가
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = resnet(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = loss.item()
        val_losses.append(val_loss)
        accuracy = 100 * val_correct / val_total
        accuracies.append(accuracy)
        print('[fold: %d, epoch: %d] train loss: %.3f, val loss: %.3f, accuracy: %.2f' % (
        fold + 1, epoch + 1, train_losses[-1], val_losses[-1], accuracy))

        # 최고 정확도일 때 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(resnet.state_dict(), 'resnet34_best.pth')

    # 결과 저장
    result_file = "result_resnet34.txt"
    with open(result_file, "a") as f:
        f.write("******** Fold {} ******/**\n".format(fold+1))
        for epoch in range(EPOCH):
            f.write("Epoch: {}\n".format(epoch + 1))
            f.write("Train Loss: {:.3f}\n".format(train_losses[epoch]))
            f.write("Val Loss: {:.3f}\n".format(val_losses[epoch]))
            f.write("Accuracy: {:.2f}%\n".format(accuracies[epoch]))
            f.write("----------\n")

# 테스트 데이터셋에 대한 DataLoader 생성
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 모델 평가
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = 100 * correct / total
print('Accuracy of the network on the testImages images: %.2f %%' % test_accuracy)

# 모델 가중치 저장
torch.save(resnet.state_dict(), 'resnet34_weights.pth')

# 결과 저장
# result_file = "result_resnet34.txt"
with open(result_file, "a") as f:
    #     for epoch in range(EPOCH):
    #         f.write("Epoch: {}\n".format(epoch + 1))
    #         f.write("Train Loss: {:.3f}\n".format(train_losses[epoch]))
    #         f.write("Val Loss: {:.3f}\n".format(val_losses[epoch]))
    #         f.write("Accuracy: {:.2f}%\n".format(accuracies[epoch]))
    #         f.write("----------\n")
    f.write("\nTest Accuracy: {:.2f}%\n".format(test_accuracy))