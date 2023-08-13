import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch


# CUDA 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.init()

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 데이터 셋
image_dataset = datasets.ImageFolder(root='abc_testImages', transform=data_transforms)

# 학습, 테스트 데이터 분리
train_size = int(0.8 * len(image_dataset))  # 전체 데이터 비율 80%
test_size = len(image_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                         shuffle=False, num_workers=0)

# 학습, 검증 데이터 분리
train_size = int(0.8 * len(train_dataset))  # 검증 데이터 비율 80%
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=False, num_workers=0)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                        shuffle=False, num_workers=0)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ResNet18 모델 생성
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
resnet.fc = nn.Linear(512, 10) # 출력층의 뉴런 수는 10

