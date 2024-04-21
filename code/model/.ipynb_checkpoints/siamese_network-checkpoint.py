from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import random
import shutil
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import datetime

# SiameseDataSet
class SiameseDataset(Dataset):
    def __init__ (self, pairs, transform = None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # First Image Get
        img1 = Image.open(pair[0])
        # print(pair[0])
        # print(pair[1])
        # print(pair[2])
        # Second Image Get
        img2 = Image.open(pair[1])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = pair[2]
        return img1, img2, torch.tensor(label, dtype = torch.float32)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Model
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # Output size: (128 - 10)/1 + 1 = 119
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output size: 119/2 = 59.5 -> 59
            
            nn.Conv2d(64, 128, 7),  # Output size: (59 - 7)/1 + 1 = 53
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output size: 53/2 = 26.5 -> 26
            
            nn.Conv2d(128, 128, 4),  # Output size: (26 - 4)/1 + 1 = 23
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output size: 23/2 = 11.5 -> 11
            
            nn.Conv2d(128, 256, 4),  # Output size: (11 - 4)/1 + 1 = 8
            nn.ReLU(inplace=True),
            # Max Pooling이 추가되지 않음, 최종 출력 크기: 8x8
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256*8*8, 4096),  # 입력 크기 조정: 256 채널 * 8 * 8
            nn.Sigmoid()
        )
        
        self.fc_out = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward_once(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # L1
        distance = torch.abs(output1 - output2)
        out = self.fc_out(distance)
        return self.sigmoid(out)

NORMAL_PATH = 'Normal_PATH'
ABNORMAL_PATH = 'Abnormal_PATH'
# DataSet 개수 동일하게 Balancing
BALANCE_ABNORMAL_PATH = 'Balance Abnormal_path'

# Data Load
def data_load(normal_path, abnormal_path):
    normal_data = []
    abnormal_data = []
    
    # normal data
    for nl in os.listdir(normal_path):
        normal_sample_path = os.path.join(normal_path, nl)
        normal_data.append(normal_sample_path)

    for anl in os.listdir(abnormal_path):
        abnormal_sample_path = os.path.join(abnormal_path, anl)
        abnormal_data.append(abnormal_sample_path)
    print(f"[+] Normal Data : {len(normal_data)} \n[+] Abnormal Data : {len(abnormal_data)}")
    return normal_data, abnormal_data

# DataSet Split(8:2)
def split_data(normal_files, abnormal_files, test_size = 0.2):
    normal_train, normal_test, abnormal_train, abnormal_test = train_test_split(normal_files, abnormal_files, test_size = test_size, random_state = 42)
    
    print(f"[+] Normal Train : {len(normal_train)}\n[+] Normal_test : {len(normal_test)}\n[+] Abnormal_train : {len(abnormal_train)}\n[+] Abnormal_test : {len(abnormal_test)}")
    return normal_train, normal_test, abnormal_train, abnormal_test

# Image Pair 생성
def generate_image_pair(normal_files, abnormal_files, is_train, data_len):
    same_class_pair = []
    other_class_pair = []

    for i in range(data_len //2):
        # Same Class Sampling(label = 1)
        if len(normal_files) >= 1:
            idx1, idx2 = random.sample(range(len(normal_files)), 2)
            label = 1
            if i % 2 == 0:
                same_class_pair.append((normal_files[idx1], normal_files[idx2], label))
            else:
                same_class_pair.append((abnormal_files[idx1], abnormal_files[idx2], label))

        # Other Class Sampling(label = 0)
        if len(abnormal_files) >= 1:
            idx1, idx2 = random.sample(range(len(abnormal_files)), 2)
            label = 0
            other_class_pair.append((normal_files[idx1], abnormal_files[idx2], label))

    return same_class_pair, other_class_pair

# Train
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, train_loss):
    model.train()
    running_loss = 0.0

    for batch_idx, (img1, img2, label) in enumerate(train_loader, 1):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, label.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    average_loss = running_loss / len(train_loader)
    train_loss.append(round(average_loss, 4))

    print(f'[+] {epoch+1} Epoch Training Loss: {average_loss:.4f}')
    return train_loss

# Train 시간 
def time_check(start):
    execute_time = time.time() - start
    times = str(datetime.timedelta(seconds = execute_time))
    short = times.split(".")[0]
    print(f"[+] 코드 실행 시간 : {short}")

# Loss 변화 시각화
def loss_change_visual(x, y):
    plt.plot(x, y, color = 'red', marker = 'o', alpha = 0.5, linewidth = 2)
    plt.title("Rate of change(Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
  
#  Data Unbalance
if not os.path.isdir(BALANCE_ABNORMAL_PATH):
    os.mkdir(BALANCE_ABNORMAL_PATH)

    try:
        # 데이터 언밸런싱 해결
        abnormal_data = os.listdir(ABNORMAL_PATH)
        abnormal_data_len = len(abnormal_data)
        random_sample_idx = np.random.choice(range(abnormal_data_len), size=10000, replace=False)
    
        for i in random_sample_idx:
            file_name = abnormal_data[i]
            shutil.copyfile(f"{ABNORMAL_PATH}/{file_name}", f"{BALANCE_ABNORMAL_PATH}/{file_name}")

        balance_abnormal_data = os.listdir(BALANCE_ABNORMAL_PATH)
        print(f"[+] Data Balancing Finish! \n[+] Balance Abnormal Data : {len(balance_abnormal_data)}")

    except Exception as e:
        print(f"[-] Error resolving data unbalancing : {e}")

# DataSet Load
normal_data, abnormal_data = data_load(NORMAL_PATH, BALANCE_ABNORMAL_PATH)

# DataSet Split
normal_train, normal_test, abnormal_train, abnormal_test = split_data(normal_data, abnormal_data)

# Image Pair 생성
train_same_class_pairs, train_other_class_pairs = generate_image_pair(normal_train, abnormal_train, 'true', 30000)
test_same_class_pairs, test_other_class_pairs = generate_image_pair(normal_test, abnormal_test, 'true', 2000)

# DataSet Loader(Image)
train_dataset = SiameseDataset(train_same_class_pairs + train_other_class_pairs, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)

test_dataset = SiameseDataset(test_same_class_pairs + test_other_class_pairs, transform = transform)
test_loader = DataLoader(test_dataset, batch_size = 20)

# Model
model = SiameseNetwork()
criterion = torch.nn.BCELoss()


start = time.time()

# 전역 변수로 train_loss 리스트를 정의
train_loss = []
loss_early_exit_number = 10
quit_flag = False

# 모델, 손실 함수, 최적화 알고리즘 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)  # SiameseNetwork 정의 필요
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00006)
epoch_list = []

# 학습 실행
for epoch in range(200):
    if not quit_flag:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, train_loss)
        epoch_list.append(epoch+1)
        # 조기 종료 조건 확인
        if epoch >= loss_early_exit_number:
            if all(train_loss[-1] == loss for loss in train_loss[-loss_early_exit_number:]):
                print(f"조기 종료: {epoch+1} 에포크에서 종료")
                quit_flag = True
    else:
        break

time_check(start)
loss_change_visual(epoch_list, train_loss)

# Test
def evaluate_model(model, test_loader, device):
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0
    total = 0
    threshold = 0.5

    with torch.no_grad():  # 기울기 계산을 비활성화
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2).squeeze()  # 차원 축소
            # 예측값 생성
            # print(outputs)
            predicted = torch.where(outputs >= threshold, 1.0, 0.0).to(device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.4f}%')

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 평가 실행
evaluate_model(model, test_loader, device)
