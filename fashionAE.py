''' 1. Module Import '''
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

''' 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인 '''
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32
EPOCHS = 10

''' 3. FashionMNIST 데이터 다운로드 (Train set, Test set 분리하기) '''
train_dataset = datasets.FashionMNIST(root = "../data/FashionMNIST",
                                      train = True,
                                      download = True,
                                      transform = transforms.ToTensor())       # 0~255 범위의 픽셀 값을 0~1 범위로 정규화하는 과정은 이 코드에서 transforms.ToTensor() 부분에 포함

test_dataset = datasets.FashionMNIST(root = "../data/FashionMNIST",
                                     train = False,
                                     transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)

''' 4. 데이터 확인하기 (1) '''
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

''' 5. 데이터 확인하기 (2) '''
pltsize = 1
plt.figure(figsize=(10 * pltsize, pltsize))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
    plt.title('Class: ' + str(y_train[i].item()))
    

''' 6. AutoEncoder (AE) 모델 설계하기 '''
class AE(nn.Module):                    
    def __init__(self):
        super(AE,self).__init__()       # 부모 클래스인 nn.Module의 초기화 메서드를 실행. AE는 nn.Module을 상속받았기 때문에, PyTorch 내부 기능이 제대로 작동하려면 super()로 부모 클래스의 __init__()을 꼭 호출
        
        self.encoder = nn.Sequential(   # nn.Sequential을 사용해서 여러 층을 순차적으로 연결. 입력인 28×28 크기의 이미지를 펼쳐서(784차원) 점점 줄여 32차원의 잠재 벡터(latent vector) 로 만드는 과정
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,32),          # 마지막 쉼표는 문법적으로 꼭 필요하진 않지만, 가독성과 유지보수를 위해 자주 넣음
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded         # 튜플 형태로 여러 값을 동시에 반환
        
''' 7. Optimizer, Objective Function 설정하기 '''
model = AE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

print(model)

''' 8. AE 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, _) in enumerate(train_loader):   # 반복문에서 리스트나 데이터셋을 순회하면서 인덱스와 값 둘 다 동시에 가져오는 파이썬 문법
        image = image.view(-1, 28 * 28).to(DEVICE)          # AE의 Input은 28*28 크기의 1차원 레이어이므로 2차원 이미지 데이터를 1차원 데이터로 재구성해 할당해야 함. 
        target = image.view(-1, 28 * 28).to(DEVICE)         # target은 복원해야할 이미지 자체. AE는 정답 클래스는 필요 없지만 입력 이미지 그대로 복원하는 것이 목표. 
        optimizer.zero_grad()                               # 이전 배치에서 계산된 기울기를 모두 0으로 초기화. PyTorch는 기본적으로 기울기를 누적하기 때문에, 매 배치마다 초기화하지 않으면 이전 값이 계속 더해져서 학습 망가짐. 
        encoded, decoded = model(image)                     # 입력 이미지를 저차원 표현으로 압축하고, 다시 복원한 결과까지 한꺼번에 받아오는 과정
        loss = criterion(decoded, target)                   # 복원된 이미지 decoded와 원본 이미지 target 사이의 오차를 계산
        loss.backward()                                     # 그 오차를 기준으로 각 가중치에 대한 기울기(gradient)를 자동으로 계산 (역전파)
        optimizer.step()                                    # 계산된 기울기를 바탕으로 가중치를 실제로 업데이트해서 모델을 학습

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item()))
            
''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    real_image = []
    gen_image = []
    with torch.no_grad():
        for image, _ in test_loader:
            image = image.view(-1, 28 * 28).to(DEVICE)
            target = image.view(-1, 28 * 28).to(DEVICE)
            encoded, decoded = model(image)
            
            test_loss += criterion(decoded, image).item()
            real_image.append(image.to("cpu"))
            gen_image.append(decoded.to("cpu"))
            
    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)

    return test_loss, real_image, gen_image

''' 10. AutoEncoder 학습 실행하며 Test set의 Reconstruction Error 확인하기 '''
for Epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    test_loss, real_image, gen_image = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}".format(Epoch, test_loss))
    f, a = plt.subplots(2, 10, figsize = (10, 4))
    for i in range(10):
        img = np.reshape(real_image[0][i], (28, 28))
        a[0][i].imshow(img, cmap = "gray_r")
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    
    for i in range(10):
        img = np.reshape(gen_image[0][i], (28, 28))
        a[1][i].imshow(img, cmap = "gray_r")
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())
    plt.show()
