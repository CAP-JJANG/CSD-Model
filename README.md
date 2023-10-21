## :raised_hands: 소개
**[ENG]**  
CSD-Model that learns and distinguishes images created using ResNet-34 architecture.

<br>

**[KOR]**  
[MakeDataset](https://github.com/CAP-JJANG/MakeDataset)를 활용하여 음향신호를 스펙트로그램으로 변환하고, 이 스펙트로그램 데이터를 CSD-Model의 학습 데이터셋으로 활용합니다.  
CSD-Model은 ResNet-34 아키텍처를 기반으로한 이미지 분류 모델로, 스펙트로그램의 시각적인 특징을 추출하는 손글씨 음향신호 인식 딥러닝 모델입니다.



<br><br>
## 💪 주요 기능
**[ENG]**
1. Set up the GPU usage environment in PyTorch.
2. Configure transformations that define data preprocessing and normalization for input images.
3. Define the dataset and apply the data transform.
4. Create an image classification model using the ResNet-34 architecture.
5. Apply L2 normalization.
6. K-Fold cross-validation learns the model and evaluates its performance.
7. Save the model weight if you have the highest accuracy per fold.
8. Save the learning and test results to a file.

<br>

**[KOR]**
1. PyTorch에서 GPU 사용 환경을 설정합니다.
2. 입력 이미지에 대한 데이터 전처리 및 정규화를 정의하는 변환을 구성합니다.
3. 데이터셋을 정의하고, 데이터 변환을 적용합니다.
4. ResNet-34 아키텍처를 사용하여 이미지 분류 모델을 생성합니다.
5. L2 정규화를 적용합니다.
6. K-Fold 교차 검증을 통해 모델을 학습하고 성능을 평가합니다.
7. 폴드별 최고 정확도를 가진 경우 모델 가중치를 저장합니다.
8. 학습 및 테스트 결과를 파일에 저장합니다.
   

<br><br>
## 🦾 주요 기술
**Server - Django**
* PyCharm IDE
* Python 3.9.13
* Scikit_learn 1.3.1
* Torch 1.13.1
* Torchvision 0.14.1

<br><br>
## 🧬 모델 아키텍처
<div align="center">
  <img width="60%" alt="image" src="https://github.com/CAP-JJANG/.github/assets/92065911/7fcd5810-2541-4a52-a0aa-a758c61e8fc8">
</div>

<br><br>
## ⭐️ 설치 방법
1. clone [github 리포지토리 주소]
2. 가상환경 생성
   ```
   python -m venv venv
   ```
   또는
   
   ```
   python3 -m venv venv
   ```
3. 가상환경 실행
    - Windows
       ```
       venv\Scripts\activate
       ```
    - macOS 및 Linux
       ```
       source venv/bin/activate
       ```
4. pip 최신버전으로 업그레이드
   ```
   python -m pip install --upgrade pip
   ```
    또는
    
   ```
   python3 -m pip install --upgrade pip
   ```
5. 패키지 설치
   ```
   pip install -r requirements.txt
   ```
   또는
   
   ```
   pip3 install -r requirements.txt
   ```
6. 프로젝트 Run

<br><br>
## 🤖 라이센스
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/CAP-JJANG/CSD-Model/blob/main/LICENSE) file for details.  
[OSS Notice](https://github.com/CAP-JJANG/CSD-Model/blob/main/OSS-Notice.md) sets forth attribution notices for third party software that may be contained in this application.

