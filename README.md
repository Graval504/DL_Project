# 2023년 2학기 딥러닝 기말 프로젝트
## ! 데이터셋은 리포지토리에 포함되지 않음 !
데이터는 eclass 공지사항에서 별도로 다운로드, train과 test 모두 data 디렉토리에 복사하여 주세요\
(data/train, data/test가 되도록 구성)\
\
데이터셋은 서경원 교수님 연구실에서 라벨링한 것이므로,\
해당 리포지토리에서는 데이터에 대한 간략한 설명 이미지만을 포함합니다.

![](./data.jpg)

발표 자료 (제작중) : https://www.miricanvas.com/v/12pa3cl
## Introduction
### 왜 나무 질병에 대해 연구해야하는가?

1. 조기 진단 및 모니터링

   인공지능을 활용한 IoT 조경수목은 병해를 조기에 선별하고 예후를 예측하여 조기진단 및 모니터링이 가능합니다.

2. 생태계 보전

   나무 질병의 조기 발견으로 인해 전염을 예방하고 퍼지는 것을 방지함으로써 산림 생태계를 보존하는 데 도움이 됩니다.

3. 임업사업 지원

   임업 산업에 적용되어 나무 생산성을 향상시키고 임산물 품질을 개선할 수 있습니다.

   예를 들어, 병해충이나 질병에 감염된 나무를 식별하여 신속하게 처리함으로써 수확량을 최적화할 수 있습니다.

## 추가 데이터셋 사용 - PlantVillage
transfer learning 과정에 PlantVillage 데이터셋을 추가로 사용하여 \
모델을 2차례에 걸쳐 학습을 진행하였습니다.\
PlantVillage 데이터셋 또한 리포지토리에 포함되지 않으며\
아래 캐글 주소에서 별도로 다운받을 수 있습니다.\
https://www.kaggle.com/datasets/emmarex/plantdisease
\
PlantVillage 데이터셋은 다운받으신 후 압축을 해제하여\
PlantVillage 폴더를 data 폴더에 위치시켜주시면 됩니다.

## How To Use
main.py는 총 다섯가지 함수로 구성되어있습니다.\
main_resnet 함수는 느티나무 질병데이터를 기반으로 kfold 모델을 학습하고 학습된 모델을 checkpoint/ResNet_Nfold.pt의 형태로 저장합니다.

main_loadmodel 함수는 checkpoint/ResNet_Nfold.pt 모델들을 불러와 이를 기반으로 test set의 성능을 평가하고\
confusion matrix 이미지를 저장합니다.

main_finetune 함수는 imagenet-21k pretrained VisionTransformer 모델을 기반으로 느티나무 질병데이터를 이용해 모델을 finetuning하며,\
학습된 kfold 모델을 checkpoint/VisionTransformer_Nfold.pt의 형태로 저장합니다.

main_finetuneWithPlantVillage 함수는 imagenet-21k pretrained VisionTransformer 모델을 PlantVillage 데이터셋에 대해 finetuning하여\
cheeckpoint/VisionTransformer_PlantVillage.pt로 저장하고, 해당 모델을 기반으로 느티나무 질병데이터에 대해 kfold finetuning을 진행하여\
학습된 모델을 checkpoint/VisionTransformer_Nfold.pt의 형태로 저장합니다.

main_load_finetunedmodel은 함수는 input bool값에 따라\
False인경우 checkpoint/finetune/VisionTransformer_Nfold.pt 모델들을 불러와 이를 기반으로 test set의 성능을 평가하고\
confusion matrix 이미지를 저장합니다.\
True인 경우 checkpoint/finetunePlantVillage/VisionTransformer_Nfold.pt 모델들을 불러와 이를 기반으로 test set의 성능을 평가하고\
confusion matrix 이미지를 저장합니다.

main.py 파일의 맨끝에 위치한 `if __name__=="__main__":` 문에서 실행하고자 하는 함수만을 주석해제한뒤 main.py를 실행하거나,\
다른 파일에서 `from main import main_load_finetunedmodel`과 같이 함수를 import하여 함수를 실행할 수 있습니다.

각각의 함수에서 사용하고자 하는 데이터셋은 data/ 디렉토리 내에 위치해야합니다.
