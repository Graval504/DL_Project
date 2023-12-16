# 2023년 2학기 딥러닝 기말 프로젝트
## ! 데이터셋은 리포지토리에 포함되지 않음 !
데이터는 eclass 공지사항에서 별도로 다운로드, train과 test 모두 data 디렉토리에 복사하여 주세요\
(data/train, data/test가 되도록 구성)\
\
데이터셋은 서경원 교수님 연구실에서 라벨링한 것이므로,\
해당 리포지토리에서는 데이터에 대한 간략한 설명 이미지만을 포함합니다.

![](./data.jpg)

## 추가 데이터셋 사용 - PlantVillage
transfer learning 과정에 PlantVillage 데이터셋을 추가로 사용하여 \
모델을 2차례에 걸쳐 학습을 진행하였습니다.\
PlantVillage 데이터셋 또한 리포지토리에 포함되지 않으며\
아래 캐글 주소에서 별도로 다운받을 수 있습니다.\
https://www.kaggle.com/datasets/emmarex/plantdisease\
\
PlantVillage 데이터셋은 다운받으신 후 압축을 해제하여 PlantVillage 폴더를\
data 폴더에 위치시켜주시면 됩니다.\

## TODO
### data_processing.py
collate_fn 만들기 ✔\
data size가 균일하지 않으므로, 이를 위한 collate_fn 필요 ✔\
padding or crop or stretch -> 아마 padding 아니면 stretch 사용 ✔\
-> Lanzcos 보간법 이용하여 resize 후 패딩하는 방식 사용

K fold crossvalidation 구현 ✔\
-> https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch 참고하기\

data transform(data augmentation) 적용 ✔

### model.py
모델 구현 -> resnet 사용 ✔

### FINAL
모델 학습 및 성능 평가 \
confusion matrix 구현 \
data augmentation \
-> 시간이많다면 randaugment를 grid search해보기(lr_rate를 늘리고, 에포크 수 줄여서) \
-> MixUp 구현 \

Transfer Learning\
Imagenet-21k model finetuning ✔\
Imagenet-21k model -> plantvillage fintuning -> our dataset fintuning
