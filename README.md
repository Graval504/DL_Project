# 2023년 2학기 딥러닝 기말 프로젝트
## ! 데이터셋은 리포지토리에 포함되지 않음 !
데이터는 eclass 공지사항에서 별도로 다운로드, train과 test 모두 data 디렉토리에 복사하여 주세요\
(data/train, data/test가 되도록 구성)\
\
데이터셋은 서경원 교수님 연구실에서 라벨링한 것이므로,\
해당 리포지토리에서는 데이터에 대한 간략한 설명 이미지만을 포함합니다.

![](./data.jpg)

## TODO
### data_processing.py
collate_fn 만들기 ✔\
data size가 균일하지 않으므로, 이를 위한 collate_fn 필요 ✔\
padding or crop or stretch -> 아마 padding 아니면 stretch 사용 ✔\
-> Lanzcos 보간법 이용하여 resize 후 패딩하는 방식 사용