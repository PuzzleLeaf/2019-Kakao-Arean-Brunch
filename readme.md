# 2019-Kakao-Arena-Brunch (Team ZINNY)

### 개요

카카오 아레나에서 개최 된 [브런치 사용자를 위한 글 추천 대회](https://arena.kakao.com/c/2) 결과 입니다.

(brunch 데이터를 활용해 사용자의 취향에 맞는 글을 예측하기)



### 구조

이외에 **magazine.json** / **metadata.json** / **users.json** 데이터 모두 res 안에 포함되어 있습니다.

~~~
└── res
    ├── predict
    └── read
~~~



### 학습

config.py 에서 predict 유저를 설정 할 수 있습니다. (default : test.users)

~~~python
python train.py # 모델 학습
python inference.py # recommend.txt 파일 생성
~~~



