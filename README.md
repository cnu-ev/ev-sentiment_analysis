# ev-sentiment_analysis
충남대학교 제1회 2019 bottom-up 경진대회 EV팀 <br>
목표 : 감성분석 시각화<br>
머신러닝을 활용한 NLP의 활용 (sentiment analysis)<br>

공개된 데이터셋(Naver sentiment movie corpus) : <https://github.com/e9t/nsmc/>
20만 문장 (train 15만 + test 5만)

오픈 소스코드를 활용하여
직접 수집한 데이터셋 (naver_movie_scraper폴더 참조) : <https://github.com/lovit/naver_movie_scraper>
27만문장

학습 및 평가에 사용된 총 데이터셋 47만문장 (train 40만 + text 7.5만)

두가지 버전으로 제작하였습니다.

Master - 문장들을 Word2Vec 임베딩하고 입력데이터의 특징벡터를 추출하여 다층 신경망을 통해 Binary logistic regression 모델

Pytorch_ver - 문장들을 Glove 임베딩하여 LSTM + Attention 를 사용하여 학습시킨 RNN 모델

두 방법의 성능 차이
