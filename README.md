# Anomaly_detection
### (1)GRU
### (2)Autoencoder +GRU stacking
### (3)USAD
### (4)USAD + GRU stacking
### 환경설정 : GTX 3090 사용

(1),(2)는 hai 보안데이터를 활용해서 실험, (3)(4) wadi, swat데이터를 활용해서 실험
(1)는 HAICon2020 산업제어시스템 보안위협 탐지 AI 경진대회 1등 모델 코드를 참고한것이다. https://dacon.io/competitions/official/235624/overview/description 
(2)는 (1)를 보완해서 gru로 autoencoder를 만든 뒤 뒤에 back모델에 gru모델을 붙여서,스태킹모델을 만들었다.
(1)결과는 데이콘대회에서 ETaPR : 0.93793이 나왔고 이를 보완한 Autoencoder +GRU stacking 모델은 0.953이 나왔다. 

(3)USAD모델은 USAD : UnSupervised Anomaly Detection on Multivariate Time Series 논문을 참고하였다 
https://dl.acm.org/doi/10.1145/3394486.3403392
USAD 모델을 참고해서 SWaT데이터와 WADI데이터를 실험해보았다
