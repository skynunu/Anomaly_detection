# Anomaly_detection
## 1. Autoencoder +GRU stacking
* hai 보안데이터를 활용해서 실험
* HAICon2020 산업제어시스템 보안위협 탐지 AI 경진대회 1등 수상모델 참고
* 대회링크 : https://dacon.io/competitions/official/235624/overview/description 
* RNN계열의 GRU모델을 두개의 스태킹 모델로 변형해서 성능을 높임
* front모델을 GRU로 구성된 Autoencoder로 변경 (layers : 200-100-200), 그다음 back 모델은 기본적인 GRU모델로 구성(layer:200-200-200)
* HAICon2020 산업제어시스템 대회에서 1등모델은 ETaPR : 0.93793이 나왔고 이를 보완한 Autoencoder +GRU stacking 모델은 0.953이 나왔다. 

## 2. USAD
* reference paper : UnSupervised Anomaly Detection on Multivariate Time Series 논문을 참고하였다 
* paper link : https://dl.acm.org/doi/10.1145/3394486.3403392
* data : USAD 모델을 참고해서 SWaT데이터와 WADI데이터를 실험해보았다



## 3. USAD + GRU stacking
## 환경설정 : GTX 3090 사용









