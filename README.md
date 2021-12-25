# Anomaly_detection
## 1. Autoencoder +GRU stacking
* hai 보안데이터를 활용해서 실험
* HAICon2020 산업제어시스템 보안위협 탐지 AI 경진대회 1등 수상모델 참고
* 대회링크 : https://dacon.io/competitions/official/235624/overview/description 
* RNN계열의 GRU모델을 두개의 스태킹 모델로 변형해서 성능을 높임
* front모델을 GRU로 구성된 Autoencoder로 변경 (layers : 200-100-200), 그다음 back 모델은 기본적인 GRU모델로 구성(layer:200-200-200)
* HAICon2020 산업제어시스템 대회에서 1등모델은 ETaPR : 0.93793이 나왔고 이를 보완한 Autoencoder +GRU stacking 모델은 0.953이 나왔다. 

## 2. USAD
* reference paper : UnSupervised Anomaly Detection on Multivariate Time Series 
* paper link : https://dl.acm.org/doi/10.1145/3394486.3403392
* data :  SWaT데이터와 WADI데이터를 실험

## 3. USAD + GRU stacking
* front모델 : USAD, back모델 : GRU
* USAD모델에서, 출력층을 anomaly score가 아닌, 전체 feature값이 나오도록 수정하고 이를 다시 GRU모델에 입력값으로 넣음










