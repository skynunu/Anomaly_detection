DATANAME 변수를 통해 HAI 데이터 이름만 기입하면 됩니다.
HAI 데이터 파일은 https://github.com/icsdataset/hai 에서 HAI-21.03 파일 참고
data는 data폴더에 저장하였습니다. HAI데이터의 경우, data/HAI2.0/를 통해 실행되며, 
HAI2.0폴더는 training폴더, validation폴더로 나눠서 데이터를 저장했습니다.

실행 방법은 다음과 같습니다.
(1) Autoencoder_GRU.ipynb 파일의 파라미터 세팅부분에서 파일이름, 데이터 종류, 하이퍼 파라미터를 설정합니다.
(2)파일 전체를 run합니다.
(3)그다음 파일 밑에 있는 threshold값을 예측 결과값에 따라서 적절하게 바꿔줍니다.
(4)실행결과는 hai_two 폴더에 저장됩니다. 평가결과는 hai_two폴더에 있는 evaluation 폴더에 저장됩니다.
