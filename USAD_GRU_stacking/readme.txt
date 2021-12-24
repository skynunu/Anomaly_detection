*데이터 설명
SWaT데이터의 경우 사용이 가능하지만, HAI와 WADI의 경우 사용하기 어렵습니다.
WADI DATA의 경우는 loss값이 NAN으로 떠서 오류가 뜨게 됩니다.
HAI DATA의 경우는 front모델인 USAD에서 학습을 하고 학습데이터를 예측한 값을 back모델에 넘겨줄 때 Cuda에서 Out of memory가 발생합니다.
data는 data폴더에 저장하였습니다. HAI데이터의 경우, data/HAI2.0/를 통해 실행되며, HAI2.0폴더는 training폴더, validation폴더로 나눠서 데이터를 저장했습니다.

*실행 방법은 다음과 같습니다.
USAD파일에서 Out of memory가 발생하므로 실행 파일을 구분하여 실행하였습니다. 

(1)USAD_front.ipynb 파일의 윗부분에 있는 파라미터 세팅부분에서 파일이름, 데이터 종류, 하이퍼 파라미터를 설정합니다.
(2)USAD_front.ipynb 파일 Data PreProcessing 이라고 적힌 부분 전까지 실행시킵니다. 이는 1차적으로 데이터를 학습한뒤, 
학습 데이터를 예측해서 저장하는 것입니다.
(3)HAI_back.ipynb를 실행합니다. testing이라고 적힌 부분 전까지 실행시킵니다. (2)에서 저장한 데이터를 바탕으로 학습을 진행합니다.
(4)USAD_front.ipynb로 돌아온 뒤, 파일의 윗부분에서  Data PreProcessing라고 적힌 부분전까지 실행하고  
파일 밑으로 내려와서 testing이라고 적힌 부분부터 끝까지 파일을 실행합니다. 
(5)HAI_back.ipynb로 와서 testing이라고 적힌 부분의 이후 부분을 실행시켜줍니다.

*실행결과는 usad_hai 폴더에 저장됩니다.

*결과값 ETaPR점수기준
