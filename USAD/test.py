import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from utils import *
from usad import *
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
device = get_default_device()

normal = pd.read_csv("./SWaT_Dataset_Normal_v1.csv")#, nrows=1000)
normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1) #axis는 열 드롭.
normal.shape
print(normal.shape) #(495000, 51)
print('normal')
print(normal.values)

# Transform all columns into float64
for i in list(normal):
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)
print('normal2')
print(normal.values)

#최소값(Min)과 최대값(Max)을 사용해서 '0~1' 사이의 범위(range)로 데이터를 표준화해주는 '0~1 변환
min_max_scaler = preprocessing.MinMaxScaler()

x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)
print('normal.head(2)')
print(normal.head(2))

attack = pd.read_csv("./SWaT_Dataset_Attack_v0.csv",sep=";")#, nrows=1000)
print('attack')
print(attack.values)
labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
# print('labels')
# print(labels)
# a = attack["Normal/Attack"].values[0]
# for label  in attack["Normal/Attack"].values:
#     print('label')
#     print(float(label!= 'Normal' ))
attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
print('attack.shape1')
print(attack.shape)

for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)
print('attack2')
print(attack.values)
# print('attack.shape2')
# print(attack.shape)
# print('labels') #0.0 0.0 0.0
# print(labels[0],labels[1],labels[2])


#normalization
x = attack.values
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)
print('attack.head(2)')
print(attack.head(2))
print('attack.shape3')
print(attack.shape)

#window
window_size=12
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
print('windows_normal.shape')
print(windows_normal.shape) #(494988, 12, 51)
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
print('windows_attack.shape')
print(windows_attack.shape) #(449907, 12, 51)

#training
BATCH_SIZE =  7919
N_EPOCHS = 100
hidden_size = 100

w_size=windows_normal.shape[1]*windows_normal.shape[2] #612
z_size=windows_normal.shape[1]*hidden_size #1200

print("w_size")
print(w_size)

print("z_size")
print(z_size)

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
print('windows_normal_train')
print(windows_normal_train.shape) #(395990, 12, 51)
print(windows_normal_train)
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]
print('windows_normal_val')
print(windows_normal_val.shape) #(98998, 12, 51)
print(windows_normal_val)

#view-->텐서의 차원을 변경 (395990,612)
train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#(98998,612)
val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = UsadModel(w_size, z_size)
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader)
# print(plot_history(history))
#객체를 디스크에 저장
torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "model.pth")

#Testing
# 객체를 역직렬화하여 메모리에 할당
checkpoint = torch.load("model.pth")
#역직렬화된 state_dict를 사용, 모델의 매개변수들을 불러옵니다.
# state_dict는 간단히 말해 각 체층을 매개변수 Tensor로 매핑한 Python 사전(dict) 객체입니다.
model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

results=testing(model,test_loader)
print('results')
print(results)
# datas = results.numpy()
# print(datas.shape)
print('results [:-1]')
print(torch.stack(results[:-1]))
print('results [-1]')
print(results[-1])

print('label len')
print(len(labels))
windows_labels=[]
time = 0
#range(len(labels)-window_size) = 449907번 반복
for i in range(len(labels)-window_size): #windowsize=12 , len(labels)=449919
    windows_labels.append(list(np.int_(labels[i:i+window_size])))
    time = time + 1

print('window labels') #(449907, 12)
# print(windows_labels)
data = numpy.array(windows_labels)
print(data.shape)
# print('window_labels') #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] window size = 12
# print(windows_labels[0],windows_labels[1],windows_labels[2])
#winow안의 각 배열값의 합이 1이상이면 anomaly
y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]
# concatenate--> Numpy 배열들을 하나로 합치는데 이용
y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])
print('y_test shape')
data2 = numpy.array(y_test)
print(data2.shape)
print('y_pred shape')
data3 = numpy.array(y_pred)
print(data3.shape)
print(y_test[0],y_test[1],y_test[2]) #0 0 0
print(len(y_test)) #449907
print('time')
print(time)

print('y_pred')
print(y_pred)#449907
print(len(y_test)) #449907
# threshold=ROC(y_test,y_pred)
# print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))

score = accuracy_score(y_test, y_pred.round(), normalize=True)
# score2 = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(score)
# print(score2)
print('y_test - y_pred')
print(y_test - y_pred)
df = pd.DataFrame(y_test - y_pred,columns=['a'])
df.to_csv('swat_residual5.csv', index=False)

df2 = pd.DataFrame(y_test,columns=['b'])
df2.to_csv('swat_label2.csv', index=False)

# plt.plot(y_pred)
# plt.ylabel('y-label')
# plt.show()
#
# plt.plot(y_test - y_pred)
# plt.ylabel('y-label')
# plt.show()
#

predicted = np.array(y_pred)
actual = np.array(y_test)

tp = np.count_nonzero(predicted * actual)
tn = np.count_nonzero((predicted - 1) * (actual - 1))
fp = np.count_nonzero(predicted * (actual - 1))
fn = np.count_nonzero((predicted - 1) * actual)

print('True Positive\t', tp)
print('True Negative\t', tn)
print('False Positive\t', fp)
print('False Negative\t', fn)

accuracy = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fmeasure = (2 * precision * recall) / (precision + recall)
# cohen_kappa_score = cohen_kappa_score(predicted, actual)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted)
auc_val = auc(false_positive_rate, true_positive_rate)
roc_auc_val = roc_auc_score(actual, predicted)
f1 = f1_score(actual,predicted.round())

print('Accuracy\t', accuracy)
print('Precision\t', precision)
print('Recall\t', recall)
print('f-measure\t', fmeasure)
# print('cohen_kappa_score\t', cohen_kappa_score)
print('auc\t', auc_val)
print('roc_auc\t', roc_auc_val)
print('f1 score\t', f1)
