import torch.utils.data as data_utils
from utils import *
from usad import *
from sklearn import preprocessing



model = UsadModel(612, 1200)
model = to_device(model,device)
checkpoint = torch.load("model.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

results=testing(model,test_loader)