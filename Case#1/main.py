from model.LSTM import LSTMModel3, LSTM_CNN_Model, GRUModel, BidirectionalLSTMModel

#For Training
model = LSTMModel3() #or model = LSTMModel3(weights=None)
model.train_model('path to data yaml')

#For Validation and predict
model = LSTMModel3(weights='path to .pt')
# For Validation
model.valid('path to data yaml')
# For predict
model.predict(text='Suka banget sama provide ini')