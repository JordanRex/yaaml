from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import pickle

def data(train, valid, ytrain, yvalid):
    
    X = np.array(train)
    XV = np.array(valid)
    Y = to_categorical(ytrain)
    YV = to_categorical(yvalid)
    
    # save backup
    nnbp = open('./backup.pkl','wb')
    pickle.dump(X, nnbp)
    pickle.dump(Y, nnbp)
    pickle.dump(XV, nnbp)
    pickle.dump(YV, nnbp)
    nnbp.close()
    
    # load backup
    nnbp = open('./backup.pkl', 'rb')
    X = pickle.load(nnbp)
    Y = pickle.load(nnbp)
    XV = pickle.load(nnbp)
    YV = pickle.load(nnbp)
    nnbp.close()
    return X, Y, XV, YV

def create_model(X, Y, XV, YV):
    '''
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    input_dim = X.shape[1]

    model = Sequential()
    model.add(Dense(input_dim, input_dim = input_dim , activation={{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    model.add(BatchNormalization())
    model.add(Dense({{choice([50, 100, 250, 500, 1000, 2000])}}, activation={{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    model.add(Dropout({{uniform(0, 0.7)}}))
    model.add(Dense({{choice([50, 100, 250, 500, 1000])}}, activation = {{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    if {{choice(['true', 'false'])}} == 'true':
        model.add(Dense({{choice([5, 20, 30, 50, 100])}}, activation = {{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    if {{choice(['true', 'false'])}} == 'true':
        model.add(Dense({{choice([5, 20, 30, 50, 100])}}, activation = {{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    model.add(Dropout({{uniform(0, 0.7)}}))
    model.add(Dense(9, activation={{choice(['softmax', 'sigmoid'])}}))

    model.compile(loss='categorical_crossentropy', optimizer = {{choice(['rmsprop', 'adam', 'sgd', 'nadam', 'adadelta'])}}, 
                  metrics=['accuracy'])
    model.fit(X, Y, batch_size={{choice([10, 20])}}, epochs=5, verbose=2, validation_data=(XV, YV), shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', patience=7, min_delta=0.0001)])
    score, acc = model.evaluate(XV, YV, verbose=1)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model, space = optim.minimize(model=create_model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=5,
                                                 trials=Trials(),
                                                 eval_space=True,
                                                 return_space=True)
    X, Y, XV, YV = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(XV, YV))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('model.h5')

    
# model = load_model('model.h5')
# print(best_run)
# model.fit(X,Y,batch_size=10, epochs=10, verbose=1, shuffle=True, validation_data=(XV,YV))
# model.evaluate(XV,YV)
# nn_pred = model.predict_classes(x=XV)
# skm.accuracy_score(y_pred=nn_pred, y_true=yvalid)
# skm.confusion_matrix(y_pred=nn_pred, y_true=yvalid)
