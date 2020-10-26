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

def save_data(train, valid, ytrain, yvalid):
    x = np.array(train)
    X = np.array(valid)
    y = to_categorical(ytrain)
    Y = to_categorical(yvalid)
    # save backup
    nnbp = open('./backup.pkl','wb')
    pickle.dump(x, nnbp)
    pickle.dump(X, nnbp)
    pickle.dump(y, nnbp)
    pickle.dump(Y, nnbp)
    nnbp.close()
    return None

def data():
    # load backup
    nnbp = open('./backup.pkl', 'rb')
    x = pickle.load(nnbp)
    X = pickle.load(nnbp)
    y = pickle.load(nnbp)
    Y = pickle.load(nnbp)
    nnbp.close()
    return x, X, y, Y

def create_model(x, X, y, Y):
    '''
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    input_dim = x.shape[1]

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
    model.fit(x, y, batch_size={{choice([10, 20])}}, epochs=5, verbose=2, validation_data=(X, Y), shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', patience=7, min_delta=0.0001)])
    score, acc = model.evaluate(X, Y, verbose=1)
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
    x, X, y, Y = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X, Y))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('model.h5')

    best_model.fit(x,y,batch_size=10, epochs=10, verbose=1, shuffle=True, validation_data=(X,Y))
    model.evaluate(X,Y)
    nn_pred = model.predict_classes(x=X)
    skm.accuracy_score(y_pred=nn_pred, y_true=Y)
    skm.confusion_matrix(y_pred=nn_pred, y_true=Y)
