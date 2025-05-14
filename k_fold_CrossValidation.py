import pickle
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Softmax, BatchNormalization
import keras
from sklearn.model_selection import KFold

# load data form pickle files
def load_pickle(filePath, classes):
    data=[]
    # iterate over files in
    # that directory
    for filename in os.listdir(filePath):
        group=classes+'.pickle'
        if filename.endswith(str(group)):
            path=os.path.join(filePath, filename)
            y = pickle.load(open(path,"rb"))
            x=np.array(y)
            size = x.shape
            for image in range(size[0]):
                im = [x[image].astype(float)]
                im = np.array(im)
                im = im.squeeze()
                data.append(im)  
    #Convert to np format
    data = np.array(data)
    #Return the data: an array
    return data




dir_base='D:/MDA machine learning/dataset/cross_validation'
# load data from pickle files
train_img=load_pickle(dir_base,'train_img_FL')
train_label=load_pickle(dir_base,'train_label')
test_img=load_pickle(dir_base,'test_img_FL')
test_label=load_pickle(dir_base,'test_label')
# convert loaded data into tensor: 
img_size=242
channels=1
X_train=train_img.reshape(-1, img_size,img_size,1)
Y_test=test_img.reshape(-1, img_size,img_size,1)


##cross validation------------------------------------------------------
cv = KFold(n_splits=6, shuffle=True, random_state=42)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold_train = [] #save accuracy from each fold
acc_per_fold_val = [] #save accuracy from each fold
acc_per_fold_test = [] #save accuracy from each fold

for train, test in cv.split(X_train, train_label):

    print('   ')
    print(f'Training for fold {fold_no} ...')

    ##Define the model - inside the loop so it trains from scratch for each fold    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu",input_shape=(img_size,img_size,channels)),
        BatchNormalization(),
        tf.keras.layers.MaxPool2D(2),
        #Dropout(rate=0.2),

        tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",activity_regularizer=keras.regularizers.L2(1e-4)),
        tf.keras.layers.MaxPool2D(2),

        tf.keras.layers.Conv2D(filters=16,kernel_size=3,activation="relu",activity_regularizer=keras.regularizers.L2(1e-4)),
        tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation="relu",activity_regularizer=keras.regularizers.L2(1e-4)),
        tf.keras.layers.MaxPool2D(2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(50,activation='relu',activity_regularizer=keras.regularizers.L2(1e-4)),
        Dropout(rate=0.2),

        tf.keras.layers.Dense(50,activation='relu',activity_regularizer=keras.regularizers.L2(1e-4)),
        Dropout(rate=0.2),

        tf.keras.layers.Dense(1, activation="sigmoid")
    
    ])

    # Compile our CNN
    model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),metrics=["accuracy"])

    # Fit the model
    history_2=model.fit(X_train[train],train_label[train],batch_size=32, epochs =25, verbose=0)

    #Save model trained on each fold.
    #model_2.save('Coross_validation/model_fold_'+str(fold_no))+'V1'

    # Evaluate the model - report accuracy and capture it into a list for future reporting
    scores_val = model.evaluate(X_train[test],train_label[test], verbose=0)
    acc_per_fold_val.append(scores_val[1] * 100)

    scores_test = model.evaluate(Y_test,test_label, verbose=0)
    acc_per_fold_test.append(scores_test[1] * 100)

    scores_train = model.evaluate(
        [train],train_label[train], verbose=0)
    acc_per_fold_train.append(scores_train[1] * 100)

    fold_no = fold_no + 1

# visualize the results
for acc in acc_per_fold_test:
    print("test accuracy for this fold is: ", acc)
for acc in acc_per_fold_val:
    print("validation accuracy for this fold is: ", acc)
for acc in acc_per_fold_train:
    print("train accuracy for this fold is: ", acc)
