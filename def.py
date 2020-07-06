import numpy as np
import pandas as pd
import pydot
from pandas.io.parsers import read_csv
from collections import Counter

from p2 import regresionLogistica, prob

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy import linalg

import seaborn as sns 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.optimizers import Adam

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek
from imblearn.keras import BalancedBatchGenerator


def get_csv(file_name):
    datos = read_csv(file_name).values
    datos = datos.astype(float)
    return datos

def get_X_dataset(datos):
    X = datos[:,1:-2]
    y = datos[:,-1]
    return X, y

def paint_classes(data): 
    n_class = pd.value_counts(data[:,-1], sort=True) 
    n_c = np.c_[np.unique(data[:,-1], return_counts=1)]
    
    frequencies = np.array(n_class)
    freq_series = pd.Series.from_array(frequencies)

    x_labels = ['Legímita (0.0)', 'Fraudulenta (1.0)']

    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title('Transacciones')
    ax.set_xlabel('Clases')
    ax.set_ylabel('Frequencia')
    ax.set_xticklabels(x_labels, rotation=0)


    def add_value_labels(ax, spacing=5):
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = spacing
            va = 'bottom'
            label = "{:.1f}".format(y_value)

            ax.annotate(
                label,                      
                (x_value, y_value),        
                xytext=(0, space),         
                textcoords="offset points", 
                ha='center',                
                va=va)                      
    add_value_labels(ax)

    plt.savefig("det_class.png")
    
def mostrarResultados(y_test,y_pred, epochs, nn_layers): 
    LABELS = ['Normal = 0.0', 'Fraud = 1.0']
    confusion_matri = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matri, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d', cmap='YlGnBu', alpha=0.5, vmin=0)
    plt.title("Matrix de Confusion")
    plt.ylabel('Verdaderos clase')
    plt.xlabel('Clase Predicción')
    plt.savefig('Figure_matrix_'+epochs+'_epochs_'+nn_layers+'.png')
    plt.show()
    print(classification_report(y_test,y_pred))

def mostrarResultados_imblearn(y_test,y_pred): 
    LABELS = ['Normal = 0.0', 'Fraud = 1.0']
    confusion_matri = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matri, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d', cmap='YlGnBu', alpha=0.5, vmin=0)
    plt.title("Matrix de Confusion")
    plt.ylabel('Verdaderos clase')
    plt.xlabel('Clase Predicción')
    plt.show()
    print(classification_report(y_test,y_pred))

def model_reg_log(X_train,y_train,X_test, y_test):
    theta = regresionLogistica(X_train,y_train)
    result = prob(theta,X_test)
    y_pred = np.where(result >= 0.5,1.,0.)
    mostrarResultados_imblearn(y_test,y_pred)
    return None

def model_subSampling(X_train,y_train, X_test, y_test):
    nm = NearMiss(version=1)
    print('NearMiss - clase minoritaria = 1 (antes) {}'.format(Counter(y_train)))
    X_train_res, y_train_res = nm.fit_sample(X_train, y_train)
    print('NearMiss - clase minoritaria = 1 {}'.format(Counter(y_train_res)))
    model_reg_log(X_train_res, y_train_res, X_test, y_test)
    return None

def model_oveSampling(X_train,y_train, X_test, y_test):
    ros = RandomOverSampler()
    print('RandomOverSampler clase mayoritaria (antes) = 0 {}'.format(Counter(y_train)))
    X_train_res, y_train_res = ros.fit_sample(X_train, y_train)
    print('RandomOverSampler clase mayoritaria = 0 {}'.format(Counter(y_train_res)))
    model_reg_log(X_train_res, y_train_res, X_test, y_test)
    return None

def model_smote(X_train,y_train, X_test, y_test):
    sm = SMOTE(random_state=27,sampling_strategy=1.0)
    print('SMOTE clase mayoritaria (antes) = 0{}'.format(Counter(y_train)))
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    print('SMOTE clase mayoritaria (despues) = 0 {}'.format(Counter(y_train_res)))
    model_reg_log(X_train_res, y_train_res, X_test, y_test)

def model_over_sub_Sampling(X_train,y_train,  X_test, y_test):
    smt = SMOTETomek()
    print('SMOTETomek {}'.format(Counter(y_train)))


    X_train_res, y_train_res = smt.fit_sample(X_train, y_train)
    print('SMOTETomek {}'.format(Counter(y_train_res)))
    #model_reg_log(X_train_res, y_train_res, X_test, y_test)
    nn_modelo(X_train_res,y_train_res, X_test, y_test)
    return None

def obtiene_class_weights(y_train, smooth_factor):
    """
    Devuelve los pesos de cada clase en función de las frecuencias de las muestras.
    : param smooth_factor: factor que suaviza los pesos extremadamente desiguales
    : param y: lista de etiquetas verdaderas (las etiquetas deben ser hashable)
    : return: diccionario con el peso de cada clase factor suave
    """
    counter = Counter(y_train)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}

def compute_class_weights(y_train):
    #class_weight = {0.0: 1,1.0: 512.87887}
    
    class_weight_list = compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weight = dict(zip(np.unique(y_train), class_weight_list))

    return class_weight



def plot_loss(history, epochs, nn_layers):
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('Figure_loss_'+epochs+'_epochs_'+nn_layers+'.png')
    plt.show()

def plot_accu(history, epochs, nn_layers):
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig('Figure_acc_'+epochs+'_epochs_'+nn_layers+'.png')
    plt.show()

def plot_accu_recall(error_df, epochs, nn_layers):
    precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, 'b', label='Precision-Recall curve')
    plt.title('Recall vs Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('Figure_accu_recall_'+epochs+'_epochs_'+nn_layers+'.png')
    plt.show()

def plot_tp_fn(error_df,  epochs, nn_layers):
    fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Figure_tp_fn_'+epochs+'_epochs_'+nn_layers+'.png')
    plt.show()




def nn_model(X_train):    
    model = Sequential()
    model.add(Dense(units=14, input_shape=(X_train.shape[1],), activation='tanh'))
    model.add(Dense(units=7, activation='tanh'))
    model.add(Dense(units=3, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def nn_model_train_pred(model, X_train, y_train, X_test, y_test, class_weight):
    epochs = 25
    nn_layers = "14_7_3_1_nor_class_smooth_factor_1"

    #NÚMERO DE CAPAS | Precision | Recall | Epochs

    #32-18-8-1 = 0.87      0.83 // 
    #32-18-8-1 = 0.72      0.85 // 73
    #32-18-8-1 = 0.89      0.81 // 86 
    #32_18_8_1 = 0.84      0.83 // 73 No normalizada
    #32_18_8_1 = 0.75      0.84 // 73 Normalizada
    
    #32_16_1 = 0.84      0.83 // 100 No normalizada
    #32_16_1 = 0.66      0.86 // Normalizada
    
    #30–7–7–1= 0.62      0.86 //
    #30-7-7-1= 0.71      0.85 //
    #30-7-7-1= 0.75      0.84 // 88 
    #30_7_7_1 = 0.84      0.83 // 100 -------------- Sin normalizar
    #30_7_7_1 = 0.76      0.84 // 100 -------------- Normalizado
    
    
    #30–14-7–7–1= 0.84      0.83 // 50 epochs
    #30–14-7–7–1= 0.91      0.68  // 100 epochs
    #30-14-7-7-1= 0.72      0.85 // 75 epochs
    #30-14-7-7-1= 0.88      0.83 // 70 epochs
    #30-14-7-7-1= 0.75      0.84 // 63
    #30-14-7-7-1= 0.77      0.84 // 56
    #30-14-7-7-1= 0.67      0.85 // 43
    #30-14-7-7-1= 0.89      0.83 // 40
    

    #------------------------------------------------

    #14-7-3-1 =  0.65      0.83
    #14-7-3-1 =  0.59      0.86 // 100
    #14-7-3-1 =  0.24      0.88 // 96
    #14-7-3-1 =  0.72      0.86 // 59 -----------
    #14-7-3-1 =  0.84      0.83 // 60 ----------- Sin normalizar
    #14-7-3-1 =  0.71      0.86 // 60 ----------- Normalizada

    #14-7-7-1 =  0.74      0.82 // 
    
    
    #30-7-7-1 =  0.73      0.86 
    #30-7-7-1 =  0.79      0.83 //40
    #30-7-7-1 =  0.78      0.83 //46
    #30-7-7-1 =  0.77      0.85 //55
    #30-7-7-1 =  0.82      0.81 //50 ------------ Nor
    #30-7-7-1 =  0.77      0.84 //100

    
    #32-16-1 = 0.55      0.86 // 55
    #32-16-1 = 0.34      0.86 // 75
    #32-16-1 = 0.75      0.85 // 78
    #32-16-1 = 0.72      0.85 // 100 ------------

    print(class_weight)

    #55
    #plot_model(model)          
                                                
    history = model.fit(X_train, y_train, batch_size=32, verbose=1,  epochs=epochs, validation_data=(X_test, y_test), class_weight=class_weight) #
    
    print(history.history.keys())
    y_pred = model.predict_classes(X_test)
    
    epochs = str(epochs)

    mse = np.mean(np.power(X_test - y_pred, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
    print(error_df.describe())
    plot_accu_recall(error_df, epochs, nn_layers)
    plot_tp_fn(error_df, epochs, nn_layers)
    plot_loss(history, epochs, nn_layers)
    plot_accu(history, epochs, nn_layers)
    mostrarResultados(y_test,y_pred, epochs, nn_layers)

    return None



def main():
    path = 'creditcard.csv'
    datos = get_csv(path)
        
    #paint_classes(datos)

    X,y = get_X_dataset(datos)

    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.20, random_state=32, stratify=y)

    ##Permite Normalizar los datos y llevar un preprocesamiento
    
    sc = MinMaxScaler(feature_range=(-1, 1))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #entrena el modelo regresion logistica 
    #model_reg_log(X_train,y_train, X_test, y_test)

    ##Usando los algoritmos de muestreo con imblearn

    #model_subSampling(X_train,y_train, X_test, y_test)
    #model_oveSampling(X_train,y_train, X_test, y_test)
    #model_smote(X_train,y_train, X_test, y_test)
    #model_over_sub_Sampling(X_train,y_train, X_test, y_test)


    

    #RED NEURONAL 
    #Se obtiene el Modelo
    model = nn_model(X_train)
    
    #Se obtiene el class_weight (balanceo de clases) con la función "obtiene_class_weights" (manual) o con "compute_class_weight" usando Scikit-learn
    
    clas_weight_ = obtiene_class_weights(y_train, 1)
    print(f'clas_weight manual :{clas_weight_}')
    clas_weight = compute_class_weights(y_train)
    print(f'clas_weight Scikit-learn :{clas_weight}')


    #Entrena y predice el modelo
    nn_model_train_pred(model, X_train,y_train, X_test, y_test, clas_weight_)


    #print(f'X_train :{X_train.shape}')
    #print(f'X_train :{y_train.shape}')
    #print(f'X_train :{X_test.shape}')
    #print(f'X_train :{y_test.shape}')


main()

