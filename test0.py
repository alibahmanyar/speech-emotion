#!/usr/bin/env python
# coding: utf-8
import os
import librosa
from librosa import display
import numpy as np
from matplotlib import pyplot as plt
import time
import random

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data


DATASET_PATH = "samples"
samples = os.listdir(DATASET_PATH)[:200]
random.shuffle(samples)

# EmotionsDict = {'N': 'Neutral', 'F':'Fear', 'H':'Happiness', 'S':'Sadness', 'W':'Surprise', 'A':'Angry'}
EmotionsDict = {'N': 0, 'F': 1, 'H': 2, 'S': 3, 'W': 4, 'A': 5}
emotions = [EmotionsDict[x[3]] for x in samples]



sr = 44100

SIG_LENGTH = 200

mfccs = []

for s in samples:
    t1 = time.time()

    sig, sr = librosa.load(f'{DATASET_PATH}/{s}', sr=sr)
    sig, _ = librosa.effects.trim(sig) # trimming beginning and ending silence


    # MFCC config
    WinLen = int(0.040 * sr) # 40 milisecond
    WinHop = WinLen // 2
    
    sig_mfcc = librosa.feature.mfcc(y=librosa.power_to_db(sig), sr=sr, n_mfcc=12, fmax=sr//2)

    # cutting every longer sequence than SIG_LENGTH and padding any sequence shorter than SIG_LENGTH
    if sig_mfcc.shape[1] < SIG_LENGTH:
        sig_mfcc = np.pad(sig_mfcc, [(0, 0), (0, SIG_LENGTH - sig_mfcc.shape[1])], mode='symmetric')
    elif sig_mfcc.shape[1] > SIG_LENGTH:
        sig_mfcc = sig_mfcc[:, :SIG_LENGTH]

    mfccs.append(sig_mfcc)


mfccs = np.array(mfccs)
emotions = np.array(emotions)

mfccs = np.expand_dims(mfccs, axis=-1)

avg = mfccs.mean()
std = mfccs.std()

mfccs = (mfccs - avg) / std

print(mfccs.shape)
print(emotions.shape)


BATCH_SIZE = 64
LAYERS_ACTIVATION = 'relu'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'

EPOCHS = 20

def get_model():
    model = tf.keras.Sequential([
              tf.keras.layers.Conv2D(64, (3, 3), activation=LAYERS_ACTIVATION, padding='same'),
              tf.keras.layers.MaxPool2D((2, 2)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(128, activation=LAYERS_ACTIVATION),
              tf.keras.layers.Dense(64, activation=LAYERS_ACTIVATION),
              tf.keras.layers.Dense(6, activation=tf.keras.activations.softmax)               
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=LOSS_FUNCTION, metrics=['accuracy'])
    return model

model = get_model()
history = model.fit(mfccs, emotions, epochs=EPOCHS, batch_size=BATCH_SIZE)

