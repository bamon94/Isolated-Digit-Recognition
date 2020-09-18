# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:05:55 2020

@author: bhara
"""


import librosa
import numpy as np
import os as os
from os import listdir
from os.path import isfile, join
from sklearn import mixture
from sklearn import metrics
from sklearn import preprocessing
import re



def extractFeature(fileList):
    features = np.empty((39, 1))
    for file in fileList:
        x, fs = librosa.load(file,sr=16000)
        x,d = librosa.effects.trim(x,top_db=20)
        x = preprocessing.normalize(x.reshape(1,-1)).squeeze()
        mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=14,n_fft=320, hop_length=40 )[1:14]
        deltaMfcc = librosa.feature.delta(mfcc,width=3, order=1)
        deltadeltaMfcc = librosa.feature.delta(mfcc,width=3, order=2)
        feature = np.concatenate((mfcc,deltaMfcc,deltadeltaMfcc),axis = 0)
        features = np.append(features,feature,axis=1)
    return features

def trainGMM(features):
    g = mixture.GaussianMixture(n_components=16)
    g.fit(features.T)
    return g

def evaluateConfusionMatrix(allTestDataFiles,inpathTestPath,gmm):
    yTrue = np.array([])
    yPredicted = np.array([])
    for i in range(10):
        match = '.*_'+str(i)+'_.*'
        r = re.compile(match)
        testDataFiles = [join(inpathTestPath, file) for file in allTestDataFiles if r.match(file)]
        yTrue = np.append(yTrue,i*np.ones(len(testDataFiles)))
        for file in testDataFiles:
            feature = extractFeature([file]).T
            yPredicted = np.append(yPredicted,np.argmax([g.score(feature) for g in gmm])).astype('int')
    return metrics.confusion_matrix(yTrue,yPredicted)
    

inpathTrainingPath = os.getcwd()+"\\train"
inpathTestPath = os.getcwd()+"\\test"
allTrainDataFiles = [f for f in listdir(inpathTrainingPath) if isfile(join(inpathTrainingPath, f))]
allTestDataFiles = [f for f in listdir(inpathTestPath) if isfile(join(inpathTestPath, f))]
featureArray = np.array([])
index = {}
total = 0
gmm = []

for i in range(10):
    match = '.*_'+str(i)+'_.*'
    r = re.compile(match)
    trainDataFiles = [join(inpathTrainingPath, file) for file in allTrainDataFiles if r.match(file)]
    features = extractFeature(trainDataFiles)
    g = trainGMM(features)
    gmm.append(g)
confusionMatrix = evaluateConfusionMatrix(allTestDataFiles,inpathTestPath,gmm)
print("The confusion matrix is",confusionMatrix)
