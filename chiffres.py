# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:26:35 2020

@author: Théophile Chancrin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.stats import multivariate_normal as mv_norm
from scipy.stats import norm

mnist = fetch_openml('mnist_784', version=1)
# Un dictionnaire qui contient notamment 70000 images de chiffres arabes manuscrits vectorisés (dans mnist['data']), ainsi que pour chaque image leur classe, dans mnist['target'] (sous forme d'une liste longue de 70000 chaînes de caractères). Ces données ne sont pas ordonnées.
# print(mnist['data'][0])
# print(mnist['target'][0])

train_data, test_data = mnist['data'][0:60000], mnist['data'][60000:70000]
train_target, test_target = mnist['target'][0:60000].astype(np.uint8), mnist['target'][60000:70000].astype(np.uint8)


######## Cas simple : classification binaire ########
# sept ou pas sept ? Créons un nouveau label :

train_7target = train_target == 7
test_7target = test_target == 7

train_7data = train_data[np.where(train_7target == True)]
train_n7data = train_data[np.where(train_7target == False)]


### Classifieur naïf de Bayes
p_7 = np.unique(train_7target, return_counts = True)[1][1] / len(train_7target) # probabilité de tomber sur 7
p_n7 = np.unique(train_7target, return_counts = True)[1][0] / len(train_7target) # probabilité de ne pas tomber sur 7

# Les variables sont numériques (les coefficients dans data). Utilisons donc la noi normale pour modéliser la distribution des données. 
# Les variables auraient été catégoriques (avec plus de deux catégories), nous aurions pu utiliser la loi multinomiale, elles auraient été binaire nous aurions pu utiliser la loi binomiale.

def dist(x) :
    mu = np.mean(x, axis = 0)
    sigma = np.eye(x.shape[1]) * np.std(x, axis = 0)
    dist = []
    for i in range(784) :
        dist.append(norm(mu[i], sigma[i, i]))
    return dist

dist_7 = dist(train_7data)
dist_n7 = dist(train_n7data)

def prob(x, prior, dist) :
    p = prior
    for i in range(784) :
        p *= dist[i].pdf(x[i])
    return p

print(prob(train_7data[0], p_7, dist_7))