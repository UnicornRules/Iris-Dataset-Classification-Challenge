# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:44:04 2020

@author: Justin
"""
import operator
import csv
import numpy as np
from collections import Counter

"""charger le dataset et créer une liste de données numériques et une liste de classes"""

x, y= [], []
z, w = [], []
with open("data.csv", "r") as f:
    csvreader = csv.reader(f)  
    for row in csvreader:
        row = row[0].split(';')
        x.append(row[:-1])
        y.append(row[-1])


with open("finalTest.csv", "r") as f1:
    csvreader = csv.reader(f1)  
    for row in csvreader:
        row = row[0].split(';')
        z.append(row)


       
"""fonction qui calcule la distance entre deux données x1 et x2"""          
def distance(x1, x2):
     a1 = 10000*(float(x2[0])-float(x1[0]))**2
     b1 = 10000*(float(x2[1])-float(x1[1]))**2
     c1 = 10000*(float(x2[2])-float(x1[2]))**2
     d1 = 10000*(float(x2[3])-float(x1[3]))**2
     a = np.array(x1,float)
     b= np.array(x2,float)
     dist = (a@b.T)/(np.linalg.norm(a)*np.linalg.norm(b))
     dist1= np.sqrt(a1+b1+c1+d1)
     
     return(dist/dist1)
 
def order(k, dt):
    dist = []
    classes = []
    for i in range(len(x)):
        dist.append(distance(x[i],dt))
        classes.append(y[i])

    top = zip(dist,classes)
    top = list(top)
    top2 = sorted(top, key = operator.itemgetter(0),reverse=True)
    #print(top2)
    top3=[]
    for j in range(k):
        top3.append(top2[j][1])
    #print(top3)
    plusFrequent = Counter(top3).most_common()
    return(plusFrequent[0][0])

def knn(k,liste):
    result =[]
    with open("test3.txt","a") as fichier :
        for i in range(len(liste)):
            fichier.write(order(k,liste[i]))
            fichier.write("\n")
            result.append(order(k,liste[i]))
    return(result)

def confusion(knn,dico_eval): 
    
    a = 0
    b = 0
      
    confusion = np.zeros((10,10))

    for i in range(len(knn)):
        if knn[i] == 'A':
            a = 0
        elif knn[i] == 'B':
            a = 1
        elif knn[i] == 'C':
            a = 2
        elif knn[i] == 'D':
            a = 3
        elif knn[i] == 'E':
            a = 4
        elif knn[i] == 'F':
            a = 5
        elif knn[i] == 'G':
            a = 6
        elif knn[i] == 'H':
            a = 7
        elif knn[i] == 'I':
            a = 8
        elif knn[i] == 'J':
            a = 9

        if dico_eval[i] == 'A':
            b = 0
        elif dico_eval[i] == 'B':
            b = 1
        elif dico_eval[i] == 'C':
            b = 2
        elif dico_eval[i] == 'D':
            b = 3
        elif dico_eval[i] == 'E':
            b = 4
        elif dico_eval[i] == 'F':
            b = 5
        elif dico_eval[i] == 'G':
            b = 6
        elif dico_eval[i] == 'H':
            b = 7
        elif dico_eval[i] == 'I':
            b = 8
        elif dico_eval[i] == 'J':
            b = 9

        confusion[a][b] += 1
    taux= np.trace(confusion)/len(knn)
    print(taux)            
    print('Total éléments :',len(knn))
    print()
    print('Classe prédite|      Classe réelle')
    print('               A | B | C | D | E | F | G | H | I | J')
    print('A        |  ',confusion[0][0],'  |     ',confusion[0][1],'    |    ',confusion[0][2],'    |    ',confusion[0][3],'    |    ',confusion[0][4],'    |    ',confusion[0][5],'    |    ',confusion[0][6],'    |    ',confusion[0][7],'    |    ',confusion[0][8],'    |    ',confusion[0][9])
    print('B        |  ',confusion[1][0],'  |     ',confusion[1][1],'    |    ',confusion[1][2],'    |    ',confusion[1][3],'    |    ',confusion[1][4],'    |    ',confusion[1][5],'    |    ',confusion[1][6],'    |    ',confusion[1][7],'    |    ',confusion[1][8],'    |    ',confusion[1][9])
    print('C        |  ',confusion[2][0],'  |     ',confusion[2][1],'    |    ',confusion[2][2],'    |    ',confusion[2][3],'    |    ',confusion[2][4],'    |    ',confusion[2][5],'    |    ',confusion[2][6],'    |    ',confusion[2][7],'    |    ',confusion[2][8],'    |    ',confusion[2][9])
    print('D        |  ',confusion[3][0],'  |     ',confusion[3][1],'    |    ',confusion[3][2],'    |    ',confusion[3][3],'    |    ',confusion[3][4],'    |    ',confusion[3][5],'    |    ',confusion[3][6],'    |    ',confusion[3][7],'    |    ',confusion[3][8],'    |    ',confusion[3][9]) 
    print('E        |  ',confusion[4][0],'  |     ',confusion[4][1],'    |    ',confusion[4][2],'    |    ',confusion[4][3],'    |    ',confusion[4][4],'    |    ',confusion[4][5],'    |    ',confusion[4][6],'    |    ',confusion[4][7],'    |    ',confusion[4][8],'    |    ',confusion[4][9]) 
    print('F        |  ',confusion[5][0],'  |     ',confusion[5][1],'    |    ',confusion[5][2],'    |    ',confusion[5][3],'    |    ',confusion[5][4],'    |    ',confusion[5][5],'    |    ',confusion[5][6],'    |    ',confusion[5][7],'    |    ',confusion[5][8],'    |    ',confusion[5][9]) 
    print('G        |  ',confusion[6][0],'  |     ',confusion[6][1],'    |    ',confusion[6][2],'    |    ',confusion[6][3],'    |    ',confusion[6][4],'    |    ',confusion[6][5],'    |    ',confusion[6][6],'    |    ',confusion[6][7],'    |    ',confusion[6][8],'    |    ',confusion[6][9]) 
    print('H        |  ',confusion[7][0],'  |     ',confusion[7][1],'    |    ',confusion[7][2],'    |    ',confusion[7][3],'    |    ',confusion[7][4],'    |    ',confusion[7][5],'    |    ',confusion[7][6],'    |    ',confusion[7][7],'    |    ',confusion[7][8],'    |    ',confusion[7][9]) 
    print('I        |  ',confusion[8][0],'  |     ',confusion[8][1],'    |    ',confusion[8][2],'    |    ',confusion[8][3],'    |    ',confusion[8][4],'    |    ',confusion[8][5],'    |    ',confusion[8][6],'    |    ',confusion[8][7],'    |    ',confusion[8][8],'    |    ',confusion[8][9]) 
    print('J        |  ',confusion[9][0],'  |     ',confusion[9][1],'    |    ',confusion[9][2],'    |    ',confusion[9][3],'    |    ',confusion[9][4],'    |    ',confusion[9][5],'    |    ',confusion[9][6],'    |    ',confusion[9][7],'    |    ',confusion[9][8],'    |    ',confusion[9][9])
        
    







