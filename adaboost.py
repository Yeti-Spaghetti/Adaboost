import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
%alias_magic t timeit

# M = 1, B = -1
def calc_errors(xt, yt, i, split, w):
    truthArr_temp = np.ones(300, dtype=bool)
    count = 0
    
    # if incorrectly classified
    for j in range(300):
        if (xt[i][j] < split and yt[j] == 1) or (xt[i][j] >= split and yt[j] == -1):
            truthArr_temp[j] = False;
            count += 1
            
    # if number of errors > 50%, invert
    invert = False
    if (count/300) > 0.5:
        count = 300 - count
        invert = True
        truthArr_temp = np.invert(truthArr_temp)
    
    #calculate total error
    total_error = 0.0
    for i in range(300):
        if truthArr_temp[i] == False:
            total_error += w[i]

    return total_error, truthArr_temp, invert
            
def find_best_stump(xt, yt, w):
    minWErr = math.inf
    stump = {}
    
    for k in range(30):
        i = random.randint(0,29)
        j = random.randint(0,299)
        temp_split = xt[i][j]
        tErr, truthArr_temp, invert = calc_errors(xt, yt, i, temp_split, w)

        # if find new stump with less errors update values
        if tErr < minWErr:
            truthArr = truthArr_temp
            minWErr = tErr
            stump['index'] = i
            stump['split'] = temp_split
            stump['invert'] = invert
                
    return stump, minWErr, truthArr
        
def calc_alpha(learn, err):
    return learn * np.log((1-err)/err)

def new_weights(alpha, truthArr, w):
    new_w = np.zeros(300)
    for i in range(300):
        if (truthArr[i] == True):
            new_w[i] = w[i] * math.exp(-alpha)
        else:
            new_w[i] = w[i] * math.exp(alpha)
            
    return new_w

def normalize(w, w_temp):
    sum_w = 0
    for i in range(300):
        sum_w += w_temp[i]
        
    for i in range(300):
        w[i] = w_temp[i] / sum_w
        
    return w

def testing(xt, yt, weakClsfy):
    correct = 0
    for j in range(269):
        prediction = 0
        for i in range(len(weakClsfy)):
            dim = weakClsfy[i]['index']
            alpha = weakClsfy[i]['alpha']
            split = weakClsfy[i]['split']
            if weakClsfy[i]['invert'] == False:
                if (xt[dim][j] < split):
                    prediction += -1 * alpha
                if (xt[dim][j] >= split):
                    prediction += 1 * alpha
                    
        prediction = np.sign(prediction)
        if prediction == yt[j]:
            correct += 1

    print('accuracy: ', (correct/269) * 100, '%')
    return correct/269
    
with open('data/wdbc_data.csv', newline='') as csvfile:
    temp = list(csv.reader(csvfile))
    X = np.array(temp)

Y = X[:,1]
# M = 1, B = -1
Y = list(map(lambda x:1 if x=="M" else -1, Y))
X = np.delete(X,0,1)
X = np.delete(X,0,1)
X = np.asfarray(X,float)

# [column = dimension, row = samples]
x_train = X[0:300].T
x_test = X[300:569].T
y_train = Y[0:300]
y_test = Y[300:569]

weights = np.ones(300)
for i in range(0,300):
    weights[i] = weights[i]/300
    
# hyperparameters
iterations = 200
learn_rate = 0.5

weakClassifiers = []
xAxis_iter = []
yAxis_acc = []

# training

for i in range(iterations):
    best_stump, wErr, truthArr = find_best_stump(x_train,y_train, weights)
#     print("iteration: ", i+1, ", best stump index: ", best_stump['index'], ", weighted error: ", wErr)
    alpha = calc_alpha(learn_rate, wErr)
#     print('alpha: ', alpha)
    best_stump['alpha'] = alpha
    weakClassifiers.append(best_stump)
    w_temp = new_weights(alpha, truthArr, weights)
    weights = normalize(weights, w_temp)
    
    print('iteration: ', i+1)
    xAxis_iter.append(i+1)
    yAxis_acc.append(testing(x_test, y_test, weakClassifiers))
    
print('iterations complete')

plt.plot(xAxis_iter, yAxis_acc)
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')
plt.show()
