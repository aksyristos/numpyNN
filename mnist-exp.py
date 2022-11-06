import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#plt.figure()
#plt.imshow(test_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_zero_l=[]                      #apomonwnw ta '0' kai ta '8'
train_zero=[]
train_eight_l=[]
train_eight=[]
for l in range(60000):
    if train_labels[l]==0: 
        train_zero.append(train_images[l])
        train_zero_l.append(train_labels[l])
    if train_labels[l]==8:
        train_eight.append(train_images[l])
        train_eight_l.append(train_labels[l])
        
test_zero_l=[]
test_zero=[]
test_eight_l=[]
test_eight=[]
for l in range(10000):
    if test_labels[l]==0: 
        test_zero.append(test_images[l])
        test_zero_l.append(test_labels[l])
    if test_labels[l]==8:
        test_eight.append(test_images[l])
        test_eight_l.append(test_labels[l])

train_zero = np.asfarray(train_zero) / 255.0        #kanonikopoihsh
test_zero = np.asfarray(test_zero) / 255.0
train_eight = np.asfarray(train_eight) / 255.0
test_eight = np.asfarray(test_eight) / 255.0
train_zero_l = np.asarray(train_zero_l)
test_zero_l = np.asarray(test_zero_l)
train_eight_l = np.asarray(train_eight_l)
test_eight_l = np.asarray(test_eight_l)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_eight[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_eight_l[i]])
plt.show()

#%% exp
plott=[]
li=[]

def g_e(x):                     #activation function
    return x

def fi(x):                      #paragwgos fi
    return 0.5*np.exp(0.5*x)

def psi(x):                     #paragwgos psi
    return -0.5*np.exp(-0.5*x)


def Relu(x):
    k=np.zeros(shape=(300,))
    for i in range(300):
        if x[i]>0:
            k[i]=x[i]
        else:
            k[i]=0
    return k

def mult(z,B):                      #ypologismos B*ReLU'
    k=np.zeros(shape=(300,))
    for i in range(300):
        if(z[i]>0):
            k[i]=B[i]
        else:
            k[i]=0
    return np.reshape(k,[300,1])
    
def cost(x1,x2):
    return np.exp(0.5*x1)+np.exp(-0.5*x2) 
    
# deterministic weights
np.random.seed(1)

# initializing weights randomly with mean 0
A = np.random.normal(0.0,1.0/1084.,(784,300))
B = np.random.normal(0.0,1.0/301.,(300,1))
a=np.zeros(shape=(300,))
b=0
m=0.001


i=k=0
p=[]

while True:
    # forward propagation
    l0f = train_zero[i].flatten()
    l1f = Relu(np.dot(l0f,A)+a)
    l2f = g_e(np.dot(l1f.T,B)+b)
    
    l0p = train_eight[i].flatten()
    l1p = Relu(np.dot(l0p,A)+a)
    l2p = g_e(np.dot(l1p.T,B)+b)
    
    #stochastic gradient descent
    A-=m*(fi(l2f)*np.dot(mult(l1f,B),np.reshape(l0f,[1,784]))+psi(l2p)*np.dot(mult(l1p,B),np.reshape(l0p,[1,784]))).T
    a-=m*np.reshape(fi(l2f)*mult(l1f,B)+psi(l2p)*mult(l1p,B),[300,])
    B-=m*np.reshape(fi(l2f)*l1f.T+psi(l2p)*l1p.T,[300,1])
    b-=m*(fi(l2f)+psi(l2p))
    
    p.append(cost(l2f,l2p))
    
    i+=1
    if (i==len(train_eight_l)):
        plott.append(np.average(p))
        p=[]
        li.append(k)
        #print (plott[k])
        if k!=0 and plott[k]>plott[k-1]:    #minimum cost function
            break
        k+=1;i=0
plt.plot(li,plott)


#%%
c1=c2=0
for i in range(len(test_zero_l)):
    l0 = test_zero[i].flatten()
    l1 = Relu(np.dot(l0,A)+a)
    l2 = g_e(np.dot(l1.T,B)+b)
    if l2>0: c1+=1                          #H1 enw eprepe H0
    
for i in range(len(test_eight_l)):    
    l0n = test_eight[i].flatten()
    l1n = Relu(np.dot(l0n,A)+a)
    l2n = g_e(np.dot(l1n.T,B)+b)
    if l2n<0: c2+=1                         #H0 enw eprepe H1
    
print (100*(c1+c2)/(len(test_zero_l)+len(test_eight_l)),'%') 
    