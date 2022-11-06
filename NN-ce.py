# %% Neural c-e
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

size=200                                    #training data
x0a=(np.random.normal(0.0,1.0,size))
x0b=(np.random.normal(0.0,1.0,size))

x11a=(np.random.normal(-1.0,1.0,size//2))
x12a=(np.random.normal(1.0,1.0,size//2))
x1a=np.concatenate((x11a,x12a),axis=0)
np.random.shuffle(x1a)
x11b=(np.random.normal(-1.0,1.0,size//2))
x12b=(np.random.normal(1.0,1.0,size//2))
x1b=np.concatenate((x11b,x12b),axis=0)
np.random.shuffle(x1b)


plott=[]
li=[]

def g_ce(x,deriv=False):        #activation function
    if deriv==True: return 1/1-x
    return 1/(1+np.exp(-x)) 

def fi(x):                      #paragwgos fi
    return 1/((1-x))

def psi(x):                      #paragwgos psi
    return -1/((x))

def Relu(x):
    k=np.zeros(shape=(20,))
    for i in range(20):
        if x[i]>0:
            k[i]=x[i]
        else:
            k[i]=0
    return k

def mult(z,B):                      #ypologismos B*ReLU'
    k=np.zeros(shape=(20,))
    for i in range(20):
        if(z[i]>0):
            k[i]=B[i]
        else:
            k[i]=0
    return np.reshape(k,[20,1])
    
def cost(x1,x2):
    return (-1)*np.log(1-x1)-np.log(x2)   
    

# deterministic weights
np.random.seed(1)

# initializing weights randomly with mean 0
A = np.random.normal(0.0,1.0/22.,(2,20))
B = np.random.normal(0.0,1.0/21.,(20,1))
a=np.zeros(shape=(20,))
b=0
m=0.0001


i=k=0
p=[]
while True:
    # forward propagation
    l0f = np.array([x0a[i],x0b[i]])
    l1f = Relu(np.dot(l0f,A)+a)
    l2f = g_ce(np.dot(l1f.T,B)+b)
    
    l0p = np.array([x1a[i],x1b[i]]).T
    l1p = Relu(np.dot(l0p,A)+a)
    l2p = g_ce(np.dot(l1p.T,B)+b)
    
    #stochastic gradient descent
    A-=m*(fi(l2f)*g_ce(np.dot(l1f.T,B)+b,deriv=True)*np.dot(mult(l1f,B),np.reshape(l0f,[1,2]))+psi(l2p)*g_ce(np.dot(l1p.T,B)+b,deriv=True)*np.dot(mult(l1p,B),np.reshape(l0p,[1,2]))).T
    a-=m*np.reshape(fi(l2f)*g_ce(np.dot(l1f.T,B)+b,deriv=True)*mult(l1f,B)+psi(l2p)*g_ce(np.dot(l1p.T,B)+b,deriv=True)*mult(l1p,B),[20,])
    B-=m*np.reshape(fi(l2f)*g_ce(np.dot(l1f.T,B)+b,deriv=True)*l1f.T+psi(l2p)*g_ce(np.dot(l1p.T,B)+b,deriv=True)*l1p.T,[1,20]).T
    b-=m*(fi(l2f)*g_ce(np.dot(l1f.T,B)+b,deriv=True)+psi(l2p)*g_ce(np.dot(l1p.T,B)+b,deriv=True))
    
    
    p.append(cost(l2f,l2p))
    
    i+=1
    if (i==size):
        plott.append(np.average(p))
        p=[]
        li.append(k)
        #print (plott[k])
        if k!=0 and plott[k]>plott[k-1]:    #minimum cost function
            break
        k+=1;i=0
    
plt.plot(li,plott)
#%%
size=int(1e6)                               #testing data
y0a=(np.random.normal(0.0,1.0,size))
y0b=(np.random.normal(0.0,1.0,size))

y11a=(np.random.normal(-1.0,1.0,size//2))
y12a=(np.random.normal(1.0,1.0,size//2))
y1a=np.concatenate((y11a,y12a),axis=0)
np.random.shuffle(y1a)
y11b=(np.random.normal(-1.0,1.0,size//2))
y12b=(np.random.normal(1.0,1.0,size//2))
y1b=np.concatenate((y11b,y12b),axis=0)
np.random.shuffle(y1b)

c1=c2=0
for i in range(size):
    l0 = np.array([y0a[i],y0b[i]])
    l1 = Relu(np.dot(l0,A)+a)
    l2 = g_ce(np.dot(l1.T,B)+b)
    if l2>0.5: c1+=1                    #H1 enw eprepe H0
    
    l0n = np.array([y1a[i],y1b[i]])
    l1n = Relu(np.dot(l0n,A)+a)
    l2n = g_ce(np.dot(l1n.T,B)+b)
    if l2n<0.5: c2+=1                   #H0 enw eprepe H1
    
print (100*(c1+c2)/(2*size),'%') 
    