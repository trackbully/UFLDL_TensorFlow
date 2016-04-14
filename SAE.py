import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
import math as math
__author__ = 'JEFFERYK','BOBW'

NUMSAMPLES = 10000
PATCHWIDTH = 8
N_HIDDEN = 25
N_INPUT = PATCHWIDTH**2
N_OUTPUT = N_INPUT
BETA = tf.constant(3.)
LAMBDA = tf.constant(.0001)
RHO = tf.constant(0.01)
EPSILON = .00001

sess = tf.InteractiveSession()


def normalize(unNormalizedTensor):
    normalized = unNormalizedTensor - np.mean(unNormalizedTensor)
    pstd = 3 * np.std(normalized)
    normalized = np.maximum(np.minimum(normalized, pstd),-1*pstd)/pstd
    normalized = (normalized +1) *.4 + 0.1
    return normalized

def displayWeights(weights, blocking=False):
    num_tiles = math.sqrt(N_HIDDEN)
    image = np.zeros((num_tiles*PATCHWIDTH + num_tiles+1,num_tiles*PATCHWIDTH + num_tiles+1))
    for i in range(N_HIDDEN):
        subWeights = normalize(weights[:,i])
        subWeights = np.reshape(subWeights,(PATCHWIDTH,PATCHWIDTH))
        denom = np.sqrt(np.dot(subWeights,subWeights))
        subWeights = np.divide(subWeights,denom)
        xIndex = i % num_tiles
        yIndex = i // num_tiles
        xStart = (xIndex+1)+(xIndex*PATCHWIDTH)
        yStart = (yIndex+1)+(yIndex*PATCHWIDTH) 
        image[xStart:xStart+PATCHWIDTH,yStart:yStart+PATCHWIDTH] = subWeights
        
    plt.figure(1)
    plt.imshow(image,interpolation='none')      
    plt.draw()
    plt.show(block=blocking)
    
def displayImageTest(orig, blocking=False):
    num_s = (np.size(orig,axis=1)//PATCHWIDTH)**2
    patches = np.zeros((num_s,PATCHWIDTH**2))
    tile_size = 512//PATCHWIDTH
    for r in range(tile_size):
        for c in range(tile_size):
            index = r*tile_size+c
            patches[index] = orig[r*PATCHWIDTH:r*PATCHWIDTH+PATCHWIDTH,c*PATCHWIDTH:c*PATCHWIDTH+PATCHWIDTH].flatten()
    
    result = normalize(sess.run(pred['out'], feed_dict={x: patches}))
    
    repro = np.zeros((512,512))
    for r in range(tile_size):
        for c in range(tile_size):
            repro[r*PATCHWIDTH:r*PATCHWIDTH+PATCHWIDTH,c*PATCHWIDTH:c*PATCHWIDTH+PATCHWIDTH] = np.reshape(result[r*tile_size+c],(PATCHWIDTH,PATCHWIDTH))
    
    
    image=np.concatenate((orig,repro),axis=1)
    plt.figure(2)        
    plt.imshow(image,interpolation='none')      
    plt.draw()
    plt.show(block=blocking)


# LOAD image data
images = np.array(scipy.io.loadmat("IMAGES.mat")["IMAGES"])

#normalize the data
for i in range(np.size(images,2)):
    images[:,:,i] = normalize(images[:,:,i])

samples = np.zeros((NUMSAMPLES,N_INPUT))
for i in range(NUMSAMPLES):
    #pick a random image
    image = images[:,:,np.random.randint(np.size(images,2))]
    x = np.random.randint(np.size(images,0)-PATCHWIDTH)
    y = np.random.randint(np.size(images,1)-PATCHWIDTH)
    subImage = image[x:x+PATCHWIDTH,y:y+PATCHWIDTH]
    samples[i] = subImage.flatten()

#Construct the tensor flow model
x = tf.placeholder("float", [None, N_INPUT])
hidden = tf.placeholder("float", [None, N_HIDDEN])

def autoencoder(X, weights, biases):
    hiddenlayer = tf.sigmoid(
        tf.add(
            tf.matmul(
                X, weights['hidden']
            ),
            biases['hidden']
        )
    )
    out = tf.sigmoid(
        tf.add(
            tf.matmul(
                hiddenlayer, weights['out']
            ), 
            biases['out']
        )
    )
    return {'out': out, 'hidden': hiddenlayer}

weights = {
    'hidden': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN])),
    'out': tf.Variable(tf.random_normal([N_HIDDEN, N_OUTPUT]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([N_HIDDEN])),
    'out': tf.Variable(tf.random_normal([N_OUTPUT]))
}

pred = autoencoder(x, weights, biases)
rho_hat = tf.div(tf.reduce_sum(pred['hidden'],0),tf.constant(float(NUMSAMPLES)))

#Construct cost
def KL_Div(rho, rho_hat):
    invrho = tf.sub(tf.constant(1.), rho)
    invrhohat = tf.sub(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho,rho_hat), logfunc(invrho, invrhohat))
    return logrho

def logfunc(x, x2):
    return tf.mul( x, tf.log(tf.div(x,x2)))

diff = tf.sub(pred['out'], x)
cost_J = tf.div(tf.nn.l2_loss(diff),tf.constant(float(NUMSAMPLES)))
cost_sparse = tf.mul(BETA,  tf.reduce_sum(KL_Div(RHO, rho_hat)))
cost_reg = tf.mul(LAMBDA , tf.add(tf.nn.l2_loss(weights['hidden']), tf.nn.l2_loss(weights['out'])))
cost = tf.add(tf.add(cost_J , cost_reg ), cost_sparse)

optimizer = tf.train.AdamOptimizer().minimize(cost)

_ = tf.histogram_summary('cost', cost)
writer = tf.train.SummaryWriter('/tmp/sae_logs', sess.graph_def)
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess.run(init)
plt.gray()
vizImage = np.random.randint(np.size(images,2))


# Training cycle
c = 0.
c_old = 1.
i = 0
while np.abs(c - c_old) > EPSILON :
    sess.run(optimizer, feed_dict={x: samples})
    if i % 1000 == 0:
        c_old = c
        c,j,reg,sparse = sess.run([cost,cost_J,cost_reg,cost_sparse], feed_dict={x: samples})
        print "EPOCH %d: COST = %f, LOSS = %f, REG_PENALTY = %f, SPARSITY_PENTALTY = %f" %(i,c,j,reg,sparse)
        displayWeights(sess.run(weights['hidden']))
        displayImageTest(images[:,:,vizImage])
    i += 1
print("Optimization Finished!")
   
        
