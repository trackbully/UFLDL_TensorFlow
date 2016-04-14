"""TensorFlow implementation of the Sparse Autoencoder from the UFLDL tutorisals
"""

import numpy as np
import scipy.io
import tensorflow as tf
import math as math
import matplotlib.pyplot as plt

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


def train():
    sess = tf.InteractiveSession()
    # LOAD image data
    images = np.array(scipy.io.loadmat("IMAGES.mat")["IMAGES"])
    def normalize(unNormalizedTensor):
        normalized = unNormalizedTensor - np.mean(unNormalizedTensor)
        pstd = 3 * np.std(normalized)
        normalized = np.maximum(np.minimum(normalized, pstd),-1*pstd)/pstd
        normalized = (normalized +1) *.4 + 0.1
        return normalized
        
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

    
    

    def displayWeights(weights, blocking=False):
        num_tiles = math.sqrt(N_HIDDEN)
        image = np.zeros((int(num_tiles*PATCHWIDTH + num_tiles+1),int(num_tiles*PATCHWIDTH + num_tiles+1)))
        for i in range(N_HIDDEN):
            subWeights = normalize(weights[:,i])
            subWeights = np.reshape(subWeights,(PATCHWIDTH,PATCHWIDTH))
            denom = np.sqrt(np.dot(subWeights,subWeights))
            subWeights = np.divide(subWeights,denom)
            xIndex = i % num_tiles
            yIndex = i // num_tiles
            xStart = int((xIndex+1)+(xIndex*PATCHWIDTH))
            yStart = int((yIndex+1)+(yIndex*PATCHWIDTH)) 
            image[xStart:int(xStart+PATCHWIDTH),yStart:int(yStart+PATCHWIDTH)] = subWeights
            
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
        
        result = sess.run(pred['out'], feed_dict={x: patches})
        
        repro = np.zeros((512,512))
        for r in range(tile_size):
            for c in range(tile_size):
                repro[r*PATCHWIDTH:r*PATCHWIDTH+PATCHWIDTH,c*PATCHWIDTH:c*PATCHWIDTH+PATCHWIDTH] = np.reshape(result[r*tile_size+c],(PATCHWIDTH,PATCHWIDTH))
        
        
        image=np.concatenate((orig,repro),axis=1)
        plt.figure(2)        
        plt.imshow(image,interpolation='none')      
        plt.draw()
        plt.show(block=blocking)


    # Input placehoolders
    with tf.name_scope('input'):
        #Construct the tensor flow model
        x = tf.placeholder("float", [None, N_INPUT], name='x-input')
        hidden = tf.placeholder("float", [None, N_HIDDEN], name='hidden-activation')

    def autoencoder(X, weights, biases):
        with tf.name_scope('hidden_layer'):
            hiddenlayer = tf.sigmoid(
                tf.add(
                    tf.matmul(
                        X, weights['hidden']
                    ),
                    biases['hidden']
                )
            )
        with tf.name_scope('output_layer'):
            out = tf.sigmoid(
                tf.add(
                    tf.matmul(
                        hiddenlayer, weights['out']
                    ), 
                    biases['out']
                )
            )
        return {'out': out, 'hidden': hiddenlayer}
  
    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)
                tf.scalar_summary('max/' + name, tf.reduce_max(var))
                tf.scalar_summary('min/' + name, tf.reduce_min(var))
                tf.histogram_summary(name, var)
          
    weights = {
        'hidden': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN])),
        'out': tf.Variable(tf.random_normal([N_HIDDEN, N_OUTPUT]))
    }
    variable_summaries(weights['hidden'], 'hidden_layer' + '/weights')
    variable_summaries(weights['out'], 'output_layer' + '/weights')
    
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN])),
        'out': tf.Variable(tf.random_normal([N_OUTPUT]))
    }
    variable_summaries(biases['hidden'], 'hidden_layer' + '/biases')
    variable_summaries(biases['out'], 'output_layer' + '/biases')

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
    
    with tf.name_scope('loss'):
        cost_J = tf.div(tf.nn.l2_loss(diff ),tf.constant(float(NUMSAMPLES)))
        tf.scalar_summary('loss',cost_J)
    
    with tf.name_scope('cost_sparse'):
        cost_sparse = tf.mul(BETA,  tf.reduce_sum(KL_Div(RHO, rho_hat)))
        tf.scalar_summary('cost_sparse',cost_sparse)
    
    with tf.name_scope('cost_reg'):
        cost_reg = tf.mul(LAMBDA , tf.add(tf.nn.l2_loss(weights['hidden']), tf.nn.l2_loss(weights['out'])))
        tf.scalar_summary('cost_reg',cost_reg)
        
    with tf.name_scope('cost'):
        cost = tf.add(tf.add(cost_J , cost_reg ), cost_sparse)
        tf.scalar_summary('cost',cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/tmp/sae_logs', sess.graph)
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
        summary,_ = sess.run([merged,optimizer], feed_dict={x: samples})
        if i % 1000 == 0:
            c_old = c
            c,j,reg,sparse = sess.run([cost,cost_J,cost_reg,cost_sparse], feed_dict={x: samples})
            print "EPOCH %d: COST = %f, LOSS = %f, REG_PENALTY = %f, SPARSITY_PENTALTY = %f" %(i,c,j,reg,sparse)
            displayWeights(sess.run(weights['hidden']))
            displayImageTest(images[:,:,vizImage])
        i += 1
        writer.add_summary(summary, i)
    print("Optimization Finished!")

def main(_):
    train()
    
if __name__ == '__main__':
  tf.app.run()
        
