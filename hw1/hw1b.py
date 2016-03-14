from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''
    f, axarr = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            D_to_plot = np.reshape(D[:, (i*4+j)], (sz, sz))
            plt.subplot(axarr[i, j])
            plt.imshow(D_to_plot, cmap='gray')
            plt.axis('off')
    f.savefig(imname)
    plt.close(f)

def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    X = np.dot(D, c).T.reshape((256, 256))
    X = X + X_mn
    plt.subplot(ax)
    plt.imshow(X, cmap='gray')
    plt.axis('off')


if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''
    file_names = []
    for root, dirs, files in walk('jaffe/', topdown=True):
        for name in files:
            file_names.append(name)
    file_names.sort()
    I = np.zeros((len(file_names), 256*256))
    for i in range(len(file_names)):
        img = Image.open('jaffe/'+file_names[i])
        I[i, :] = np.asarray(img).reshape((256*256,))
    
    Ims = I.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    X_var = T.dmatrix()
    D_var = T.dmatrix()
    lambdas = T.dvector()
    ita = 0.1
    d = theano.shared(np.random.randn(65536))
    x_term = T.dot(T.dot(X_var, d).T, T.dot(X_var, d))
    d_term = T.dot(T.dot(D_var, d).T, T.dot(D_var, d)*lambdas)
    gradient = T.grad(x_term-d_term, d)
    y = d + ita * gradient
    y_norm = T.sqrt(T.sum(T.sqr(y)))
    loop_body = theano.function(inputs=[X_var, D_var, lambdas], 
                                updates=((d, (d + ita*gradient) / y_norm),)
                               )
    
    lambdas_here, D = np.array([]), np.empty(shape=(0, 65536))
    for i in range(16):
        for t in range(100):
            loop_body(X, D, lambdas_here)
            d_current = d.get_value()
            '''
            if len(D) >= 2:
                if np.abs(np.linalg.norm(D[-2]) - np.linalg.norm(D[-1])) < epsilon:
                    break
            '''
        lambdas_here = np.append(lambdas_here, np.dot(np.dot(X, d_current).T, np.dot(X, d_current)))
        D = np.vstack((D, d_current))
        d.set_value(np.random.randn(65536))
        print("Principal component {} found.".format(i))
    c = np.dot(D, X.T)
        
    for i in range(0,200,10):
        plot_mul(c, D.T, i, X_mn.reshape((256, 256)), 
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])
        print("PCA No. {} finished.".format(i))

    plot_top_16(D.T, 256, 'output/hw1b_top16_256.png')
    print('Program finished')

