from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs, neibs2images

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs, n_blocks):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
            Dij = D[:, :nc]
            plot(cij, Dij, n_blocks, X_mn, axarr[i, j])

    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)

def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

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

    
def plot(c, D, n_blocks, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
    X = np.dot(D, c).T
    X = X + np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)
    neib = T.matrix()
    image = neibs2images(neib, (int(256/n_blocks), int(256/n_blocks)), (1, 1, 256, 256))[0, 0, :, :]
    reconstruct = theano.function([neib], image)
    img = reconstruct(X)
    plt.subplot(ax)
    plt.imshow(img, cmap='gray')
    plt.axis('off')


def main():
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    file_names = []
    for root, dirs, files in walk('jaffe/', topdown=True):
        for name in files:
            file_names.append(name)
    file_names.sort()
    Ims = np.zeros((len(file_names), 256, 256))
    for i in range(len(file_names)):
        img = Image.open('jaffe/'+file_names[i])
        Ims[i, :, :] = np.asarray(img)
    Ims = Ims.reshape((1, Ims.shape[0], Ims.shape[1], Ims.shape[2]))
        
    szs = [16, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''
        images = T.tensor4(str(sz))
        neibs = images2neibs(images, (sz, sz))
        divide = theano.function([images], neibs, allow_input_downcast=True)
        X = divide(Ims)
        
        X_mn = np.mean(X, 0)
        X = X - np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        eigen_val, eigen_vec = np.linalg.eigh(np.dot(X.T, X))
        D = np.fliplr(eigen_vec)
        
        c = np.dot(D.T, X.T)
        size = [sz, sz]
        for i in range(0, 200, 10):
            plot_mul(c, D, i, X_mn.reshape(size), num_coeffs=nc, n_blocks=int(256/sz))

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
