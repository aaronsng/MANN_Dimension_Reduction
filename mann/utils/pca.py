from PIL.Image import NONE
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import os

class PCA(object):
    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        pass

    def set_spectral_image(self, imager: np.array):
        """
        Sets the spectral image 
        """

        self.spectral_image = imager
        img_shape = self.spectral_image.shape[:2]

        # specify no of bands in the image
        n_bands = self.num_channels

        self.MB_img = np.zeros((img_shape[0],img_shape[1],n_bands))  # 3 dimensional dummy array with zeros

        for i in range(n_bands):
            self.MB_img[:,:,i] = self.spectral_image[:,:,i]  # stacking up images into the array

    def show_spectral_bands(self):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as grid

        fig,axes = plt.subplots(6,5,figsize=(23,23),sharex='all', sharey='all')   #img_shape[0]/50,img_shape[1]/50
        fig.subplots_adjust(wspace=0., hspace=0.1)
        fig.suptitle('Intensities at Different Bandwidth in the visible and Infra-red spectrum', fontsize=30)

        axes = axes.ravel()

        for i in range(self.num_channels):
            axes[i].imshow((self.MB_img[:,:,i] * 255).astype(np.uint8),cmap='gray', vmin=0, vmax=255)
            axes[i].set_title('band '+str(i+1),fontsize=25)
            axes[i].axis('off')
        fig.delaxes(axes[-1])

    def display_n_principal_components(self):
        """
        Display the n principal components
        """
        assert self.PC != None

        # Rearranging 1-d arrays to 2-d arrays of image size
        img_shape = self.spectral_image.shape

        PC_2d = np.zeros((img_shape[0],img_shape[1],self.num_channels))
        for i in range(self.num_channels):
            PC_2d[:,:,i] = self.PC[:,i].reshape(-1,img_shape[1])

        # Normalising between 0 to 255
        PC_2d_Norm = np.zeros((img_shape[0],img_shape[1],self.num_channels))
        for i in range(self.num_channels):
            PC_2d_Norm[:,:,i] = cv2.normalize(PC_2d[:,:,i], np.zeros(img_shape), 0, 255, cv2.NORM_MINMAX)

        fig,axes = plt.subplots(6,5,figsize=(50,23),sharex='all', sharey='all')   #img_shape[0]/50,img_shape[1]/50
        fig.subplots_adjust(wspace=0.1, hspace=0.15)
        fig.suptitle('Intensities of Principal Components ', fontsize=30)

        axes = axes.ravel()
        for i in range(self.num_channels):
            axes[i].imshow(PC_2d_Norm[:,:,i],cmap='gray', vmin=0, vmax=255)
            axes[i].set_title('PC '+str(i+1),fontsize=25)
            axes[i].axis('off')
        fig.delaxes(axes[-1])

    def perform_pca(self, num_channels: int):
        # Convert 2d band array in 1-d to make them as feature vectors and Standardization  
        MB_matrix = np.zeros((self.MB_img[:,:,0].size, self.num_channels))
        for i in range(self.num_channels):
            MB_array = self.MB_img[:,:,i].flatten()  # covert 2d to 1d array 
            MB_arrayStd = (MB_array - MB_array.mean()) / MB_array.std()  # Standardize each variable 
            MB_arrayStd[np.isnan(MB_arrayStd)] = 0
            MB_matrix[:,i] = MB_arrayStd

        # Covariance
        np.set_printoptions(precision=3)
        cov = np.cov(MB_matrix.transpose())

        # Eigen Values
        EigVal,EigVec = np.linalg.eig(cov)

        # Ordering Eigen values and vectors
        order = EigVal.argsort()[::-1]
        EigVal = EigVal[order]
        EigVec = EigVec[:,order]

        #Projecting data on Eigen vector directions resulting to Principal Components 
        self.PC = np.matmul(MB_matrix,EigVec)   #cross product
        print("\nFirst PC retain "+str(int(sum(EigVal[:1])/sum(EigVal)*100))+" % of information")

        return self.PC[...,:num_channels] # returns the most significant principal component