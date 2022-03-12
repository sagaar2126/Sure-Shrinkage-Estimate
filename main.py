    import numpy as np
import matplotlib.pyplot as plt
import skimage.io 
import cv2
import math

plt.rcParams['figure.dpi'] = 500
path='lighthouse2.bmp'


#============================================================================================================
#This function is only used for calculating MSE and denoised  images for problem 1.
#for each filter size, denoise=[] array contains denoised images for sigma = {0.1,1,2,4,8}
#============================================================================================================

def calcmse_1(size,Y,sigma,X):
    H,W=Y.shape
    denoise=[]
    mse=[]
    for k in sigma:
        kernel=get_Gaussian_Blur(size,k)
        Mu_Y=Convolve(Y,kernel)
        # Mu_Y=Mu_Y.astype('uint8')
        Mu_Y[Mu_Y<0]=0
        Mu_Y[Mu_Y>255]=255
        denoise.append[Mu_Y]
        e = (np.sum(np.square(X-Mu_Y)))/(H*W)
        mse.append(e)
    return mse,denoise

#=============================================================================================================

#============================================================================================================
#This function is only used for calculating MSE and denoised  images for problem 2.
#for each filter size denoise array contains denoised images for sigma = {0.1,1,2,4,8}
#============================================================================================================
def calcmse_2(Size,Y,Z,X,sigma):
    H,W=X.shape
    # A contains multiple of 16 upto image height. Will be used for creating patches.  (512/16)+1 = 32
    A=[i*16 for i in range(33)]
    
    # B contains multiple of 16 upto image Width. Will be used for creating patches.  (768/16)+1 = 49
    B=[i*16 for i in range(49)]
    
    
    denoise=[]        #<------- contains denoised images
    mse=[]
    # Initializing denoised image.
    denoised=np.zeros((H,W))
    
    for r in sigma:
        kernel=get_Gaussian_Blur(Size,r)
        k=2;
        
        for i in range(len(A)-2):
            l=2
            for j in range(len(B)-2):
                patch_Y = Y[A[i]:A[k],B[j]:B[l]]
                # patch_Z = Z[A[i]:A[k],B[j]:B[l]]
                Mu_Y=Convolve(patch_Y,kernel)
                # Mu_Z=Convolve(patch_Z,kernel)
                Y1=(patch_Y-Mu_Y)
                # Z1=(patch_Z-Mu_Z)
            
                Variance_Y1= get_Variance(Y1)
                # Variance_Z1= (Size*Size-np.sum(np.square(kernel)))*100
                # Variance_Z1= get_Variance(Z1)
                Variance_Z1 = (1-np.sum(np.square(kernel)))*100  
            
                Variance_X1= Variance_Y1-Variance_Z1
                
                X1_hat=(Variance_X1/Variance_Y1)*Y1
            
                denoised[A[i]:A[k],B[j]:B[l]]=Mu_Y+X1_hat
                l+=1
            k+=1
        
        denoised[denoised<0]=0
        denoised[denoised>255]=255
        
        e = (np.sum(np.square(X-denoised)))/(H*W) 
        denoise.append(denoised)

        mse.append(e)    
    return mse , denoise 
#=======================================================================================================================

def calcmse_3(Y,Z,X,size,sigma):
    
    H,W=X.shape
    
    
    A=[i*32 for i in range(17)]
    B=[i*32 for i in range(25)]
    t_sure=[]
    denoise=[]
    denoised= np.zeros(X.shape)
    mse=[]
    for r in sigma:
         kernel=get_Gaussian_Blur(size, r)
         Mu_Y=Convolve(Y, kernel)
      
         Y_1 = Y-Mu_Y
         k=1
         t_iteration=[]
         for i in range(len(A)-1):
            l=1 
            for j in range(len(B)-1):
                # taking patch of high pass component of noisy image.
                patch_Y = Y_1[A[i]:A[k],B[j]:B[l]]     
            
                # taking patch of high pass component of noise.
                # patch_Z = Z_1[A[i]:A[k],B[j]:B[l]]
            
                # Calculating variance of high pass component of noise.
                
                Variance_Z1 = (1-np.sum(np.square(kernel)))*100  
                # Variance_Z1=get_Variance(patch_Z)
           
                # It contains possible values fo threshold t.
                R=list(np.linspace(0,100,150))            
            
                Z=0
                score=0
                t=0
                for p in R:                   # Iterativley minimizing over t.
                
                    # This will brodcast zero where t is greater than zero.
                    M=np.where(abs(patch_Y)<p,patch_Y,0)       
                
                    # Counting number of zeros.This calculates in 32*32 patch where t is smaller.
                    num_zero = 32*32-np.count_nonzero(M)
                   
                    # ΣΣg(y(i,j))^2 = ΣΣ M(i,j)^2+num_zero*t^2
               
                    # np.sum(np.where(abs(patch_Y)<p)) Calculates cardinality of set where |Y1(i,j)|<t.
                
                    Z=(32**2)*Variance_Z1-2*Variance_Z1*np.sum(np.where(abs(patch_Y)<p))+np.sum(np.square(M))+num_zero*p**2      
                    
                    if(Z<score):
                       score=Z
                       t=p
                zero_arr=np.zeros((32,32))
                    
                diff=abs(patch_Y)-t
                
                
                denoised[A[i]:A[k],B[j]:B[l]]=np.multiply(np.sign(patch_Y),np.maximum(diff,zero_arr))
            
                t_iteration.append(t)                                               
            
                l=l+1
            k=k+1
         t_sure.append(t_iteration)   
         X_hat=Mu_Y+denoised
         denoise.append(X_hat)
         
         e = (np.sum(np.square(X-X_hat)))/(H*W)
         mse.append(e)   
    return mse , denoise , t_sure   
        
        
    

    
#=================================================================================================================
             # Below given functions are supporting functions.
#================================================================================================================       
# This calculate approximate Variance.      


def get_Variance(img):
    h,w=img.shape
    square=np.sum(np.square(img))
    return square/(h*w)

# Convolution.
def Convolve(img,kernel):
    
    Convolved=cv2.filter2D(img, -1, kernel)
    return Convolved

 # This function Compute gaussian kernel
def get_Gaussian_Blur(N,sigma):
   
    X_=np.linspace(-int(N/2),int(N/2),N)
    Y_=np.linspace(-int(N/2),int(N/2),N)
    X,Y=np.meshgrid(X_,Y_)
    
    gaussian_kernel = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
    return gaussian_kernel/np.sum(gaussian_kernel)

#=================================================================================================================

   # problem1(),problem2(),problem3()--->Image denoising , problem4()----->Image Sharpnening
   
   # Main functions starts from below..
#==================================================================================================================
def problem1():
    image=skimage.io.imread(path)
    
    # X is clear image , same notations is used as taught in class.
    X=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    
    # Z is height*width gaussian random array , act as nosie.
    sigmaZ=math.sqrt(100)
    Z=np.random.normal(0,sigmaZ,(X.shape))
    
    # Y is noise corrupted image.
    
    Y=X+Z
    # Making Pixel values in between 0 to 255.
    Y[Y<0]=0
    Y[Y>255]=255
    # plt.imshow(Y,cmap='gray')
    # plt.show()

    
    sigma=[0.1,1,2,4,8]
    
    # Calculating MSE for different filter sizes
    mse3,denoise3=calcmse_1(3,Y,sigma,X)
    mse7,denoise7=calcmse_1(7,Y,sigma,X)
    mse11,denoise11=calcmse_1(11,Y,sigma,X)
    fig, ax = plt.subplots()
        
    ax.scatter(sigma, mse3, label='Filter size=3') 
    ax.scatter(sigma, mse7, label='Filter size=7')  
    ax.scatter(sigma, mse11, label='Filter size=11')  
    ax.set_xlabel('Sigma')  
    ax.set_ylabel('mse')
    ax.legend()
    ax.grid(True)
    plt.show()    
    print(mse3)
    print(mse7)
    print(mse11)
    
# Adaptive Weiner Filter

def problem2():
    
    # Loading image and converting it to gray scale.
    image=skimage.io.imread(path)
    X=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    H,W=X.shape
    
    plt.imshow(X, cmap='gray')
    plt.show()
    # Generating noise.
    sigmaZ=math.sqrt(100)
    Z=np.random.normal(0,sigmaZ,(H,W))
    
    
    Y=X+Z
    Y[Y<0]=0
    Y[Y>255]=255
    plt.imshow(Y,cmap='gray')
    plt.show()
    sigma=[0.1,1,2,4,8]
  
    mse3,denoise3   =     calcmse_2(3,Y,Z,X,sigma)
    mse7,denoise7   =     calcmse_2(7,Y,Z,X,sigma)
    mse11,denoise11 =     calcmse_2(11,Y,Z,X,sigma)
    fig, ax = plt.subplots()
    
    # If you want to plot denoised image then denoise3,denoise7,denoise11,Contains all
    # the images. Uncomment following code to display image.
    
    plt.imshow(denoise3[0],cmap='gray')
    plt.show()    
    
    
    ax.scatter(sigma, mse3, label='Filter size=3') 
    ax.scatter(sigma, mse7, label='Filter size=7')  
    ax.scatter(sigma, mse11, label='Filter size=11')  
    ax.set_xlabel('Sigma')  
    ax.set_ylabel('Mse')
    ax.legend()
    ax.grid(True)
    plt.show()    
    print(mse3)
    print(mse7)
    print(mse11)
    
    
def problem3():
    image=skimage.io.imread(path)
    X=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    H,W=X.shape
    
    
    # Generating noise.
    sigmaZ=math.sqrt(100)
    Z=np.random.normal(0,sigmaZ,(H,W))
    
   
    Y=X+Z
    
    # Avoiding saturation.
    
    Y[Y<0]=0
    Y[Y>255]=255
  
    sigma=[0.1,1,2,4,8]
    # sigma=[1]
    
    # Since this program is calulating mse for all possible filter sizes and sigma
    # This could take time , if you want to compute only for sigma=1 , then uncomment
    # upper give line and put comment on sigma=[0.1,1,2,4,8]
    
    mse3,denoise3,t3     =     calcmse_3(Y,Z,X,3,sigma)
    mse7,denoise7,t7     =     calcmse_3(Y,Z,X,7,sigma)
    mse11,denoise11,t11  =     calcmse_3(Y,Z,X,11,sigma)
    fig, ax = plt.subplots()
    
    
    # If you want to plot denoised image then denoise3,denoise7,denoise11,Contains all
    # the images (for all sigma values). Uncomment following code to display image.
    
    # denoise3[0],denoise3[1],denoise3[2],denoise3[3]...etc are the images.
    
    # plt.imshow(denoise3[0],cmap='gray')
    # plt.show()
    # print(t3)
    
    
    ax.scatter(sigma, mse3, label='Filter size=3') 
    ax.scatter(sigma, mse7, label='Filter size=7')  
    ax.scatter(sigma, mse11, label='Filter size=11')  
    ax.set_xlabel('Sigma')  
    ax.set_ylabel('Mse')
    ax.legend()
    ax.grid(True)
    plt.show()    
    print(mse3)
    print(mse7)
    print(mse11)
 
def problem4():
    
    
    #Same as problem 1. Computing denoised image.
    image=skimage.io.imread(path)
    X=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    H,W=X.shape
    sigmaZ=math.sqrt(100)
    Z=np.random.normal(0,sigmaZ,(H,W))
    Y=X+Z
    Y[Y<0]=0
    Y[Y>255]=255
    
    #Chosing filter sizer and Gaussian blur variance (best suitable filter)
    Filter_size=3
    sigma=1
    kernel=get_Gaussian_Blur(Filter_size,sigma)
   
    
    # Solution starts from here
    High_pass_kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    
    
    
    # Computing Low pass pass component of denoised image.
    img = Convolve(Y, kernel)
    
    # plt.imshow(img,cmap='gray')
    # plt.show()
    
    # Computing High pass component of denoised image.
    
    high_pass=Convolve(img,High_pass_kernel)
    # plt.imshow(high_pass,cmap='gray')
    # plt.show()
    # Taking combination of low pass component wih high pass component with constant gain being multiplied with high pass.
    
    lamb=list(np.linspace(0,10,1000))
    mse=[]
    e_min=1000
    t_optimal=0
    for r in lamb:
        
       # Taking combination of low pass component wih high pass component with constant gain being multiplied with high pass.

        X_hat=r*high_pass+img
        X_hat[X_hat<0]=0
        X_hat[X_hat>255]=255
       
        e = (np.sum(np.square(X-X_hat)))/(H*W)
        
        mse.append(e)
        
        # for determining optimal gain
        if(e<e_min):
            e_min=e
            t_optimal=r
    
    ## Optimal gain factor.....
    print(t_optimal)
    
    #plotting curve MSE vs Gain...
    
    fig, ax = plt.subplots()
    ax.plot(lamb, mse)  
    ax.set_xlabel('Lambda')  
    ax.set_ylabel('Mse')
    ax.legend()
    ax.grid(True)
    
    plt.show()   
    
    # displaying image for optimal gain..
    
    X_hat=t_optimal*high_pass+img
    X_hat[X_hat<0]=0
    X_hat[X_hat>255]=255
    plt.imshow(X_hat,cmap='gray')
    plt.show()
    
    # MSE for optmal gain...
    e = (np.sum(np.square(X-X_hat)))/(H*W)
    print(e)
    

def main():
    # problem1()
    # problem2()
    problem3()
    # problem4()



if __name__ == "__main__":
    main()