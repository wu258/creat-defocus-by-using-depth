import numpy as np
import os
import ctypes as c
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import circle
from multiprocessing import Process,Manager
import multiprocessing
import itertools
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
defocusKernelDims = [3,5,7,9]
def convolveInPiex(kernel,img,depth_map,depth):
    circleCenterCoord,circleRadius=getKernalParmate(kernel)
    
    sum=0
    len_y=len(img)
    len_x=len(img[0])
  
    sum=convolve2d(img, kernel, mode='valid', fillvalue=255).astype("uint8")
    return sum
def getKernalParmate(kernel):
    dim=len(kernel)
    circleCenterCoord=0
    circleRadius=0
    if dim%2==0:
        circleCenterCoord = (dim-1) // 2
        circleRadius = dim / 2
    else:
        circleCenterCoord = (dim) // 2
        circleRadius = (dim) / 2
    return circleCenterCoord,circleRadius

def getNewMatrix(y,x,kernel,img):
    dim=len(kernel)
    result = [0 for i in range(dim*dim)]  
    result=np.reshape(result, (dim, dim))
    circleCenterCoord,circleRadius=getKernalParmate(kernel)
    start_y=y-circleRadius
    start_x=x-circleRadius
    
def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))    
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)

def DefocusBlur(img, dim):
    imgarray = np.array(img, dtype="float32")
    kernel = DiskKernel(dim)
    convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img
def getPeddingSize(depth_map,focus_dis):
    map = np.array(depth_map)
    diff = np.absolute(map - focus_dis)
    return np.max(diff)

def deal_y_single(result,pedding_size,img,depth_map,y,focus_dis):
    index_x = itertools.count(pedding_size, 1) 
    for x in index_x:
        if(x>=len(img[y])-pedding_size):
            break
        #print(x)
        kernel = DiskKernel(abs(depth_map[y-pedding_size][x-pedding_size]-focus_dis))
        if len(kernel)==0:
            result[y-pedding_size][x-pedding_size]=img[y][x]
            continue
        circleCenterCoord,circleRadius=getKernalParmate(kernel)
        tem_y_u=y-(circleCenterCoord)
        tem_y_d=y+(len(kernel)-circleCenterCoord)
        tem_x_l=x-circleCenterCoord
        tem_x_r=x+(len(kernel[0])-circleCenterCoord)
        temp_matrix=img[tem_y_u:tem_y_d,tem_x_l:tem_x_r]
        depth_matrix=depth_map[tem_y_u:tem_y_d,tem_x_l:tem_x_r]
        #print(len(kernel))
        #print(temp_matrix)
        convolved = convolveInPiex(kernel,temp_matrix,depth_matrix,depth_map[y-pedding_size][x-pedding_size])
        #print(convolved)
        result[y-pedding_size][x-pedding_size]=convolved
    #del result
    #existing_shm.close()
    #print("********************************")
    #print(result[0][0])
    #print("_________________________________________")
    print("y!")
    return
def deal_y(pedding_size,img,depth_map,y,focus_dis,shm_name,y_len,x_len):
    index_x = itertools.count(pedding_size, 1) 
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    result=np.ndarray((y_len,x_len), dtype=img.dtype, buffer=existing_shm.buf)
    for x in index_x:
        if(x>=len(img[y])-pedding_size):
            break
        #print(x)
        kernel = DiskKernel(abs(depth_map[y-pedding_size][x-pedding_size]-focus_dis))
        if len(kernel)==0:
            result[y-pedding_size][x-pedding_size]=img[y][x]
            continue
        circleCenterCoord,circleRadius=getKernalParmate(kernel)
        tem_y_u=y-(circleCenterCoord)
        tem_y_d=y+(len(kernel)-circleCenterCoord)
        tem_x_l=x-circleCenterCoord
        tem_x_r=x+(len(kernel[0])-circleCenterCoord)
        temp_matrix=img[tem_y_u:tem_y_d,tem_x_l:tem_x_r]
        #print(len(kernel))
        #print(temp_matrix)
        convolved = convolve2d(temp_matrix, kernel, mode='valid', fillvalue=255.0).astype("uint8")
        #print(convolved)
        result[y-pedding_size][x-pedding_size]=convolved[0][0]
    #print(result[y-pedding_size][500])
    del result
    existing_shm.close()
    #print("********************************")
    #print(result[0][0])
    #print("_________________________________________")
    return
    
def DefocusBlurFromDepth(img,depth_map,focus_dis):
    imgarray = np.array(img, dtype="float32")
    manager=Manager()
    y_len=len(img)
    x_len=len(img[0])
    pedding_size=int(getPeddingSize(depth_map,focus_dis))
    result=[0 for i in range(len(img)*len(img[0]))]
    result=np.reshape(result,(len(img),len(img[0])))
    t_img=[0 for i in range(len(img)*len(img[0]))]
    t_depth=[0 for i in range(len(img)*len(img[0]))]
    t_img[:]=img[:]
    t_depth[:]=depth_map[:]
    t_img=np.pad(t_img, ((pedding_size,pedding_size),(pedding_size,pedding_size)), 'constant')
    #t_depth=np.pad(t_depth, ((pedding_size,pedding_size),(pedding_size,pedding_size)), 'constant')
    #rint(img)
    index_y = itertools.count(pedding_size, 1)
    '''
    for y in index_y:
        if(y>=len(img)-pedding_size):
            break
        deal_y_single(result,pedding_size,t_img,t_depth,y,focus_dis)
    
    '''
    cores = multiprocessing.cpu_count()
    print(cores)
    pool = multiprocessing.Pool(processes=cores)
    shm = shared_memory.SharedMemory(create=True, size=result.nbytes)
    b = np.ndarray(result.shape, dtype=result.dtype, buffer=shm.buf)
    b[:] = result[:]
    for y in index_y:
        if(y>=len(t_img)-pedding_size):
            break
        
        p_result=pool.apply_async(func=deal_y, args=(pedding_size,t_img,t_depth,y,focus_dis,shm.name,y_len,x_len,))

        #p_result.get()
    pool.close()
    pool.join()
    
    existing_shm = shared_memory.SharedMemory(name=shm.name)
    b=np.ndarray((len(img),len(img[0])), dtype=img.dtype, buffer=existing_shm.buf)
    print(b[200][500])
    result[:]=b[:]
    del b
    existing_shm.close();
    shm.close()
    shm.unlink()

    print(result[0][0])
    return result

def get_Image_path(path,result):
    for filename in os.listdir(path):
        file_path=path+filename
        if os.path.isdir(file_path):
            get_Image_path(file_path+"/",result)
        else:
            file_type=os.path.splitext(file_path)[-1]
            if ".png"==file_type:
                #png_path=os.path.splitext(file_path)[0]+".npy"
                if file_path.split('/')[-1]=="im0.png":
                   result.append((file_path))
    return result
def get_DepthMap_path(path):
    path=path.replace("MiddEval3-data-F","MiddEval3-GT0-F")
    path=path.replace(path.split('/')[-1],"disp0GT.npy")
    return path


def DiskKernel(dim):
    kernelwidth = dim
    dim=kernelwidth=kernelwidth.astype(np.int)
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    if dim%2==0:  
        circleCenterCoord = (dim-1) / 2
        circleRadius = dim / 2
    else:
        circleCenterCoord = (dim) // 2
        circleRadius = (dim) / 2
    
    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr,cc]=1
    #print(kernel)
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel

def deal(filename):
    depthMap_path=get_DepthMap_path(filename)
    #print(depthMap_path)
    img= Image.open(filename)
    im_array = np.array(img)
    depth_array = np.load(depthMap_path)
    depth_array=np.array(depth_array)
    print(filename.replace(filename.split('/')[-1],"defocusIm1.png"))
    #print(np.dtype(depth_array))
    #print(np.dtype(im_array))
    #K=DiskKernel(10)
    im_array[:, :, 0]=DefocusBlurFromDepth(im_array[:, :, 0],depth_array,50)
    print("dsdsdsdsssssssssssssssssssssssssssssssssssdsdsdsds")
    im_array[:, :, 1]=DefocusBlurFromDepth(im_array[:, :, 1],depth_array,50)
    print("dsdsdsdsssssssssssssssssssssssssssssssssssdsdsdsds")
    im_array[:, :, 2]=DefocusBlurFromDepth(im_array[:, :,2],depth_array,50)
    print(im_array)
    im = Image.fromarray(im_array)
    im.save(filename.replace(filename.split('/')[-1],"defocusIm1.png"))
    print("sssssssssssssssssssssssssssssssssssssssss")
 
def main():
    path_set=[]
    path_set=get_Image_path("/scratch1/wu096/MiddEval3/MiddEval3-data-F/MiddEval3-data-F/MiddEval3/trainingF/",path_set)
    #cores = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(processes=cores)
    #for y in pool.imap(deal, path_set):
        #print(y)
    for item in path_set:
        deal(item)
main()

