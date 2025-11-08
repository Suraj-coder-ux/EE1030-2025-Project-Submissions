import numpy as np 
import time 
from PIL import Image
import os

#Read the image 
filename = input("Enter image name : ") 
img = Image.open(filename)
img_array = np.array(img) #convert image into a numpy array

#get only the base name of the filename 
base_name,extension = os.path.splitext(filename) #split the name into basename and extention

#k values to test 
ks = [5,20,50,100] 

#to check if image is greyscale or colored image 
if len(img_array.shape)==2: #if array has 2 dimensions ,then it is greyscale
    #it is greyscale 
    print("\nGrayscale Image")
    print("k   | Time(s)  | Error")
    print("----|----------|----------")

    A = img_array.astype(float) #convert to float 

    for k in ks :

        start = time.time() 

        #compute svd
        U,S,VT = np.linalg.svd(A,full_matrices=False) #full_matrices=False means it calculates reduced svd ,where u : mxr , vt : rxn and r = min(m,n)

        #low rank approximation (keeping top k values)
        U_k = U[:,:k] #k columns
        S_k = S[:k] #k singular values
        VT_k = VT[:k,:] #k rows

        #reconstruct the new compressed matrix 
        A_k = U_k @ np.diag(S_k) @VT_k 

        time_taken = time.time() - start

        #calculate frobenius error 
        error = np.sqrt(np.sum((A-A_k)**2)) #each element of A-A_k is squared ,then added and then sqrt is applied on it.

        #save the new compressed image 
        A_k = np.clip(A_k,0,255).astype(np.uint8) #to ensure all values are in the range of 0 to 255 , and convert it into unsigned 8-bit integer from float
        out_img = Image.fromarray(A_k,mode = 'L') # L specifies the image is greyscale, create PIL image object from numpy array A_k
        out_img.save(f"{base_name}_numpy_{k}{extension}") 

        print(f"{k:<4}| {time_taken:<8.4f}| {error:.2f}") 

else: #colored image 

    print("\nColor Image")
    print("k   |  Time(s)  | R Error | G Error | B Error")
    print("----|---------|---------|---------|--------") 

    #divide into r,g,b channels 
    R = img_array[:,:,0].astype(float) #extracts red channel
    G = img_array[:,:,1].astype(float) #extracts green channel
    B = img_array[:,:,2].astype(float) #extracts blue channel

    for k in ks:

        start = time.time()

        #compute svd for r channel 
        Ur , Sr , VTr = np.linalg.svd(R,full_matrices=False)
        R_k = Ur[:,:k] @ np.diag(Sr[:k]) @ VTr[:k,:] 

        #compute svd for g channel 
        Ug , Sg , VTg = np.linalg.svd(G,full_matrices=False)
        G_k = Ug[:,:k] @ np.diag(Sg[:k]) @ VTg[:k,:] 

        #compute svd for b channel 
        Ub , Sb , VTb = np.linalg.svd(B,full_matrices=False)
        B_k = Ub[:,:k] @ np.diag(Sb[:k]) @ VTb[:k,:] 

        time_taken = time.time() - start 

        #calculate errors 

        err_r = np.sqrt(np.sum((R-R_k)**2))
        err_g = np.sqrt(np.sum((G-G_k)**2))
        err_b = np.sqrt(np.sum((B-B_k)**2)) 

        #merge channels and save image 
        img_out = np.stack([R_k,G_k,B_k],axis = 2) #stacks three reconstructed channels along the 3rd dimension to form image array, R,G,B have 2 dimensions height x width and axis=2 means the new axis will be inserted at the index=2 (3rd dimension)
        img_out = np.clip(img_out,0,255).astype(np.uint8)
        out_img = Image.fromarray(img_out,mode="RGB") #to denote it is colored image , and create pil image object from img_out
        out_img.save(f"{base_name}_numpy_{k}{extension}")

        print(f"{k:<4}| {time_taken:<8.4f}| {err_r:<8.2f}| {err_g:<8.2f}| {err_b:<8.2f}")

print("\ncompleted")


    
