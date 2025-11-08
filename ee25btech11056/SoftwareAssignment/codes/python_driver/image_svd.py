import numpy as np 
import time 
from PIL import Image
import os

#Read the image 
filename = input("Enter image name : ") 
img = Image.open(filename)
img_array = np.array(img) 

#get only the base name of the filename 
base_name,extension = os.path.splitext(filename)

#k values to test 
ks = [5,20,50,100] 

#to check if image is greyscale or colored image 
if len(img_array.shape)==2:
    #it is greyscale 
    print("\nGrayscale Image")
    print("k   | Time(s)  | Error")
    print("----|----------|----------")

    A = img_array.astype(float) #convert to float 

    for k in ks :

        start = time.time() 

        #compute svd
        U,S,VT = np.linalg.svd(A,full_matrices=False) 

        #low rank approximation (keeping top k values)
        U_k = U[:,:k]
        S_k = S[:k]
        VT_k = VT[:k,:] 

        #reconstruct the new compressed matrix 
        A_k = U_k @ np.diag(S_k) @VT_k 

        time_taken = time.time() - start

        #calculate frobenius error 
        error = np.sqrt(np.sum((A-A_k)**2))

        #save the new compressed image 
        A_k = np.clip(A_k,0,255).astype(np.uint8)
        out_img = Image.fromarray(A_k,mode = 'L')
        out_img.save(f"{base_name}_numpy_{k}{extension}") 

        print(f"{k:<4}| {time_taken:<8.4f}| {error:.2f}") 

else: #colored image 

    print("\nColor Image")
    print("k   |  Time(s)  | R Error | G Error | B Error")
    print("----|---------|---------|---------|--------") 

    #divide into r,g,b channels 
    R = img_array[:,:,0].astype(float)
    G = img_array[:,:,1].astype(float)
    B = img_array[:,:,2].astype(float) 

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
        img_out = np.stack([R_k,G_k,B_k],axis = 2)
        img_out = np.clip(img_out,0,255).astype(np.uint8)
        out_img = Image.fromarray(img_out,mode="RGB")
        out_img.save(f"{base_name}_numpy_{k}{extension}")

        print(f"{k:<4}| {time_taken:<8.4f}| {err_r:<8.2f}| {err_g:<8.2f}| {err_b:<8.2f}")

print("\ncompleted")


    
