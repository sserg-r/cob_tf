import numpy as np
import math
def interpolate_angles(I):
    import numpy as np
    '''
    Interpolating orientations strength maps to 1 orientation map
    
    I - np array of orientation strength maps size of [n, h,w] n -numbers of maps
    
    '''
    
    numClusters=I.shape[0]
    [Max1,ind1] = [np.max(I,axis=0), np.argmax(I,axis=0)]

    for i in range(numClusters):
        I[i,:,:]=(I[i,:,:]!=Max1)*I[i,:,:]

    [Max2,ind2] = [np.max(I,axis=0), np.argmax(I,axis=0)]    

    ind1=ind1+1
    ind2=ind2+1
    ind1[(ind1*ind2>0)*(ind2-ind1==numClusters-1)]=numClusters+1
    ind2[(ind1*ind2>0)*(ind1-ind2==numClusters-1)]=numClusters+1
    ind2[(ind1*ind2>0)*abs(ind1-ind2)>1]=0

    ab=np.vstack((ind1[None,:],ind2[None,:]))
    w1w2=np.vstack((Max1[None,:],Max2[None,:]))

    ind1[Max1<0.01]=0
    ind2[Max2<0.01]=0

    O=(np.sum(ab*w1w2,axis=0)/np.sum((ab>0)*w1w2,axis=0)*((ind1*ind2==0)+(abs(ind1-ind2)==1))-1)*np.pi/numClusters

    O[Max1<0.5]=-1
    O[O<0] = np.pi*np.random.rand(np.sum(O<0))
    return O

def get_l_p(L):
    '''calculate division parameters of line of size L to overlapping segments,
    return dict of {l:segment length, p: overlapping, n: segment counts, res: overage of L}'''
    import math
    if L<=500: return {'l': L, 'p': 0, 'n': 1, 'res': 0}    
#     assert L>500, 'L must be greater then 500'
    p0=100
    l0=500
    n=math.ceil((L-p0)/(l0-p0))
    L_hat=l0*n-(n-1)*p0
    dL=L_hat-L    
    diff=dL
    a0=0
    b0=0
    for a in range (dL//n,-1,-1):
        dl_res=dL%(n*a) if a>0 else dL
        b=dl_res//(n-1)
        
        if dl_res%(n-1)==0:
            a0=a
            b0=b
            diff=0
            break        
        
        if dl_res%(n-1)<diff:        
            diff=dl_res%(n-1)
            a0=a
            b0=b               
    
    return {'l': l0-a0,'p':b0+p0, 'n': n, 'res':diff}

def batcherize (image):
    '''divide a 3-dims image with the sum of 2 dimensions more than 1000 into n patches with the size of h or w no more than 500'''
    h,w,_=image.shape    
    out_batch=[]    
    if h+w<=1000:
        out_batch.append(image)
    else:
        par_h=get_l_p(h)
        par_w=get_l_p(w)
        tim_padded=np.pad(image, ((0,par_h['res']),(0,par_w['res']),(0,0)),mode='symmetric')
        step_h=par_h['l']-par_h['p']
        step_w=par_w['l']-par_w['p']        
        for ih in range(par_h['n']):
            for iw in range( par_w['n']):  
                out_batch.append(tim_padded[ih*step_h:ih*step_h+par_h['l'],iw*step_w:iw*step_w+par_w['l'],:])
    
    overlapping=(par_h['p'], par_w['p'])
    res=(par_h['res'], par_w['res'])
    n=(par_h['n'],par_w['n'])
    return out_batch, overlapping, n, res

def merger (batch, overlapping, n, res):
    
    '''merges batch of images in 1 image taking into account number of patches in h & w dims, 
    image's overlapping and trim resulting image right and bottom according to the res'''
    
    N, h,w,b=batch.shape
    
    step_h=h-overlapping[0]
    step_w=w-overlapping[1]
    
    
    L_h=h*n[0]-(n[0]-1)*overlapping[0]
    L_w=w*n[1]-(n[1]-1)*overlapping[1]
    
    out=np.zeros((L_h,L_w,b))
    ittr=iter(batch)

    for ih in range(n[0]):
        shift_h=overlapping[0]//2 if ih!=0 else 0 
        for iw in range( n[1]):
            shift_w=overlapping[1]//2 if iw!=0 else 0                         
            out[ih*step_h+shift_h:ih*step_h+h,iw*step_w+shift_w:iw*step_w+w,:]=next(ittr)[shift_h:,shift_w:,:]
            
    return out[:L_h-res[0],:L_w-res[1],:]

# def get_l_p(L):
#     import math
#     p0=100
#     l0=500
#     n=math.ceil((L-p0)/(l0-p0))
#     L_hat=l0*n-(n-1)*p0
#     dL=L_hat-L
#     cnt=0
#     while dL%n:
#         cnt+=1
#         dL-=(n-1)
#     return cnt+p0, l0-dL//n, n