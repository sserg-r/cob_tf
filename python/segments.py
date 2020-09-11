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