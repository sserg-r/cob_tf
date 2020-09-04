import numpy as np
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