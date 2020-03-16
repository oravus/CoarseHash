import numpy as np
import os
from sklearn.decomposition import PCA

def addNoiseToFt(inFt,mu,var,fac):
    """
    Expected inFt of size n,d
    and mu and var of size d
    """
    inFt_n = inFt.copy()
    np.random.seed(349)
    for i1 in range(inFt.shape[1]):
        noiseVals = np.random.normal(mu[i1],fac*np.sqrt(var[i1]),inFt.shape[0])
        inFt_n[:,i1] += noiseVals
    return inFt_n

def loadVecs_py(path,N,D):
    dat = np.fromfile(path,"int32",count=N*(D+1))
    dat = dat.reshape([-1,(D+1)])[:,1:].view('float32')
    return dat

def loadD1B(readPath,dStr,w):

    N = 20000
    if dStr == "1M":
        N = 1000000
    elif dStr == "10M":
        N = 10000000
    ft1 = loadVecs_py(os.path.join(readPath,"Deep1B/base/base_00"),N,96)

    print("\t Performing sliding window average on D1B descriptors")
    ft1A = []
    for i1 in range(ft1.shape[0]):
        sIdx = max(0,i1-w//2)
        eIdx = min(ft1.shape[0],i1+w//2)
        fTmp = np.mean(ft1[sIdx:eIdx,:],axis=0)
        ft1A.append(fTmp)
    ft1 = np.array(ft1A).copy()
    print("\t Shape after dilution:",ft1.shape)
    del ft1A

    return ft1
        
def loadFAS(readPath):
    preCompPath = os.path.join(readPath,"FAS/FAS_uniSam5_all.npz")
    if os.path.exists(preCompPath):
        preLoad = np.load(preCompPath)
        ft1, ft2, gt, p1, p2, _, _ = [preLoad['arr_{}'.format(dIter)] for dIter in range(7)]

#     return ft1, ft2[1103:], gt[1103:], p1, p2[1103:]
    return ft1, ft2, gt, p1, p2

def calcChange(dataPath):
    ft1_f, ft2_f, gt_f, pos1_f, pos2_f = loadFAS(dataPath)
    
    print("Running PCA on FAS")
    pcaFAS = PCA(n_components=96,random_state=0)
    ft1_f = pcaFAS.fit_transform(ft1_f)
    ft2_f = pcaFAS.transform(ft2_f)
    
    print("\t Measuring pairwise descriptor 'change' in FAS using GT")    
    ft1_gt = ft1_f[gt_f]
    ftDiff = (ft1_gt - ft2_f)
    return ftDiff
    
def createDataset(dataPath,dStr,sigScale=1):
    """
    dStr from ["20K", "1M", "10M"]
    """
    
    print("Loading D1B dataset...")
    ft1_d = loadD1B(dataPath,dStr,w=40)

    if dStr=="20K":
        ft1_d = ft1_d[:10000,:]

    print("Running PCA on D1B")
    pcaD1B = PCA(n_components=ft1_d.shape[1],random_state=0)
    ft1_d = pcaD1B.fit_transform(ft1_d)

    print("Loading FAS dataset")
    ft1_f, ft2_f, gt_f, pos1_f, pos2_f = loadFAS(dataPath)

    if dStr=="20K":
        ft1_f = ft1_f[:10000,:]  
        ft2_f = ft2_f[:10000,:]

    print("Running PCA on FAS")
    pcaFAS = PCA(n_components=ft1_d.shape[1],random_state=0)
    ft1_f = pcaFAS.fit_transform(ft1_f)
    ft2_f = pcaFAS.transform(ft2_f)

    print("Re-scaling Variance of D1B using FAS data")
    ft1_d = np.std(ft1_f,axis=0)*ft1_d/np.std(ft1_d,axis=0)

    print("Computing a new version of D1B to be used as a query traverse")
    
    ftDiff = calcChange(dataPath)
    
    noiseVar = np.var(ftDiff,axis=0)
    noiseMean = np.mean(ftDiff,axis=0) 

    print("\t Incorporating the 'change' from FAS along with some noise")
    ft1_n = addNoiseToFt(ft1_d,noiseMean,noiseVar,sigScale)

    print("Concatenating the two datasets")
    ft1 = np.concatenate([ft1_d,ft1_f],axis=0)
    ft2 = np.concatenate([ft1_n,ft2_f],axis=0)
      
    del ft1_d, ft1_n, ft1_f, ft2_f
    
    return ft1, ft2
        

def main():
    
    dPath = "./"
    dataStr = "20K"
    sigMul = 1

    f1, f2 = createDataset(dPath,dataStr,sigScale=sigMul)
    print("Final Dataset Shape",f1.shape,f2.shape)
    np.savez("./locData_{}_queryNoiseMul_{}.npz".format(dataStr,sigMul),f1,f2)
    
    return

if __name__== "__main__":
    main()
