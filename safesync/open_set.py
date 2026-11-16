"""Standard class-conditional Mahalanobis novelty on frozen features.

No novelty is claimed for this estimator. Operational known labels fit pooled
statistics; a held-out class never contributes fitting vectors. This does not
mean the frozen foundation model has never seen that visual concept.
"""
import numpy as np
from sklearn.covariance import LedoitWolf


class KnownClassDistance:
    def __init__(self,means,precision,labels,counts):
        self.means,self.precision,self.labels,self.counts=means,precision,labels,counts

    @classmethod
    def fit(cls,features,labels):
        features=np.asarray(features,np.float64);labels=np.asarray(labels)
        classes=[int(k) for k in np.unique(labels) if (labels==k).sum()>=5]
        if not classes:raise ValueError("no operational known class has five pure cells")
        means=np.stack([features[labels==k].mean(0) for k in classes])
        residual=np.concatenate([features[labels==k]-mu for k,mu in zip(classes,means)])
        cov=LedoitWolf(assume_centered=True).fit(residual)
        return cls(means,cov.precision_,np.array(classes),{int(k):int((labels==k).sum()) for k in classes})

    def score(self,features):
        x=np.asarray(features,np.float64)
        # Equivalent quadratic form avoids storing N×classes×384 scratch.
        px=x@self.precision
        base=np.sum(px*x,axis=1)
        center=np.sum((self.means@self.precision)*self.means,axis=1)
        distance=base[:,None]-2*px@self.means.T+center
        return np.maximum(distance.min(axis=1),0)/x.shape[1]

    def save(self,path):
        np.savez_compressed(path,means=self.means,precision=self.precision,labels=self.labels)
