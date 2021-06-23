from segmentation_models.base import Loss
from segmentation_models.base import functional as F

import segmentation_models as sm
import keras.backend as K
import tensorflow as tf 

class L1Loss(Loss):
    def __init__(self):
        super().__init__(name='l1')
    def __call__(self, gt, pr):
        return l1(gt, pr, **self.submodules)
        
class L2Loss(Loss):
    def __init__(self):
        super().__init__(name='l2')
    def __call__(self, gt, pr):
        return l2(gt, pr, **self.submodules)
    
class TverskyLoss(Loss):
    def __init__(self, alpha=0.5):
        super().__init__(name='tversky')
        self.alpha = alpha
    def __call__(self, gt, pr):
        return 1 - tversky(gt, pr, alpha=self.alpha, **self.submodules)

class TverskyFocalLoss(Loss):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__(name='tverskyfocal')
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, gt, pr):
        tverloss = 1 - tversky(gt, pr, alpha=self.alpha, **self.submodules)
        return K.pow(tverloss, self.gamma)

class WeightedBinaryCE(Loss):
    def __init__(self, alpha=0.5):
        super().__init__(name='weightedBCE')
        self.alpha = alpha
        
    def __call__(self, gt, pr):
        return weightedBCE(gt, pr, alpha=self.alpha, **self.submodules) 

class ComboLoss(Loss):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__(name='combo')
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, gt, pr):
        return alpha * modifiedCE(gt, pr, beta=self.beta, **self.submodules) - (1-alpha) * F.fscore(gt, pr, **self.submodules)

class SimpleSSLoss(Loss):
    def __init__(self, alpha=0.5):
        super().__init__(name='simplessl')
        self.alpha = alpha
        
    def __call__(self, gt, pr):
        return 1 - 0.5 * (self.alpha * sensitivity(gt, pr, **self.submodules) + (1-self.alpha) * specificity(gt, pr, **self.submodules))

def l1(gt, pr, **kwargs):
    backend = kwargs['backend']
    return backend.mean(backend.abs(gt-pr))

def l2(gt, pr, **kwargs):
    backend = kwargs['backend']
    return backend.mean(backend.square(gt-pr))

def tversky(gt, pr, alpha=0.5, **kwargs):
    backend = kwargs['backend']
    alpha = alpha     # penalty for false positive
    beta = 1 - alpha  # penalty for false negative
    smooth = 1e-6
    
    tp = backend.sum(gt * pr)
    fp = backend.sum((1-gt) * pr)
    fn = backend.sum(gt * (1-pr))
    
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    return tversky

def weightedBCE(gt, pr, alpha=0.5, **kwargs):
    backend = kwargs['backend']
    smooth = 1e-6
    
    return -tf.reduce_sum(alpha*gt*backend.log(pr+smooth) + (1-alpha)*(1-gt)*backend.log(1-pr+smooth))
    
def modifiedCE(gt, pr, beta=0.5, **kwargs):
    backend = kwargs['backend']
    beta = beta
    smooth=1e-6
    
    return -backend.reduce_sum(beta*(gt - backend.log(pr)) + (1-beta)*(1-gt)*backend.log(1-pr))
                     
def sensitivity(gt, pr, **kwargs):
    backend = kwargs['backend']
    smooth = 1e-6
    
    tp = backend.sum(gt * pr)
    tn = backend.sum((1-gt) * (1-pr))
    fp = backend.sum((1-gt) * pr)
    fn = backend.sum(gt * (1-pr))
    
    return tp / (tp + fn)
    
def specificity(gt, pr, **kwargs):
    backend = kwargs['backend']
    smooth = 1e-6
    
    tp = backend.sum(gt * pr)
    tn = backend.sum((1-gt) * (1-pr))
    fp = backend.sum((1-gt) * pr)
    fn = backend.sum(gt * (1-pr))
    
    return tn / (fp + tn)