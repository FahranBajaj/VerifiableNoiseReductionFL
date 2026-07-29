from logging import INFO

from flwr.common.logger import log
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def pca(weights: np.ndarray, ncomponents: int):
    """Apply PCA to weights submitted by clients
    
    Arguments:

    weights - model weights submitted by clients. Must be a 2-dimensional numpy array where each row corresponds to one client

    ncomponents - number of components to output for each client
    """
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-dimensional numpy array")
    
    pca = PCA(n_components=ncomponents, random_state = 42)
    return pca.fit_transform(weights)

#Binary tree clustering methods below
class Node(object):
    def __init__(self, index, lchild=None, rchild=None, counts=None):
        self.index = index  
        self.lchild = lchild  
        self.rchild = rchild  
        self.counts = counts  

    def __str__(self):
        return f"Index: {self.index}\nCount: {self.counts}\n Left child: {self.lchild}\n Right child: {self.rchild}" 

def leaves_under(node: Node):
    if node.counts == 1:
        return [node.index]
    return leaves_under(node.lchild) + leaves_under(node.rchild)

def benign_and_malicious(weights: np.ndarray, min_cluster_size: int):
    agglomer = AgglomerativeClustering(n_clusters=2, linkage='average')
    agglomer.fit(weights)
    children = agglomer.children_
    n_samples = agglomer.n_leaves_

    #for each internal vertex, count the number of leaves under it
    counts = np.zeros(agglomer.children_.shape[0])
    for i, merge in enumerate(children):
        total_leaves = 0
        #merge is an 2 element array containing the indices of the left and right children
        for child_idx in merge:
            if child_idx < n_samples:
                #child is a leaf
                total_leaves += 1
            else:
                #child is an internal node
                total_leaves += counts[child_idx - n_samples]
        counts[i] = total_leaves

    indices_to_node_objects = {}
    root = None

    #create node objects, which contain child pointers and leaf counts
    for internal_node_index in range(n_samples, n_samples + len(children)):
        child_idxs = children[internal_node_index - n_samples]
        count = counts[internal_node_index - n_samples]
        if child_idxs[0] < n_samples:
            lchild = Node(child_idxs[0], counts=1)
        else:
            lchild = indices_to_node_objects[child_idxs[0]]
            del indices_to_node_objects[child_idxs[0]]
    
        if child_idxs[1] < n_samples:
            rchild = Node(child_idxs[1], counts=1)
        else:
            rchild = indices_to_node_objects[child_idxs[1]]
            del indices_to_node_objects[child_idxs[1]]
    
        root = Node(internal_node_index, lchild, rchild, count)
        indices_to_node_objects[internal_node_index] = root

    #iterate past outliers until we get 2 clusters at least as large as min_cluster_size
    outliers = []
    original_root = root
    while root.rchild.counts < min_cluster_size or root.lchild.counts < min_cluster_size:
    
        if root.rchild.counts >= 2*min_cluster_size:
            outliers += leaves_under(root.lchild)
            root = root.rchild
    
        elif root.lchild.counts >= 2*min_cluster_size:
            outliers += leaves_under(root.rchild)
            root = root.lchild
    
        else:
            outliers += leaves_under(root)
    
        if len(outliers) >= int(n_samples/2):
            log(INFO, "Too many outliers to perform malicious client detection")
            root = None
            break
    
    #NOTE: FedDMC was unclear what to do if (a) there are too many outliers 
    # or (b) the left/right subtrees are the same size. In either case, I will
    #simply say that detection "fails" this round, and trust scores will not be
    # updated this round. Case (a) is handled above, (b) handled below
    
    benign, malicious = [], []
    if root:
        if root.rchild.counts < root.lchild.counts:
            malicious = leaves_under(root.rchild) + outliers
            benign = leaves_under(root.lchild)
        elif root.rchild.counts > root.lchild.counts:
            benign = leaves_under(root.rchild)
            malicious = leaves_under(root.lchild) + outliers
        else:
            log(INFO, "Cannot perform malicious client detection because cluster sizes match")

    return benign, malicious

def update_trust_scores(trust_scores: dict[int, float], benign_ids: list[int], malicious_ids: list[int], alpha: float):
    """Given lists of benign and malicious clients, updates the trust scores dictionary in-place
    
    Arguments:
    
    trust_scores: dictionary of trust scores to be updated. Maps node ids to their trust scores.

    benign_ids: list of node ids detected as benign in the most recent clustering

    malicious_ids: list of node ids detected as malicious in the most recent clustering

    alpha: hyperparameter. New trust scores computed as alpha*old_score + (1-alpha)*most_recent_detection
    """

    if not (alpha >= 0 and alpha <= 1):
        raise ValueError("alpha must be a real number between 0 and 1 (inclusive)")
    
    for id in benign_ids:
        trust_scores[id] = alpha*trust_scores[id] + (1-alpha)
    for id in malicious_ids:
        trust_scores[id] *= alpha
    

