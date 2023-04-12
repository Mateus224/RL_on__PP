import torch
import torch.nn.functional as F
#from pointnet2_ops import pointnet2_utils
#from pytorch3d.ops import sample_farthest_points




def cal_loss(pred, ground_truth, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    ground_truth = ground_truth.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, ground_truth.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, ground_truth, reduction='mean')

    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query.

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    
    Output:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.

    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    
    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx



def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    
    Output:
        new_points:, indexed points data, [B, S, C]
    """
    
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_ball_group(s, radius, n, coords, features):
    """
    Sampling by FPS and grouping by ball query.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by ball query
        n[int]: fix number of points in ball neighbor
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # ball_query grouping
    idx = query_ball_point(radius, n, coords, new_coords)              # [B, s, n]
    
    grouped_features = index_points(features, idx)                     # [B, s, n, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, n, D]

    # Concat, my be different in many networks
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, n, 1)], dim=-1)  # [B, s, n, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, n, 2D]


def sample_and_knn_group(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """

    batch_size = coords.shape[0]
    coords = coords.contiguous()
    #feature_k1 = features.narrow(2,0,64)
    #feature_k2 = features.narrow(2,64,64)
    # FPS sampling
    #fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    #new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    #new_features = index_points(features, fps_idx)                     # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, coords)                                       # [B, s, k]
    #grouped_features = index_points(features, idx)       
    #grouped_features1 = features1
    grouped_features = index_points(features, idx)
    #grouped_features2 = index_points(feature_k2, idx)
    #print(grouped_features1.shape,grouped_features2.shape,'grouped_features2')
    #grouped_features3 = index_points(features3, idx)
    #grouped_features4 = index_points(features4, idx)                             # [B, s, k, D]
    #grouped_features = torch.cat([grouped_features1, grouped_features2, grouped_features3, grouped_features4], dim=3)  
    # Matrix sub
    grouped_features_norm = grouped_features - features.view(batch_size, s, 1, -1)  # [B, s, k, D]
    aggregated_features = torch.cat([grouped_features_norm, features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1) 
    # Concat
    #aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]
    return  coords, aggregated_features#1, grouped_features2#, grouped_features3, grouped_features4  # [B, s, 3], [B, s, k, 2D]

def sample_and_knn_group1(s, k, coords, features, t):
    """
    Sampling by FPS and grouping by KNN.
    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()
    # FPS sampling
    batch_FPSPoints= np.empty([batch_size,s,3])
    batch_index=np.empty([batch_size,s])
    #print(s, coords.shape)
    for i in range(batch_size):
        fps = FPS(coords[i,:,:].cpu(), s)
        fps.fit()
        batch_FPSPoints[i,:,:]=torch.from_numpy(fps.get_selected_pts())
        #print(batch_FPSPoints[i,:,:].shape, coords[i,:,:].cpu().numpy().shape)
        for w in range(s):
            batch_index[i,w]=np.where(np.all(batch_FPSPoints[i,w,:]==coords[i,:,:].cpu().numpy(), axis=1))[0][0]
                    #print(s_)
    #fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    #batch_index=batch_index#.long()#torch.from_numpy(batch_index).long()
    new_coords = index_points(coords, batch_index)                      # [B, s, 3]
    #print(new_coords.shape)
    new_features = index_points(features, batch_index)                     # [B, s, D]
    # K-nn grouping
    #new_coords=batch_FPSPoints#.to(device='cuda')
    idx = knn_point(k, coords, new_coords)                                           # [B, s, k]
    grouped_features = index_points(features, idx)                                      # [B, s, k, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    if t==False:
        aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]

        return new_coords, aggregated_features 
    else:
        return new_coords, grouped_features_norm 

import numpy as np

class FPS:
    def __init__(self, pcd_xyz, n_samples):
        self.n_samples = n_samples
        self.pcd_xyz = pcd_xyz
        self.n_pts = pcd_xyz.shape[0]
        self.dim = pcd_xyz.shape[1]
        self.selected_pts = None
        self.selected_pts_expanded = np.zeros(shape=(n_samples, 1, self.dim))
        self.remaining_pts = np.copy(pcd_xyz)

        self.grouping_radius = None
        self.dist_pts_to_selected = None  # Iteratively updated in step(). Finally re-used in group()
        self.labels = None

        # Random pick a start
        self.start_idx = np.random.randint(low=0, high=self.n_pts - 1)
        self.selected_pts_expanded[0] = self.remaining_pts[self.start_idx]
        self.n_selected_pts = 1

    def get_selected_pts(self):
        self.selected_pts = np.squeeze(self.selected_pts_expanded, axis=1)
        return self.selected_pts

    def step(self):
        if self.n_selected_pts < self.n_samples:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts, self.selected_pts_expanded[:self.n_selected_pts]).T
            dist_pts_to_selected_min = np.min(self.dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[res_selected_idx]

            self.n_selected_pts += 1
        else:
            print("Got enough number samples")


    def fit(self):
        for _ in range(1, self.n_samples):
            self.step()
        return self.get_selected_pts()

    def group(self, radius):
        self.grouping_radius = radius   # the grouping radius is not actually used
        dists = self.dist_pts_to_selected

        # Ignore the "points"-"selected" relations if it's larger than the radius
        dists = np.where(dists > radius, dists+1000000*radius, dists)

        # Find the relation with the smallest distance.
        # NOTE: the smallest distance may still larger than the radius.
        self.labels = np.argmin(dists, axis=1)
        return self.labels


    @staticmethod
    def __distance__(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)

class Logger():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def sample_and_knn_group_(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.
    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = sample_farthest_points(coords,None, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, new_coords)                                              # [B, s, k]
    grouped_features = index_points(features, idx)                                      # [B, s, k, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]

    return new_coords, aggregated_features, fps_idx  # [B, s, 3], [B, s, k, 2D]




from random import randint
from typing import List, Optional, Tuple, Union

import torch
#from pytorch3d import _C

#from .utils import masked_gather



def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Check dtypes are correct and convert if necessary
    if not (points.dtype == torch.float32):
        points = points.to(torch.float32)
    if not (lengths.dtype == torch.int64):
        lengths = lengths.to(torch.int64)
    if not (K.dtype == torch.int64):
        K = K.to(torch.int64)

    # Generate the starting indices for sampling
    start_idxs = torch.zeros_like(lengths)
    if random_start_point:
        for n in range(N):
            # pyre-fixme[6]: For 1st param expected `int` but got `Tensor`.
            start_idxs[n] = torch.randint(high=lengths[n], size=(1,)).item()

    with torch.no_grad():
        # pyre-fixme[16]: `pytorch3d_._C` has no attribute `sample_farthest_points`.
        idx = _C.sample_farthest_points(points, lengths, K, start_idxs)
    #sampled_points = masked_gather(points, idx)

    return idx


if __name__ == '__main__':
    points = torch.rand(32, 1024, 3).to('cuda')
    features = torch.rand(32, 1024, 128).to('cuda')
    new_points, new_features = sample_and_knn_group(512, 32, points, features)
    print(new_points.size())
    print(new_features.size())
