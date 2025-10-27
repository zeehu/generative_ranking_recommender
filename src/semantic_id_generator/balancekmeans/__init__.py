from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm
from .soft_dtw_cuda import SoftDTW
from sklearn.decomposition import PCA
import pickle

import torch

def auction_lap_half(job_and_worker_to_score, return_token_to_worker=True):
    """
    Solving the balanced linear assignment problem with auction algorithm.
    Arguments:
        - job_and_worker_to_score -> N x M euclidean distances between N data points and M cluster centers
    Returns:
        - assignment -> balanced assignment between jobs and workers
    """
    # 转换输入为半精度
    job_and_worker_to_score = job_and_worker_to_score.half()

    # 计算epsilon并确保数值稳定性
    with torch.no_grad():
        eps = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50
        eps = max(eps, torch.tensor(1e-04, dtype=torch.float16, device=job_and_worker_to_score.device))
    
    assert not torch.isnan(job_and_worker_to_score).any()
    if torch.isnan(job_and_worker_to_score).any():
        raise Exception("NaN distance")
    # 转置矩阵并释放原矩阵
    worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous()
    del job_and_worker_to_score
    torch.cuda.empty_cache() if worker_and_job_to_score.is_cuda else None
    
    num_workers, num_jobs = worker_and_job_to_score.size()
    jobs_per_worker = num_jobs // num_workers
    
    # 使用半精度初始化变量
    value = worker_and_job_to_score.clone()
    bids = torch.zeros((num_workers, num_jobs), 
                      dtype=torch.float16,
                      device=worker_and_job_to_score.device,
                      requires_grad=False)
    counter = 0
    index = None
    cost = torch.zeros((1,num_jobs,),
                        dtype=torch.float16,
                        device=worker_and_job_to_score.device,
                        requires_grad=False)
    while True:
        # 分批计算topk减少显存峰值
        chunk_size = min(1000, num_workers)  # 根据显存调整
        top_values_list, top_index_list = [], []
        
        for i in range(0, num_workers, chunk_size):
            chunk = value[i:i+chunk_size]
            tv, ti = chunk.topk(jobs_per_worker + 1, dim=1)
            top_values_list.append(tv)
            top_index_list.append(ti)
            del chunk
            
        top_values = torch.cat(top_values_list)
        top_index = torch.cat(top_index_list)
        del top_values_list, top_index_list
        
        # 计算bid增量并释放中间变量
        bid_increments = (top_values[:,:-1] - top_values[:,-1:]).add_(eps)
        del top_values

        assert bid_increments.size() == (num_workers, jobs_per_worker)
        # 使用原地操作更新bids
        with torch.no_grad():
            bids.zero_()
            bids.scatter_(dim=1, index=top_index[:,:-1], src=bid_increments.to(bids.dtype))
            
            # 特殊处理保留bid
            if counter < 100 and index is not None:
                bids.view(-1)[index] = eps
            if counter > 1000:
                bids.view(-1)[jobs_without_bidder] = eps
                
        del bid_increments, top_index

        # 查找有bid的job
        with torch.no_grad():
            jobs_with_bidder = (bids > 0).any(0).nonzero(as_tuple=False).squeeze(1)
            jobs_without_bidder = (bids == 0).all(0).nonzero(as_tuple=False).squeeze(1)

        # 分批查找最高bid
        chunk_size = min(1000, jobs_with_bidder.size(0))
        high_bids_list, high_bidders_list = [], []
        
        for i in range(0, jobs_with_bidder.size(0), chunk_size):
            chunk = jobs_with_bidder[i:i+chunk_size]
            hb, hbr = bids[:, chunk].max(dim=0)
            high_bids_list.append(hb)
            high_bidders_list.append(hbr)
            
        high_bids = torch.cat(high_bids_list)
        high_bidders = torch.cat(high_bidders_list)
        del high_bids_list, high_bidders_list

        # 检查是否所有job都有bid
        if high_bidders.size(0) == num_jobs:
            break
            
        # 更新cost和value
        with torch.no_grad():
            cost[:, jobs_with_bidder] += high_bids
            value = worker_and_job_to_score - cost
            
            # 确保热门item保留
            index = (high_bidders * num_jobs) + jobs_with_bidder
            value.view(-1)[index] = worker_and_job_to_score.view(-1)[index]
            
        counter += 1
        torch.cuda.empty_cache() if value.is_cuda else None
    

    # 返回结果并清理显存
    if return_token_to_worker:
        return high_bidders
    _, sorting = torch.sort(high_bidders)
    assignment = jobs_with_bidder[sorting]
    assert len(assignment.unique()) == num_jobs

    # 释放所有中间变量
    del worker_and_job_to_score, value, bids, cost, high_bids, high_bidders, jobs_with_bidder
    torch.cuda.empty_cache() if worker_and_job_to_score.is_cuda else None
    
    return assignment.view(-1)
    
def auction_lap_full(job_and_worker_to_score, return_token_to_worker=True):
    """
    Solving the balanced linear assignment problem with auction algorithm.
    Arguments:
        - job_and_worker_to_score -> N x M euclidean distances between N data points and M cluster centers
    Returns:
        - assignment -> balanced assignment between jobs and workers
    """
    eps = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50
    eps.clamp_min_(1e-04)
    assert not torch.isnan(job_and_worker_to_score).any()
    if torch.isnan(job_and_worker_to_score).any():
        raise Exception("NaN distance")
    worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous()
    num_workers, num_jobs = worker_and_job_to_score.size()
    jobs_per_worker = num_jobs // num_workers
    value = torch.clone(worker_and_job_to_score)
    bids = torch.zeros((num_workers, num_jobs),
                        dtype=worker_and_job_to_score.dtype,
                        device=worker_and_job_to_score.device,
                        requires_grad=False)
    counter = 0
    index = None
    cost = torch.zeros((1,num_jobs,),
                        dtype=worker_and_job_to_score.dtype,
                        device=worker_and_job_to_score.device,
                        requires_grad=False)
    while True:
        top_values, top_index = value.topk(jobs_per_worker + 1, dim=1)
        # Each worker bids the difference in value between that job and the k+1th job
        bid_increments = top_values[:,:-1] - top_values[:,-1:]  + eps
        assert bid_increments.size() == (num_workers, jobs_per_worker)
        bids.zero_()
        bids.scatter_(dim=1, index=top_index[:,:-1], src=bid_increments)

        if counter < 100 and index is not None:
            # If we were successful on the last round, put in a minimal bid to retain
            # the job only if noone else bids. After N iterations, keep it anyway.
            bids.view(-1)[index] = eps
            # 
        if counter > 1000:
            bids.view(-1)[jobs_without_bidder] = eps
        # Find jobs that was a top choice for some worker
        jobs_with_bidder = (bids > 0).any(0).nonzero(as_tuple=False).squeeze(1)
        jobs_without_bidder = (bids == 0).all(0).nonzero(as_tuple=False).squeeze(1)

        # Find the highest bidding worker per job
        high_bids, high_bidders = bids[:, jobs_with_bidder].max(dim=0)
        if high_bidders.size(0) == num_jobs:
            # All jobs were bid for
            break
        
        # Make popular items more expensive
        cost[:, jobs_with_bidder] += high_bids
        value = worker_and_job_to_score - cost

        # # Hack to make sure that this item will be in the winning worker's top-k next time
        index = (high_bidders * num_jobs) + jobs_with_bidder
        value.view(-1)[index] = worker_and_job_to_score.view(-1)[index]
        counter += 1
    

    if return_token_to_worker:
        return high_bidders
    _, sorting = torch.sort(high_bidders)
    assignment = jobs_with_bidder[sorting]
    assert len(assignment.unique()) == num_jobs

    return assignment.view(-1)



def batchify(a, n=2):
    for i in np.array_split(a, n, axis=0):
        yield i

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.shape[0] - 1))
    return t.kthvalue(k, dim=0).values
    

class KMeans(object):
    def __init__(self, n_clusters=None, cluster_centers=None, device=torch.device('cpu'), balanced=False):
        self.n_clusters = n_clusters
        self.cluster_centers = cluster_centers
        self.device = device
        self.balanced = balanced
    
    @classmethod
    def load(cls, path_to_file):
        with open(path_to_file, 'rb') as f:
            saved = pickle.load(f)
        return cls(saved['n_clusters'], saved['cluster_centers'], torch.device('cpu'), saved['balanced'])
    
    def save(self, path_to_file):
        with open(path_to_file, 'wb+') as f :
            pickle.dump(self.__dict__, f)

    def initialize(self, X):
        """
        initialize cluster centers
        :param X: (torch.tensor) matrix
        :param n_clusters: (int) number of clusters
        :return: (np.array) initial state
        """
        num_samples = len(X)
        indices = np.random.choice(num_samples, self.n_clusters, replace=False)
        initial_state = X[indices]
        return initial_state
    
    # 最大程度上减少没有填充数据的情况
    def fit_by_min_loss(self, 
                        X, 
                        target_nodes_num, # 目标的节点数, 达不成则认为是有损失的
                        distance='euclidean', 
                        tol=1e-3, 
                        tqdm_flag=True, 
                        iter_limit=0, 
                        gamma_for_soft_dtw=0.001, 
                        half=False,  # 是否开启半精度计算
                        online=False,
                        iter_k=None):
        
        if tqdm_flag:
            print(f'running k-means on {self.device}..')
        
        if distance == 'euclidean':
            if half:
                pairwise_distance_function = partial(pairwise_distance_half, device=self.device, batch_size=100000)  # 可调整批次大小
            else:
                pairwise_distance_function = partial(pairwise_distance_full, device=self.device, batch_size=100000)
        elif distance == 'cosine':
            pairwise_distance_function = partial(pairwise_cosine, device=self.device, batch_size=100000)  # 可调整批次大小
        elif distance == 'soft_dtw':
            sdtw = SoftDTW(use_cuda=self.device.type == 'cuda', gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=self.device)
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()

        # transfer to device
        X = X.to(self.device)

        # initialize
        if not online or (online and iter_k == 0):
            self.cluster_centers = self.initialize(X)
            

        iteration = 0
        if tqdm_flag:
            tqdm_meter = tqdm(desc='[running kmeans]')
        done=False
        min_loss_cluster_centers = None # 记录截至到当前为止最小loss的cluster_centers
        min_loss = float('inf') # 记录最小loss
        while True:
            if iteration > 0 and iteration % 10 == 0: # 迭代太多次并不会直接降低loss, 因此, 每10次迭代就随机初始化
                self.cluster_centers = self.initialize(X) 

            distance_matrix = pairwise_distance_function(X, self.cluster_centers)
            if self.balanced:
                cluster_assignments = auction_lap_half(-distance_matrix)
            else:
                cluster_assignments = torch.argmin(distance_matrix, dim=1)
            
            initial_state_pre = self.cluster_centers.clone()
            for index in range(self.n_clusters):
                selected = torch.nonzero(cluster_assignments == index).squeeze().to(self.device)

                selected = torch.index_select(X, 0, selected)

                # https://github.com/subhadarship/kmeans_pytorch/issues/16
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]
                
                self.cluster_centers[index] = selected.mean(dim=0)
            
            # 计算loss
            cur_distance_matrix = pairwise_distance_function(X, self.cluster_centers)
            min_cluster_assignments = torch.argmin(cur_distance_matrix, dim=1)
            min_cluster_assignments = torch.bincount(min_cluster_assignments, minlength=self.n_clusters)
            
            # 遍历每种数量, 如果小于目标节点数, loss则等于目标节点数减去当前数量的差值
            # 如果大于目标节点数, 则loss为0
            cur_loss = 0
            for i in range(self.n_clusters):
                if min_cluster_assignments[i] > target_nodes_num:
                    cur_loss += min_cluster_assignments[i] - target_nodes_num
            print(f"cur_loss: {cur_loss}")
            if cur_loss <= min_loss:
                min_loss = cur_loss
                min_loss_cluster_centers = self.cluster_centers.clone()
                print(f"cur min_cluster_assignments: {min_cluster_assignments}")

            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.cluster_centers - initial_state_pre) ** 2, dim=1)
                ))

            # increment iteration
            iteration = iteration + 1
            
            # update tqdm meter
            if tqdm_flag:
                tqdm_meter.set_postfix(
                    iteration=f'{iteration}',
                    center_shift=f'{center_shift ** 2:0.6f}',
                    tol=f'{tol:0.6f}'
                )
                tqdm_meter.update()
            if center_shift ** 2 < tol:
                break
            if iter_limit != 0 and iteration >= iter_limit:
                break

        self.cluster_centers = min_loss_cluster_centers
        return 
        
    
    def fit(
            self,
            X,
            distance='euclidean',
            tol=1e-3,
            tqdm_flag=True,
            iter_limit=0,
            gamma_for_soft_dtw=0.001,
            half=False,  # 是否开启半精度计算
            online=False,
            iter_k=None
    ):
        """
        perform kmeans
        :param X: (torch.tensor) matrix
        :param n_clusters: (int) number of clusters
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param tol: (float) threshold [default: 0.0001]
        :param device: (torch.device) device [default: cpu]
        :param tqdm_flag: Allows to turn logs on and off
        :param iter_limit: hard limit for max number of iterations
        :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
        :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
        """
        if tqdm_flag:
            print(f'running k-means on {self.device}..')
        
        if distance == 'euclidean':
            if half:
                pairwise_distance_function = partial(pairwise_distance_half, device=self.device, batch_size=100000)  # 可调整批次大小
            else:
                pairwise_distance_function = partial(pairwise_distance_full, device=self.device, batch_size=100000)
        elif distance == 'cosine':
            pairwise_distance_function = partial(pairwise_cosine, device=self.device, batch_size=100000)  # 可调整批次大小
        elif distance == 'soft_dtw':
            sdtw = SoftDTW(use_cuda=self.device.type == 'cuda', gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=self.device)
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()

        # transfer to device
        X = X.to(self.device)

        # initialize
        if not online or (online and iter_k == 0):
            self.cluster_centers = self.initialize(X)
            

        iteration = 0
        if tqdm_flag:
            tqdm_meter = tqdm(desc='[running kmeans]')
        done=False
        while True:
            distance_matrix = pairwise_distance_function(X, self.cluster_centers)
            if self.balanced:
                # 打印shape信息
                # print(f"输入数据X的shape: {X.shape}")  # 格式为(样本数, 特征维度)
                # print(f"聚类中心shape: {self.cluster_centers.shape}")  # 格式为(聚类数, 特征维度)
                cluster_assignments = auction_lap_half(-distance_matrix)
            else:
                cluster_assignments = torch.argmin(distance_matrix, dim=1)
            
            initial_state_pre = self.cluster_centers.clone()
            for index in range(self.n_clusters):
                selected = torch.nonzero(cluster_assignments == index).squeeze().to(self.device)

                selected = torch.index_select(X, 0, selected)

                # https://github.com/subhadarship/kmeans_pytorch/issues/16
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]
                
                self.cluster_centers[index] = selected.mean(dim=0)
            
            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.cluster_centers - initial_state_pre) ** 2, dim=1)
                ))

            # increment iteration
            iteration = iteration + 1
            
            # update tqdm meter
            if tqdm_flag:
                tqdm_meter.set_postfix(
                    iteration=f'{iteration}',
                    center_shift=f'{center_shift ** 2:0.6f}',
                    tol=f'{tol:0.6f}'
                )
                tqdm_meter.update()
            if center_shift ** 2 < tol:
                break
            if iter_limit != 0 and iteration >= iter_limit:
                break
        return cluster_assignments.cpu()


    def plot(self, data, labels, plot_file):
        if self.cluster_centers is None:
            raise Exception("Fit the KMeans object first before plotting!")
        plt.figure(figsize=(4, 3), dpi=160)
        pca = PCA(n_components=2)
        master = np.concatenate([data, self.cluster_centers], 0)
        pca = pca.fit(master)
        data = pca.transform(data)
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        cluster_centers = pca.transform(self.cluster_centers)
        plt.scatter(
            self.cluster_centers[:, 0], self.cluster_centers[:, 1],
            c='white',
            alpha=0.6,
            edgecolors='black',
            linewidths=2
        )
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)


    def predict(
            self,
            X,
            distance='euclidean',
            gamma_for_soft_dtw=0.001,
            tqdm_flag=False,
            return_distances=False,
            balanced=False
    ):
        """
        predict using cluster centers
        :param X: (torch.tensor) matrix
        :param cluster_centers: (torch.tensor) cluster centers
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param device: (torch.device) device [default: 'cpu']
        :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
        :return: (torch.tensor) cluster ids
        """

        if distance == 'euclidean':
            pairwise_distance_function = partial(pairwise_distance_full, device=self.device, batch_size=100000)
        elif distance == 'cosine':
            pairwise_distance_function = partial(pairwise_cosine, device=self.device, batch_size=100000)  # 可调整批次大小
        elif distance == 'soft_dtw':
            sdtw = SoftDTW(use_cuda=self.device.type == 'cuda', gamma=gamma_for_soft_dtw)
            pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=self.device)
        else:
            raise NotImplementedError

        # convert to float
        X = X.float()
        # transfer to device
        if self.device != torch.device('cpu'):
            X = X.to(self.device)
        if balanced:
            distance_matrix = pairwise_distance_function(X, self.cluster_centers)
            cluster_assignments = auction_lap_full(-distance_matrix)
        else:
            distance_matrix = pairwise_distance_function(X, self.cluster_centers)
            cluster_assignments = torch.argmin(distance_matrix, dim=1 if len(distance_matrix.shape) > 1 else 0)
            if len(distance_matrix.shape) == 1:
                cluster_assignments = cluster_assignments.unsqueeze(0)
        if return_distances:
            return cluster_assignments.cpu(),distance_matrix
        else:
            return cluster_assignments.cpu()

def pairwise_distance_half(data1, data2, device=torch.device('cpu'), batch_size=20000):
    """
    全半精度优化的距离计算
    改进点:
    1. 输入输出全程使用FP16
    2. 增大batch_size(因内存占用减半)
    3. 添加数值稳定保护
    
    参数:
        data1: 样本数据 (N x D), 自动转为FP16
        data2: 聚类中心 (K x D), 自动转为FP16
        batch_size: 可增大至原2倍
    返回:
        FP16精度的距离矩阵 (N x K)
    """
    # 自动调整batch_size避免OOM
    free_mem = torch.cuda.mem_get_info()[0] if device.type == 'cuda' else float('inf')
    safe_batch_size = min(batch_size, int(free_mem / (data1.shape[1] * 2 * 2)))  # 2个FP16张量
    batch_size = max(1, safe_batch_size)

    # 转换到半精度
    data1 = data1.half().to(device)
    data2 = data2.half().to(device)
    
    # 预分配半精度结果矩阵
    distances = torch.zeros((len(data1), len(data2)), dtype=torch.float16, device=device)
    
    for i in range(0, len(data1), batch_size):
        batch = data1[i:i+batch_size]
        # 计算距离(保持FP16)
        batch_dist = torch.cdist(batch.unsqueeze(0), data2.unsqueeze(0)).squeeze(0)
        # 添加微小值防止下溢
        batch_dist = torch.clamp(batch_dist, min=1e-5)
        distances[i:i+batch_size] = batch_dist
        
        del batch, batch_dist
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return distances

def pairwise_distance_full(data1, data2, device=torch.device('cpu'), batch_size=10000):
    """
    分批计算欧氏距离矩阵，避免OOM
    参数:
        data1: 样本数据 (N x D)
        data2: 聚类中心 (K x D)
        batch_size: 每批处理的样本数
    返回:
        距离矩阵 (N x K)
    """
    data1 = data1.float().to(device)
    data2 = data2.float().to(device)
    
    # 预分配结果矩阵
    distances = torch.zeros((len(data1), len(data2)), device=device)
    
    # 分批处理样本数据
    for i in range(0, len(data1), batch_size):
        batch = data1[i:i+batch_size]
        # 计算批数据与所有中心的距离 (广播机制)
        batch_dist = torch.cdist(batch.unsqueeze(0), data2.unsqueeze(0)).squeeze(0)
        distances[i:i+batch_size] = batch_dist
        
        # 清理中间变量
        del batch, batch_dist
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return distances

# def pairwise_cosine(data1, data2, device=torch.device('cpu')):
#     # transfer to device
#     data1, data2 = data1.to(device), data2.to(device)

#     # N*1*M
#     A = data1.unsqueeze(dim=1)

#     # 1*N*M
#     B = data2.unsqueeze(dim=0)

#     # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
#     A_normalized = A / A.norm(dim=-1, keepdim=True)
#     B_normalized = B / B.norm(dim=-1, keepdim=True)

#     cosine = A_normalized * B_normalized

#     # return N*N matrix for pairwise distance
#     cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
#     return cosine_dis

def pairwise_cosine(data1, data2, device=torch.device('cpu'), batch_size=1000):
    """
    分批计算余弦距离，内存优化版
    参数:
        data1: 样本数据 (N x D)
        data2: 聚类中心 (K x D)
        batch_size: 每批处理的样本数
    返回:
        余弦距离矩阵 (N x K)
    """
    data1 = data1.float().to(device)
    data2 = data2.float().to(device)
    
    # 归一化聚类中心 (只需计算一次)
    centers_norm = data2 / data2.norm(dim=1, keepdim=True)
    distances = torch.zeros((len(data1), len(data2)), device=device)
    
    # 分批处理样本数据
    for i in range(0, len(data1), batch_size):
        batch = data1[i:i+batch_size]
        # 归一化批数据
        batch_norm = batch / batch.norm(dim=1, keepdim=True)
        # 矩阵乘法计算相似度
        sim = torch.mm(batch_norm, centers_norm.t())
        distances[i:i+batch_size] = 1 - sim
        
        # 清理中间变量
        del batch, batch_norm, sim
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return distances

def pairwise_soft_dtw(data1, data2, sdtw=None, device=torch.device('cpu')):
    if sdtw is None:
        raise ValueError('sdtw is None - initialize it with SoftDTW')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # (batch_size, seq_len, feature_dim=1)
    A = data1.unsqueeze(dim=2)

    # (cluster_size, seq_len, feature_dim=1)
    B = data2.unsqueeze(dim=2)

    distances = []
    for b in B:
        # (1, seq_len, 1)
        b = b.unsqueeze(dim=0)
        A, b = torch.broadcast_tensors(A, b)
        # (batch_size, 1)
        sdtw_distance = sdtw(b, A).view(-1, 1)
        distances.append(sdtw_distance)

    # (batch_size, cluster_size)
    dis = torch.cat(distances, dim=1)
    return dis
