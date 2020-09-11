import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from train_embed.utils_experiment import load_model as load_embed_model
from train_filter.utils_experiment import load_model as load_filter_model
from build_graphs import filter_one_neighborhood, EdgeData, predict_pairs,  apply_filter, load_event, get_emb_neighbors, my_collate

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'

def get_one_emb_eff_purity(index, hits, truth, emb_neighbors):
    vol = hits[['volume_id', 'layer_id']].values.T
    hit = hits.iloc[index]
    pid = truth.iloc[index]['particle_id']
    
    # get true neighbors based on particle id
    if pid == 0:
        true_neighbors = []
    else:
        hit_idx = truth[truth['particle_id']==pid]['hit_id']
        true_hits = hits[hits['hit_id'].isin(hit_idx) & (hits['hit_id'] != hit['hit_id'])]
        neighbors = true_hits.index.values.astype(int)
        true_neighbors = filter_one_neighborhood(hit['volume_id'], hit['layer_id'], neighbors, vol[0], vol[1])
    
    emb_neighbors = filter_one_neighborhood(hit['volume_id'], hit['layer_id'], emb_neighbors[index], vol[0], vol[1])
    
    # calculate purity and eff
    n_true_neighbors = sum(map(lambda n : n in true_neighbors, emb_neighbors))
    purity = n_true_neighbors / len(emb_neighbors) if len(emb_neighbors) > 0 else None
    efficiency = n_true_neighbors / len(true_neighbors) if len(true_neighbors) > 0 else None
    return purity, efficiency

def get_emb_eff_purity(hits, truth, emb_neighbors):
    n_iter = len(hits)
    purity = 0
    efficiency = 0

    for i in range(n_iter):
        p, eff = get_one_emb_eff_purity(i, hits, truth, emb_neighbors)
        if p: purity += p
        if eff: efficiency += eff
        if i % (n_iter//10) == 0:
            print(int((i+1) / (n_iter//100)),"% done")

    return purity/n_iter, efficiency/n_iter

def get_filter_eff_purity(hits, truth, neighbors):
    vol = hits[['volume_id', 'layer_id']].values.T

    batch_size = 64
    num_workers = 12 if DEVICE=='cuda' else 0
    dataset = EdgeData(hits[feature_names].values, vol, neighbors)
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        num_workers = num_workers,
                        collate_fn = my_collate)
    # apply filter model
    idx_pairs, scores = predict_pairs(loader, filter_model, batch_size)
    
    # get true pairs
    true_pairs = []
    truth_np = np.matrix(truth.values)

    with torch.autograd.no_grad():
        for i, pair in enumerate(idx_pairs):
            hit_a = truth_np[pair[0], 1]
            hit_b = truth_np[pair[1], 1]
            if hit_a != 0 and hit_a == hit_b: #compare particle id
                true_pairs.append((pair[0], pair[1]))
            if i % (len(idx_pairs)//10) == 0:
                print(int((i+1) / (len(idx_pairs)//100)),"% done")
                
    # get filtered pairs
    filter_pairs, _ = apply_filter(idx_pairs, scores, 0.3)
    filter_pairs = [(pair[0], pair[1]) for pair in filter_pairs]
    
    # calculate efficiency and purity
    true_pairs_set = frozenset(true_pairs)
    n_true_pairs = sum(map(lambda n : n in true_pairs_set, filter_pairs))
    purity = n_true_pairs / len(filter_pairs)
    efficiency = n_true_pairs / len(true_pairs)
    return purity, efficiency


if __name__ == "__main__":
    feature_names = ['x', 'y', 'z', 'cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    
    # load models
    artifact_storage_path = "/global/cfs/cdirs/m3443/usr/aoka/data/artifacts/Training_Example_0"
    best_emb_path = os.path.join(artifact_storage_path, 'metric_learning_emb', 'best_model.pkl')
    best_filter_path = os.path.join(artifact_storage_path, 'metric_learning_filter', 'best_model.pkl')    

    emb_model = load_embed_model(best_emb_path, DEVICE).to(DEVICE)
    filter_model = load_filter_model(best_filter_path, DEVICE).to(DEVICE)
    emb_model.eval()
    filter_model.eval()
    
    
    # calculate purity and efficiency
    noise_keeps = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    emb_ps = []
    emb_efs = []
    filter_ps = []
    filter_efs = []
    for noise_keep in noise_keeps:
        event_name = "event000001000.pickle"
        data_path = f"/global/cfs/cdirs/m3443/usr/aoka/data/classify/Classify_Example_{noise_keep}/preprocess_raw"
        hits, truth = load_event(data_path, event_name)
        print("event:", noise_keep, "number of hits:", len(hits))

        neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, 0.4)

        emb_purity, emb_efficiency = get_emb_eff_purity(hits, truth, neighbors)
        print("emb result:", emb_purity, emb_efficiency)
        emb_ps.append(emb_purity)
        emb_efs.append(emb_efficiency)

        filter_purity, filter_efficiency = get_filter_eff_purity(hits, truth, neighbors)
        print("filter result:", filter_purity, filter_efficiency)
        filter_ps.append(filter_purity)
        filter_efs.append(filter_efficiency)
        
    
    # plot 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    x = [float(keep) for keep in noise_keeps]
    ax1.plot(x, emb_ps)
    ax1.set_title("embedding purity")
    ax1.set_xlabel("noise_keep")
    ax1.set_ylabel("%")
    ax2.plot(x, emb_efs)
    ax2.set_title("embedding efficiency")
    ax2.set_xlabel("noise_keep")
    ax2.set_ylabel("%")

    ax3.plot(x, filter_ps)
    ax3.set_title("filtering purity")
    ax3.set_xlabel("noise_keep")
    ax3.set_ylabel("%")
    ax4.plot(x, filter_efs)
    ax4.set_title("filtering efficiency")
    ax4.set_xlabel("noise_keep")
    ax4.set_ylabel("%")

    plt.tight_layout()
    plt.show()
    
    