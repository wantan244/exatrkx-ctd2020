import sys
import os

# Pick up local packages
sys.path.append('..')
sys.path.append('/global/homes/c/caditi97/exatrkx-ctd2020/MetricLearning/src/preprocess_with_dir/')
sys.path.append('..')
sys.path.append('/global/homes/c/caditi97/exatrkx-ctd2020/MetricLearning/src/metric_learning_adjacent/')


import time
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import functools
import seaborn as sns
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Pool as ProcessPool 
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
# %matplotlib inline
import trackml.dataset
from preprocess import get_one_event, load_detector
from tqdm import tqdm
import statistics

# Local imports
from build_graphs import *
from GraphLearning.src.trainers import get_trainer
from utils.data_utils import (get_output_dirs, load_config_file, load_config_dir, load_summaries,
                      save_train_history, get_test_data_loader,
                      compute_metrics, save_metrics, draw_sample_xy)


# Get rid of RuntimeWarnings, gross
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)








feature_names = ['x', 'y', 'z', 'cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
noise_keeps = ["0", "0.2", "0.4", "0.6", "0.8", "1"]




#############################################
#               GET DATA                    #
#############################################

# given amount of noise get data from respective file
def get_data(event_name, data_path=None, noise_keep=None): 
    # for no ptcut = "/global/cfs/projectdirs/m3443/usr/aoka/data/artifacts/Training_Example_no_ptcut"
    # xiangyang's model = "/global/cfs/projectdirs/m3443/usr/dtmurnane/artifacts/adjacent/"
    # misaligned data = "/global/cfs/projectdirs/m3443/data/trackml-kaggle/misaligned"
    # noise path = f"/global/cfs/cdirs/m3443/usr/aoka/data/classify/Classify_Example_{noise_keep}/preprocess_raw"
    artifact_storage_path = "/global/cfs/projectdirs/m3443/usr/dtmurnane/artifacts/adjacent/"
    best_emb_path = os.path.join(artifact_storage_path, 'metric_learning_emb', 'best_model.pkl')
    best_filter_path = os.path.join(artifact_storage_path, 'metric_learning_filter', 'best_model.pkl') 
    if noise_keep is None:
        noise_keep = 0
    else:
        data_path = f"/global/cfs/cdirs/m3443/usr/aoka/data/classify/Classify_Example_{noise_keep}/preprocess_raw"
    emb_model = load_embed_model(best_emb_path, DEVICE).to(DEVICE)
    filter_model = load_filter_model(best_filter_path, DEVICE).to(DEVICE)
    emb_model.eval()
    filter_model.eval()
    hits, truth = load_event(data_path, event_name)
    print("noise: " +str(noise_keep)+ ", number of hits:", len(hits))
    return hits, truth, emb_model, filter_model





#############################################
#               ADD NOISE                   #
#############################################

# add some percent noise 
def remove_all_noise(hits, cells, truth, perc = 0.0):
    print("removing " + str(perc) + " % noise")
    if perc >= 1.0:
        return hits,cells,truth
    
    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    noise_hits = unique_ids[np.where(unique_ids == 0)]
    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
    hits_reduced  = hits[where_to_keep]
    hit_ids_red = hits_reduced.hit_id.values
    noise_ids = hits[~where_to_keep].hit_id.values
    
    if perc <= 0.0:
        noise_ids = []
    else:
        num_rows = int(perc * noise_ids.shape[0])
        noise_ids = np.random.permutation(noise_ids)[:num_rows]

    #add noise
    hits_ids_noise = np.concatenate([hit_ids_red, noise_ids])
    
    noise_hits = hits[hits['hit_id'].isin(hits_ids_noise)]
    noise_truth = truth[truth['hit_id'].isin(hits_ids_noise)]
    noise_cells = cells[cells['hit_id'].isin(noise_truth.hit_id.values)]
    
    return noise_hits, noise_cells, noise_truth



#############################################
#                  PLOTS                    #
#############################################


# scatter plot of noise hits vs non-noise hits given percentage of noise,
# index of neighborhood/hit and hits and truth data
def plot_noise(noise_hits,noise_truth,noise_keep,index):
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(noise_hits.shape)
    print("truth")
    print(noise_truth.shape)

    unique_ids = noise_truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    where_to_keep = noise_truth['particle_id'].isin(track_ids_to_keep)
    not_noise  = noise_hits[where_to_keep]
    noise = noise_hits[~where_to_keep]
    print("Not Noise Hits = " + str(len(not_noise)))
    print("Noise Hits = " + str(len(noise)))

    g3 = sns.jointplot(not_noise.x, not_noise.y, s=2, height=12, label = "not noise")
    g3.x = noise.x
    g3.y = noise.y
    g3.plot_joint(plt.scatter, c='r', s=1, label = "noise")


    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.title('Noise Distribution')
    plt.savefig('noise_[' +str(index)+ ']_' + str(noise_keep) + '.png', bbox_inches='tight')
    plt.show()
    
    
# given hits, truth data, neighbors, index of hit and percentage of noise 
# give a scatter plot of the hits and noise inside neighborhood
def plot_neighborhood(hits, truth, neighbors, noise_keep, k=None):
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(hits.shape)
    print("truth")
    print(truth.shape)
    
    hitidx = neighbors[k]
    hitids = hits.iloc[hitidx]['hit_id'].values
    print("len(neighbors[k]) = " +str(len(hitids)))
    sel_hits = hits[hits['hit_id'].isin(hitids)]
    # hits in a neighborhood
    print("Hits in the Neighborhood = " + str(len(sel_hits)))
    diff_n = len(hits) - len(sel_hits)
    print("Hits outside the Neighborhood = " + str(diff_n))
    g = sns.jointplot(sel_hits.x, sel_hits.y, s = 5, height = 12, label ='neighborhood')
    
    #noise in neighborhood
    truth_np = np.array(truth.values)
    noise_ids = []
    for i in hitidx:
            if truth_np[i, 1] == 0: noise_ids.append(truth_np[i, 0])
#     noise_idx = truth[truth['particle_id'] == 0]
#     noise_ids = noise_idx[noise_idx['hit_id'].isin(hitids)]
    noise_in = hits[hits['hit_id'].isin(noise_ids)]
    
    g.x = noise_in.x
    g.y = noise_in.y
    g.plot_joint(plt.scatter, c = 'r', s=5, label='noise in neighborhood')
    print("Noise in Neighborhood = " + str(len(noise_in)))
#     diff = len(noise) - len(noise_in)
#     print("Noise outside Neibhorhood = " + str(diff))
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.savefig('neighbor[' +str(k)+ ']_' + str(noise_keep) + '.png', bbox_inches='tight')
    plt.show()
    
    
    
    
# plot the hits and noise in the neighborhood with respect to all hits in the 
# event
def plot_allhits_with_neighborhood(hits, truth, neighbors, noise_keep, k):
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(hits.shape)
    print("truth")
    print(truth.shape)

    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
    not_noise  = hits[where_to_keep]
    noise = hits[~where_to_keep]
    print("Not Noise Hits = " + str(len(not_noise)))
    print("Noise Hits = " + str(len(noise)))
    
    #noise vs not noise
    g = sns.jointplot(not_noise.x, not_noise.y, s=1, height=20, label = "not noise")
    g.x = noise.x
    g.y = noise.y
    g.plot_joint(plt.scatter, c='r', s=1, label = "noise")
    
    # vs neighborhood
    hitidx = neighbors[k]
    hitids = hits.iloc[hitidx]['hit_id'].values
    print("len(neighbors[k]) = " +str(len(hitids)))
    # hits in a neighborhood
    sel_hits = hits[hits['hit_id'].isin(hitids)]
    print("Hits in the Neighborhood = " + str(len(sel_hits)))
    diff_h = len(hits) - len(sel_hits)
    print("Hits outside the Neighborhood = " + str(diff_h))
    g.x = sel_hits.x
    g.y = sel_hits.y
    g.plot_joint(plt.scatter, c = 'k', s=2, label='neighborhood')
    
    #noise in neighborhood
    truth_np = np.array(truth.values)
    noise_ids = []
    for i in hitidx:
            if truth_np[i, 1] == 0: noise_ids.append(truth_np[i, 0])
    noise_in = hits[hits['hit_id'].isin(noise_ids)]
    
    g.x = noise_in.x
    g.y = noise_in.y
    g.plot_joint(plt.scatter, c = 'y', s=3, label='noise in neighborhood')
    print("Noise in Neighborhood = " + str(len(noise_in)))
    diff_n = len(noise) - len(noise_in)
    print("Noise outside Neibhorhood = " + str(diff_n))
    
    if(len(noise) == 0):
        in_hits = len(sel_hits)/len(hits)
        out_hits = diff_h/len(hits)
        in_noise = 0
        out_noise = 0
    else:
        in_hits = len(sel_hits)/len(hits)
        out_hits = diff_h/len(hits)
        in_noise = len(noise_in)/len(noise)
        out_noise = diff_n/len(hits)
        
    
    
    print("----------------")
    print("% Hits inside = " +str(in_hits))
    print("% Hits outside = " +str(out_hits))
    print("% Noise inside = " +str(in_noise))
    print("% Noise outside = " +str(out_noise))
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.savefig('all_neighbor[' +str(k)+ ']_' + str(noise_keep) + '.png', bbox_inches='tight')
    plt.show()
    
    return in_hits, out_hits, in_noise, out_noise




    
# function to show various kinds of plots for a single noise value
def plots(hits, truth, index, emb_model, radius = 0.4):
    neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, radius)
    print("Total Neighborhoods/Hits = " + str(len(neighbors)))
    print("Chosen neighborhood/Hit = " + str(index))
    
    plot_noise(hits,truth,noise_keep, index)
    
    in_hits, out_hits, in_noise, out_noise = plot_allhits_with_neighborhood(hits, truth, neighbors, noise_keep, index)
    
    plot_neighborhood(hits,truth, neighbors, noise_keep, index)
    
    return in_hits, out_hits, in_noise, out_noise
    
    
    
# function to get various plots for all noise values
def overall(index):
    
    in_hits =[]
    out_hits =[]
    in_noise=[]
    out_noise =[]
    
    for noise_keep in noise_keeps:
        hits, truth, emb_model, filter_model = get_data(event_name,None,noise_keep)
        in_h, out_h, in_n, out_n = plots(hits, truth, noise_keep, feature_names, index, emb_model, radius=0.4)
        in_hits.append(in_h)
        out_hits.append(out_h)
        in_noise.append(in_n)
        out_noise.append(out_n)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
    x = [float(keep) for keep in noise_keeps]
    ax1.plot(x, in_hits)
    ax1.set_title("% Hits inside Neighborhood")
    ax1.set_xlabel("noise_keep")
    ax2.plot(x, out_hits)
    ax2.set_title("% Hits outside Neighborhood")
    ax2.set_xlabel("noise_keep")
    
    ax3.plot(x, in_noise)
    ax3.set_title("% Noise inside Neighborhood")
    ax3.set_xlabel("noise_keep")
    ax4.plot(x, out_noise)
    ax4.set_title("% Noise outside Neighborhood")
    ax4.set_xlabel("noise_keep")
    
    plt.savefig("overall_[" +str(index)+ "].png", bbox_inches='tight')
    plt.tight_layout()
    
# compare results before and after removing tails
def plot_new_dist(count8, count13, count17):
    avg8_n, avg8_o, std8_n, std8_o = remove_tails(count8)
    avg13_n, avg13_o, std13_n, std13_o = remove_tails(count13)
    avg17_n, avg17_o, std17_n, std17_o = remove_tails(count17)
    
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(10,10))
    x = [float(keep) for keep in noise_keeps]
    ax1.errorbar(x,avg8_n, xerr=std8_n, label="After Removing Lower End")
    ax1.errorbar(x,avg8_o, xerr=std8_o, label="Before Removing Lower End")
    ax1.set_title("Volume 8")
    
    ax2.errorbar(x,avg13_n, xerr=std13_n, label="After Removing Lower End")
    ax2.errorbar(x,avg13_o, xerr=std13_o, label="Before Removing Lower End")
    ax2.set_title("Volume 13")
    
    ax3.errorbar(x,avg17_n, xerr=std17_n, label="After Removing Lower End")
    ax3.errorbar(x,avg17_o, xerr=std17_o, label="Before Removing Lower End")
    ax3.set_title("Volume 17")
    
    
    
    
#############################################
#         EMBEDDING NOISE RATIOS            #
#############################################
    
# helper function to calculate embedding metrics/ratios for one noise value
def ratios(hits, truth, emb_model,radius=0.4):
    neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, radius)
    print("----" + str(noise_keep) + " Noise----")
    print("hits")
    print(hits.shape)
    print("truth")
    print(truth.shape)

    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
    not_noise  = hits[where_to_keep]
    noise = hits[~where_to_keep]
    print("Not Noise Hits = " + str(len(not_noise)))
    print("Noise Hits = " + str(len(noise)))
    
    truth_np = np.array(truth.values)
    in_hits =[]
    out_hits =[]
    in_noise =[]
    out_noise =[]
        
    n_nbr = len(neighbors)
    for nbr in tqdm(range(n_nbr)):
        hood = neighbors[nbr]
        in_h = len(hood)/len(hits)
        out_h = (len(hits)-len(hood))/len(hits)
        in_hits.append(in_h)
        out_hits.append(out_h)
        noise_count = 0
        if (len(noise) == 0):
            in_noise =[]
            out_noise =[]
            in_noise_mean = 0 
            out_noise_mean = 0
        else:
            for hit in hood:
                if truth_np[hit, 1] == 0: noise_count+=1
            in_n = noise_count/len(hood)
            out_n = (len(noise) - noise_count)/len(hits)
            in_noise.append(in_n)
            out_noise.append(out_n)
            
    if(len(noise)!=0):
        in_noise_mean = statistics.mean(in_noise)
        out_noise_mean = statistics.mean(out_noise)
        
    return statistics.mean(in_hits), statistics.mean(out_hits), in_noise_mean, out_noise_mean


# function to get embedding metrics/ratios for all noise values    
def overall_ratios():
    
    in_hits =[]
    out_hits =[]
    in_noise=[]
    out_noise =[]
    
    for noise_keep in noise_keeps:
        hits, truth, emb_model, filter_model = get_data(event_name, None, noise_keep)
        in_h, out_h, in_n, out_n = ratios(hits, truth, feature_names,noise_keep, emb_model,0.4)
        in_hits.append(in_h)
        out_hits.append(out_h)
        in_noise.append(in_n)
        out_noise.append(out_n)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
    x = [float(keep) for keep in noise_keeps]
    ax1.plot(x, in_hits)
    ax1.set_title("% Hits inside Neighborhood")
    ax1.set_xlabel("noise_keep")
    ax2.plot(x, out_hits)
    ax2.set_title("% Hits outside Neighborhood")
    ax2.set_xlabel("noise_keep")
    
    in_noise.pop(0)
    out_noise.pop(0)
    noise_keepsn = ["0.2", "0.4", "0.6", "0.8", "1"]
    xn = [float(keep) for keep in noise_keepsn]
    ax3.plot(xn, in_noise)
    ax3.set_title("% Noise inside Neighborhood")
    ax3.set_xlabel("noise_keep")
    ax4.plot(xn, out_noise)
    ax4.set_title("% Noise outside Neighborhood")
    ax4.set_xlabel("noise_keep")
    
    plt.savefig("overall_allhits.png", bbox_inches='tight')
    plt.tight_layout()
    
    
    
    
    
#############################################
#           FILTERING METRICS               #
#############################################

# get pairs from truth data that are also inside filtered neighborhood
def get_truth_pairs(hits, truth):
    vol = hits[['volume_id', 'layer_id']].values.T
    true_pairs = []
    pids = truth[truth['particle_id'] != 0]['particle_id'].unique()
    for pid in tqdm(pids):
        seed_hits = hits[truth['particle_id']==pid].index.values.astype(int)
        for i in seed_hits:
            hit = hits.iloc[i]
            true_neighbors = filter_one_neighborhood(hit['volume_id'], hit['layer_id'], seed_hits, vol[0], vol[1])
            true_pairs += [(i, n) for n in true_neighbors]
    return true_pairs


# apply filtering model and if select is true filter pairs with score >= 95
def apply_filter_model(hits, filter_model, neighbors, select = True, radius=0.4, threshold=0.95):
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
    
    if (select):
        idx_pairs, scores = apply_filter(idx_pairs, scores, threshold)
        print("   {:6.5f}% neighbors after filter".format( (1.0 * len(scores)) / len(hits)) +" ---#pairs = {}".format(len(idx_pairs)))
    else:
        print("   {:6.5f}% neighbors before filter".format((1.0 * len(scores)) / len(hits)) +" ---#pairs = {}".format(len(idx_pairs)))
        
    return idx_pairs, scores

# got pairs with atleast one noise hit in them from the neighborhood
def get_noise_pairs(pairs,truth):
    truth_np = np.array(truth.values)
    n = 0
    for pair in tqdm(pairs):
        hit_a = truth_np[pair[0], 1]
        hit_b = truth_np[pair[1], 1]
        if hit_a == 0 or hit_b == 0: 
            n += 1
    return n

# get ratios for filtered pairs and plot them
def get_filter_metrics():
    t_pairs = []
    f_pairs = []
    purity = []
    efficiency = []
    in_noise = []
    total_noise = []
    noise_ratios = []
    
    for noise in noise_keeps:
        hits, truth, emb_model, filter_model = get_data(noise)
        print("-----Getting Neighbors-----")
        neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, 0.4)
        print("#Neighbors = {}".format(len(neighbors)))
        print("-----Getting All Pairs-----")
        all_pairs, all_scores = apply_filter_model(hits, filter_model, neighbors,False)
        print("-----Filtering Pairs-----")
        filter_pairs, filter_scores = apply_filter_model(hits, filter_model, neighbors,True)
        print("-----Getting True Pairs-----")
        all_true_pairs = get_truth_pairs(hits,truth)
        n_pairs = [(pair[0], pair[1]) for pair in filter_pairs]
        t_pairs.append(len(all_pairs))
        f_pairs.append(len(filter_pairs))
        print("-----Getting Total Noise-----")
        t_noise = get_noise_pairs(all_pairs,truth)
        print("-----Getting Noise Above Threshold-----")
        f_noise = get_noise_pairs(filter_pairs,truth)
        total_noise.append(t_noise)
        in_noise.append(f_noise)
        if t_noise == 0:
            n_ratio = 0
        else:
            n_ratio = f_noise/t_noise
        print("Noise Ratio = " + str(n_ratio))
        noise_ratios.append(n_ratio)
        all_t = frozenset(all_true_pairs)
        n_true_f = sum(map(lambda n : n in all_t, n_pairs))
        p = n_true_f/len(filter_pairs)
        e = n_true_f/len(all_true_pairs)
        purity.append(p)
        efficiency.append(e)

    
        fig, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3,2, figsize=(20,25))
        x = [float(keep) for keep in noise_keeps]

        ax1.plot(x, t_pairs)
        ax1.set_title("Total Number of Filtered Pairs")
        ax1.set_xlabel("noise_keep")
        ax2.plot(x, f_pairs)
        ax2.set_title("Filtered Pairs above 0.95 Threshold")
        ax2.set_xlabel("noise_keep")

        ax3.plot(x, purity)
        ax3.set_title("Purity")
        ax3.set_xlabel("noise_keep")
        ax4.plot(x, efficiency)
        ax4.set_title("Efficiency")
        ax4.set_xlabel("noise_keep")

        noise_ratios.pop(0)
        total_noise.pop(0)
        in_noise.pop(0)
        noise_keepsn = ["0.2", "0.4", "0.6", "0.8", "1"]
        xn = [float(keep) for keep in noise_keepsn]
        ax5.plot(xn, noise_ratios, label='Noise Ratio')
        ax5.set_title("Noise Ratio")
        ax5.set_xlabel("noise_keep")
        plt.legend()
        ax6.plot(xn, in_noise)
        ax6.plot(xn, total_noise, label='Total Noise')
        ax6.set_title("Noise above Threshold")
        ax6.set_xlabel("noise_keep")


        plt.savefig("filter_metrics.png", bbox_inches='tight')
        plt.tight_layout()
        plt.show()
   
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,25))
        ax1.plot(x, purity)
        ax1.set_title("Purity")
        ax1.set_xlabel("noise_keep")
        ax2.plot(x, efficiency)
        ax2.set_title("Efficiency")
        ax2.set_xlabel("noise_keep")
        
        
        
    
#############################################
#            EMBEDDING METRICS              #
#############################################



# get hits and noise inside a neighborhood
def neighborhood_hits(hits, truth, neighbors, k):
    hitidx = neighbors[k]
    hitids = hits.iloc[hitidx]['hit_id'].values
    sel_hits = hits[hits['hit_id'].isin(hitids)]
   
    
    #noise in neighborhood
    truth_np = np.array(truth.values)
    noise_ids = []
    for i in hitidx:
            if truth_np[i, 1] == 0: noise_ids.append(truth_np[i, 0])
    noise_in = hits[hits['hit_id'].isin(noise_ids)]
    
    return noise_in, sel_hits


# get average ratio of noise inside/hits inside and total ratio of
# noise in event/hits in event
# scale the entries in historam by 1/#of entries
# entries are randomly selected non noise hits in neighborhood
def random_neighborhood(event_name):
    radius = 0.4
    fig1, axes1 = plt.subplots(2, 3, figsize=(16,10))
    fig2, axes2 = plt.subplots(2, 3, figsize=(16,10))
    avg_ratio = []
    total_noise = []
    noise_dist = []
    for i,noise_keep in enumerate(noise_keeps):
        hits, truth, emb_model, filter_model = get_data(event_name,None,noise_keep)
        unique_ids = truth.particle_id.unique()
        track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
        where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
        not_noise  = hits[where_to_keep]
        noise = hits[~where_to_keep]
        
        t_noise = len(noise)/len(hits)
        total_noise.append(t_noise)
        print("Not Noise Hits = " + str(len(not_noise)))
        print("Noise Hits = " + str(len(noise)))
        neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, radius)

        ratios = []
        
        #np.random.seed(0)
        for j in tqdm(range(5000)):
            k = np.random.choice(not_noise.index.values.astype(int))
            n_in, h_in = neighborhood_hits(hits, truth, neighbors, k)
            r = len(n_in) / len(h_in)
            ratios.append(r)
        print("mean = " + str(statistics.mean(ratios)))
        avg_ratio.append(statistics.mean(ratios))
        print("----------")
        
        (counts, bins) = np.histogram(ratios,bins=50)
        factor = 1/5000
        axes1[i//3][i%3].hist(bins[:-1], bins, weights=factor*counts)
        mu, std = norm.fit(ratios)
        if len(noise) == 0:
            continue
        else:
            noise_dist.append(ratios)
            sns.distplot(ratios, fit=norm, hist=True, kde=False, ax = axes2[i//3][i%3])
            title = "mu = %.2f,  std = %.2f" % (mu, std)
            axes2[i//3][i%3].set_title(title)
        
    f, a = plt.subplots(1, 1, figsize=(10,10))
    x = [float(keep) for keep in noise_keeps]
    a.plot(x,avg_ratio, label="Average Noise Ratio")
    a.plot(x,total_noise, label="Total Noise Ratio")
    plt.legend()
    plt.show()
    
    return noise_dist, total_noise
    


# get the distribution of hits inside a neighborhood according to the
# volume they are present in and plot the distribution for each 
# volume id
def hits_by_vol():
    radius = 0.4
    
    count8 = []
    count13 = []
    count17 = []

    fig1, axes1 = plt.subplots(2, 3, figsize=(16,10))
    fig2, axes2 = plt.subplots(2, 3, figsize=(16,10))
    fig3, axes3 = plt.subplots(2, 3, figsize=(16,10))

    for i,noise_keep in enumerate(noise_keeps):
        hits, truth, emb_model, filter_model = get_data(noise_keep)
        np_hits = np.array(hits.values)

        neighbors = get_emb_neighbors(hits[feature_names].values, emb_model, radius)

        vols8 = []
        vols13 = []
        vols17 = []

        #for each hitidx
        for k in tqdm(range(len(hits))):
            vol8 = 0
            vol13 = 0
            vol17 = 0
            #get hits inside neighborhood
            hitidx = neighbors[k]
            for idx in hitidx:
                hit = np_hits[idx]
                if (hit[4] == 8):
                    vol8+=1
                if (hit[4] == 13):
                    vol13+=1
                if (hit[4] == 17):
                    vol17+=1
            vols8.append(vol8)
            vols13.append(vol13)
            vols17.append(vol17)

      
        # get normal distribution for each plot
        mu8, std8 = norm.fit(vols8)
        mu13, std13 = norm.fit(vols13)
        mu17, std17 = norm.fit(vols17)
        
        # Plot the PDF.
        sns.distplot(vols8, fit=norm, kde=False, ax = axes1[i//3][i%3])
        title8 = "mu = %.2f,  std = %.2f" % (mu8, std8)
        axes1[i//3][i%3].set_title(title8)
        fig1.suptitle("Volume 8")
        
        sns.distplot(vols13, fit=norm, kde=False, ax = axes2[i//3][i%3])
        title13 = "mu = %.2f,  std = %.2f" % (mu13, std13)
        axes2[i//3][i%3].set_title(title13)
        fig2.suptitle("Volume 13")
        
        sns.distplot(vols17, fit=norm, kde=False, ax = axes3[i//3][i%3])
        title17 = "mu = %.2f,  std = %.2f" % (mu17, std17)
        axes3[i//3][i%3].set_title(title17)
        fig3.suptitle("Volume 17")
        
        count8.append(vols8)
        count13.append(vols13)
        count17.append(vols17)
        
    return count8, count13, count17
    

# given distribution remove entries that are one standard deviation
# below the mean
def remove_tails(dist):
    avg_new = []
    avg_old = []
    std_new = []
    std_old = []
    fig, axes = plt.subplots(2, 3, figsize=(16,10))
    
    for i,noise_level in enumerate(dist):
        mu, std = norm.fit(noise_level)
        # remove data below 1 std
        lower = mu - std
        new_dist = [d for d in noise_level if d >= lower]
        mu_n, std_n = norm.fit(new_dist)
        sns.distplot(new_dist, fit=norm, kde=False, ax = axes[i//3][i%3])
        title = "mu = %.2f,  std = %.2f" % (mu_n, std_n)
        axes[i//3][i%3].set_title(title)
        avg_new.append(mu_n)
        avg_old.append(mu)
        std_new.append(std_n)
        std_old.append(std)
    return avg_new, avg_old, std_new, std_old
        
    
def get_one_emb_eff_purity(index, hits, truth, emb_neighbors, only_adjacent=False):
    vol = hits[['volume_id', 'layer_id']].values.T
    hit = hits.iloc[index]
    pid = truth.iloc[index]['particle_id']
    
    # get true neighbors based on particle id
    if pid == 0:
        true_neighbors = []
    else:
        hit_idx = truth[truth['particle_id']==pid]['hit_id']
        true_hits = hits[hits['hit_id'].isin(hit_idx) & (hits['hit_id'] != hit['hit_id'])]
        true_neighbors = true_hits.index.values.astype(int)
        if only_adjacent:
            true_neighbors = filter_one_neighborhood(hit['volume_id'], hit['layer_id'], true_neighbors, vol[0], vol[1])
    
    emb_neighbors = emb_neighbors[index]
    if only_adjacent:
        emb_neighbors = filter_one_neighborhood(hit['volume_id'], hit['layer_id'], emb_neighbors, vol[0], vol[1])
    
    # calculate purity and eff
    n_true_neighbors = sum(map(lambda n : n in true_neighbors, emb_neighbors))
    purity = n_true_neighbors / len(emb_neighbors) if len(emb_neighbors) > 0 else None
    efficiency = n_true_neighbors / len(true_neighbors) if len(true_neighbors) > 0 else None
    return purity, efficiency


def get_emb_eff_purity(hits, truth, emb_neighbors, only_adjacent=False):
    n_iter = len(hits)
    purity = []
    efficiency = []

    for i in tqdm(range(n_iter)):
        p, eff = get_one_emb_eff_purity(i, hits, truth, emb_neighbors, only_adjacent)
        if p: purity.append(p)
        if eff: efficiency.append(eff)

    return purity, efficiency
    
    

        