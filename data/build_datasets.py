import os

import torch
from torch.utils import data
import numpy as np

from data.pose_graph_tools import Graph
from data.sample_data import random_sample, sequential_sample

def build_sub_graph(runs, data_dir):
    """
        Build a pose graph from the stored data files. We build the pose graph using the tools provided in
        https://github.com/utiasASRL/vtr-dataset-tools. We use the data provided by the UTIAS Long-Term Localization
        and Mapping Dataset that can be found at http://asrl.utias.utoronto.ca/datasets/2020-vtr-dataset/index.html.

        Args:
            runs (list[int]): the ids of the runs from the dataset that we want to include in the pose graph.
            data_dir (string): the top-level directory that holds the data for the path for which we build a pose graph.

        Returns:
            graph (Graph): a pose graph built from the runs that was provided.
    """
    teach_file = f'{data_dir}/run_000000/transforms_temporal.txt'
    repeat_files = []

    for run in runs:
        if run != 0:
            run_str = str(run).zfill(6)
            run_file = f'{data_dir}/run_{run_str}/transforms_spatial.txt'
            if os.path.isfile(run_file):
                repeat_files.append(run_file)
            else:
                print(f'Building pose graph, run_file does not exist: {run_file}')
                exit(1)

    return Graph(teach_file, repeat_files)

def build_random_loc_dataset(data_path, path_names, runs, num_samples, temporal_length):
    """
        Build a dataset that localizes vertices of one run in the pose graph to another. We sample vertex pairs
        randomly from the pose graph.

        Record the pose transforms as 4x4 matrices and the 6 DOF vector equivalents. The pose from vertex, v1, to
        vertex, v2, is given as T_v2_v1. Create sample ids from the vertex ids. A vertex id consists of the id of the
        run the vertex belongs to and the id of the pose along that run, i.e. vertex_id = (run_id, pose_id). The sample
        id corresponding to pose transform T_v2_v1 is on the form pathname_runid1_poseid1_runid2_poseid2, for
        instance: mutliseason_1_531_5_542.

        Args:
            data_path (string): path to where the different runs of data are stored.
            path_names (list[string]): the paths we will use for sampling. One pose graph is created for each path.
            runs (dict): map from the path names to a list of the runs to use for localization pose sampling for the
                         given path.
            num_samples (dict): map from the path names to the number of samples to generate for the given paths.
            temporal_length (int): we can 'walk along' the pose graph to pair vertices that har further apart (i.e.
                                   not the closest pair). We set a fixed topological distance/steps we move away
                                   from the start vertex.

        Returns:
             samples (list[string]): list of all the sample ids.
             labels_se3 (dict): dictionary mapping sample id to pose transform 4x4 matrix provided as a torch.Tensor.
             labels_log (dict): dictionary mapping sample id to pose transform 6 DOF vector provided as a torch.Tensor.
    """
    all_ids = []
    labels_se3 = {}
    labels_log = {}

    for path_name in path_names:

        num_samples_path = num_samples[path_name]
        num_runs = len(runs[path_name])

        print(f'\nRandom dataset from path:  {path_name}')
        print(f'Sample from runs: {runs[path_name]}')
        print(f'Collect {num_samples_path} samples \n')

        data_dir = f'{data_path}/{path_name}'
        pose_graph = build_sub_graph(runs[path_name], data_dir)

        path_ids, path_labels_se3, path_labels_log = random_sample(path_name, pose_graph, runs[path_name],
                                                                   num_samples_path, temporal_length)

        all_ids = all_ids + path_ids
        labels_se3 = {**labels_se3, **path_labels_se3}
        labels_log = {**labels_log, **path_labels_log}

    print(f'\nRandom dataset total samples: {len(all_ids)}\n')
                        
    return all_ids, labels_se3, labels_log

def build_sequential_loc_dataset(data_path, path_name, map_run_id, live_run_ids, temporal_length):
    """
        Build a dataset that localizes all the vertices of one or more runs in the pose graph to the vertices on one
        map (or teach) run. I.e. we localize one or more live (or repeat) runs to one run that we choose as the map
        run. We get relative pose transforms for each localized vertex in the order that the vertices were created when
        driving the robot during data collection.

        Record the pose transforms as 4x4 matrices and the 6 DOF vector equivalents. The pose from vertex, v1, to
        vertex, v2, is given as T_v2_v1. Create sample ids from the vertex ids. A vertex id consists of the id of the
        run the vertex belongs to and the id of the pose along that run, i.e. vertex_id = (run_id, pose_id). The sample
        id corresponding to pose transform T_v2_v1 is on the form pathname_runid1_poseid1_runid2_poseid2, for
        instance: mutliseason_1_531_5_542.

        Args:
            data_path (string): path to where the different runs of data are stored.
            path_name (string): name given to the path that the pose graph represents.
            map_run_id (int): id of the run to localize to, i.e. compute the relative pose to vertices on this run.
            live_run_ids (list[int]): the runs we localize to the map run, i.e. compute relative pose from vertices on
                                      these runs.
            temporal_length (int): we can 'walk along' the pose graph to pair vertices that har further apart (i.e.
                                   not the closest pair). We set a fixed topological distance/steps we move away
                                   from the start vertex.

        Returns:
             samples (list[string]): list of all the sample ids.
             labels_se3 (dict): dictionary mapping sample id to pose transform 4x4 matrix provided as a torch.Tensor.
             labels_log (dict): dictionary mapping sample id to pose transform 6 DOF vector provided as a torch.Tensor.
    """

    print(f'\nSequential dataset from path: {path_name}')
    print(f'Map (teach) run: {map_run_id}')
    print(f'Live (repeat) runs to localize: {live_run_ids} \n')

    data_dir = f'{data_path}/{path_name}'
    pose_graph = build_sub_graph([map_run_id] + live_run_ids, data_dir)

    return sequential_sample(path_name, pose_graph, map_run_id, live_run_ids, temporal_length)

