from operator import itemgetter
import re
import random

import torch
import numpy as np

from src.utils.transform import Transform

def random_sample(path_name, pose_graph, runs, num_samples, max_temporal_length):
    """
        Sample relative pose transforms for localization randomly from the pose graph. Compute the pose between vertices
        from different experiences in the pose graph. Record the pose transform as a 4x4 matrix and the 6 DOF vector
        equivalent. The pose from vertex, v1, to vertex, v2, is given as T_v2_v1.

        Create sample ids the vertex ids. A vertex id consists of the id of the run the vertex belongs to and the id of
        the pose along that run, i.e. vertex_id = (run_id, pose_id). The sample id corresponding to pose transform
        T_v2_v1 is on the form pathname_runid1_poseid1_runid2_poseid2, for instance: mutliseason_1_531_5_542.

        Args:
            path_name (string): name given to the path that the pose graph represents.
            pose_graph (Graph): the pose graph.
            runs (list[int]): list of the run ids of runs to sample from.
            num_samples (int): the number of samples to collect.
            max_temporal_length (int): we can 'walk along' the pose graph to pair vertices that har further apart (i.e.
                                       not the closest pair). This is the max topological distance/steps we move away
                                       form the start vertex.

        Returns:
             samples (list[string]): list of all the sample ids.
             labels_se3 (dict): dictionary mapping sample id to pose transform 4x4 matrix provided as a torch.Tensor.
             labels_log (dict): dictionary mapping sample id to pose transform 6 DOF vector provided as a torch.Tensor.
    """
    sample_counter = 0
    sample_counter_identity = 0
    samples = []
    labels_se3 = {}
    labels_log = {}

    # The ids of all the vertices in the pose graph.
    vertex_ids = pose_graph.get_all_vertices().keys()

    while sample_counter < num_samples:

        # Randomly choose a live index, live_id is on the form (run_id, pose_id)
        live_id = random.sample(vertex_ids, 1)[0]
        live_vertex = pose_graph.get_vertex(live_id)

        if live_vertex is None:
            print('live vertex is None')
            continue

        # The chosen vertex is not in one of the runs we want to sample from.
        if live_id[0] not in runs:
            continue

        # Get all the neighbouring vertices (on all runs/experiences) withing the specified distance.
        try:
            neighbour_ids = pose_graph.get_topo_neighbours(live_id, max_temporal_length + 1)
        except Exception as e:
            print(e)
            print(f'Could not get topological neighbours for live_id: {live_id}')
            continue

        if len(neighbour_ids) == 0:
            print(f'{path_name} - Random sampling: Could not find neighbours for vertex {live_id} within topological' \
                  f' distance {max_temporal_length}')
            continue

        # Randomly pick a target vertex from the neighbours.
        other_id = random.sample(neighbour_ids, 1)[0]

        if other_id[0] not in runs:
            print(f'{path_name} - Random sampling: other vertex is in run {other_id[0]}, which is not in desired runs')
            continue

        if live_id[0] == other_id[0]:
            print(f'{path_name} - Random sampling: do not want to localize to vertex on the same run')
            continue

        other_vertex = pose_graph.get_vertex(other_id)

        if other_vertex is None:
            print('other vertex is None')
            continue 
        
        # Create a sample ID and check that it has not already been added.
        sample_id = f'{path_name}-{live_id[0]}-{live_id[1]}-{other_id[0]}-{other_id[1]}'
        
        if not sample_id in samples:

            T_other_live = pose_graph.get_transform(live_id, other_id) 
            log_other_live = Transform.LogMap(T_other_live)

            samples.append(sample_id)
            labels_se3[sample_id] = torch.tensor(T_other_live.matrix, dtype=torch.float)
            labels_log[sample_id] = torch.tensor(log_other_live, dtype=torch.float)

            sample_counter += 1

        else:

            print(f'{path_name} - Random sampling: sample {sample_id} has already been added', flush=True)
            print(f'Number of sampled poses so far: {sample_counter}')

    return samples, labels_se3, labels_log


def sequential_sample(path_name, pose_graph, map_run_id, live_runs, temporal_length):
    """
        Sample relative pose transforms for localization sequentially from the pose graph. Compute the pose from
        vertices from each of the live runs to one map run. Compute the pose for each vertex on the live runs
        sequentially. Record the pose transform as a 4x4 matrix and the 6 DOF vector equivalent. The pose from vertex,
        v1, to vertex, v2, is given as T_v2_v1.

        Create sample ids the vertex ids. A vertex id consists of the id of the run the vertex belongs to and the id of
        the pose along that run, i.e. vertex_id = (run_id, pose_id). The sample id corresponding to pose transform
        T_v2_v1 is on the form pathname_runid1_poseid1_runid2_poseid2, for instance: mutliseason_1_531_5_542.

        Args:
            path_name (string): name given to the path that the pose graph represents.
            pose_graph (Graph): the pose graph.
            map_run_id (int): id of the run to localize to, i.e. compute the relative pose to vertices on this run.
            live_runs (list[int]): the runs we localize to the map run, i.e. compute relative pose from vertices on
                                   these runs.
            temporal_length (int): we can 'walk along' the pose graph to pair vertices that har further apart (i.e.
                                   not the closest pair). We set a fixed topological distance/steps we move away
                                   from the start vertex.

        Returns:
             samples (list[string]): list of all the sample ids.
             labels_se3 (dict): dictionary mapping sample id to pose transform 4x4 matrix provided as a torch.Tensor.
             labels_log (dict): dictionary mapping sample id to pose transform 6 DOF vector provided as a torch.Tensor.
    """
    samples = []
    added_live = []
    labels_se3 = {}
    labels_log = {}

    # Get the graph vertices in a sorted list by run_id and pose_id. The vertex ids are on the form (run_id, pose_id).
    # We retrieve all the vertices in the pose graph and filter out the correct ones later.
    vertex_ids = pose_graph.get_all_vertices().keys()
    vertex_ids = sorted(vertex_ids, key=itemgetter(0,1))

    # Localize each vertex in the live run sequentially.
    for live_id in vertex_ids:
        
        live_vertex = pose_graph.get_vertex(live_id)

        if live_vertex is None:
            continue
        
        if (live_id[0] in live_runs) and (live_id[0] != map_run_id):

            # Vertices are connected via the teach run 0 in the pose graph, which adds one to the distance count between
            # them, i.e. if run5_pose50 is localized directly to run3_pose54, we get r5_p50 -> r0_p52 -> r3_p54 so a
            # radius of 2 instead of 1.
            radius = temporal_length + 1
            max_radius = 8
            map_pose_id = -1
            smallest_metric_dist = 1000
            chosen_topo_dist = 1000

            # Try to find a vertex on the map run to localize against. If we can find one at the specified topological
            # distance, then increase the search radius.
            while (map_pose_id == -1) and (radius <= max_radius):
                try:
                    neighbour_ids = pose_graph.get_topo_neighbours(live_id, radius)
                except Exception as e:
                    print(e)
                    print(f'{path_name} - Sequential sampling: Could not localize {live_id} to map run, topological' \
                          f' neighbours failed')
                    radius += 1
                    continue

                # Find the closest vertex (on the map run) of the valid neighbours to localize to.
                for n_id in neighbour_ids:
                    if n_id[0] == map_run_id:
                        topo_dist = pose_graph.get_topological_dist(live_id, n_id)
                        T_n_live = pose_graph.get_transform(live_id, n_id)
                        metric_dist = np.linalg.norm(T_n_live.r_ab_inb)
                        if (metric_dist < smallest_metric_dist) and (topo_dist >= temporal_length + 1):
                            smallest_metric_dist = metric_dist
                            chosen_topo_dist = topo_dist
                            map_pose_id = n_id[1]

                radius += 1

            if map_pose_id == -1:
                print(f'{path_name} - Sequential sampling: Could not localize {live_id} to map run ' \
                      f'within {max_radius -1} edges.')
                continue

            if chosen_topo_dist != temporal_length + 1:
                print(f'{path_name} - Sequential sampling: Could not match {live_id} at topological distance ' \
                      f'{temporal_length}, matched at length {chosen_topo_dist - 1} instead.')

            # Get the map vertex we want to localize to.
            map_id = (map_run_id, map_pose_id)
            map_vertex = pose_graph.get_vertex(map_id)

            if map_vertex is None:
                print(f'{path_name} - Sequential sampling: Could not localize {live_id} at topological distance ' \
                      f'{temporal_length} (map vertex is None).')
                continue

            # Create a sample ID and check that it has not already been added
            sample_id = f'{path_name}-{live_id[0]}-{live_id[1]}-{map_id[0]}-{map_id[1]}'
            
            if not sample_id in samples:
                
                T_map_live = pose_graph.get_transform(live_id, map_id)
                log_map_live = Transform.LogMap(T_map_live)

                samples.append(sample_id)
                labels_se3[sample_id] = torch.tensor(T_map_live.matrix, dtype=torch.float)
                labels_log[sample_id] = torch.tensor(log_map_live, dtype=torch.float)

            else:

                print(f'{path_name} - Sequential sampling: sample {sample_id} has already been added')

    return samples, labels_se3, labels_log