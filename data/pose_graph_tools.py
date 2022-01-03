"""Copied from vtr-dataset-tools: https://github.com/utiasASRL/vtr-dataset-tools/blob/master/tools.py"""

import copy
import numpy as np

from src.utils.transform import Transform

class Vertex:
    """
    The Vertex object represents a vertex that is part of a pose graph.
    Args:
        run_id (int): Id of the run the vertex belongs to.
        pose_id (int): Id of the pose for the vertex.
        next_run_id (int): Id of the run of the vertex that this vertex is connected to, 
            either spatially (from repeat to teach) or temporally (one teach vertex to another).
        next_pose_id (int): Id of the pose of the vertex that this vertex is connected to, 
            either spatially (from repeat to teach) or temporally (one teach vertex to another).
        next_transform (Transform): Transform from this to the connected vertex.
        teach (bool): True if vertex belongs to a teach run, False if repeat run. 
        prev_run_id (int): Id of the previous consecutive run (optional).
        prev_pose_id (int): Id of the previous consecutive pose (optional).
        timestamp (float): timestamp in seconds of the image associated with the vertex (optional). 
        gps_time (float): timestamp in seconds of the GPS measurement associated with the vertex (optional). 
        latitude (float): GPS measurement associated with the vertex (optional).
        longitude (float): GPS measurement associated with the vertex (optional).
        altitude (float): GPS measurement associated with the vertex (optional).
    """
    def __init__(self, run_id, pose_id, next_run_id, next_pose_id, next_transform, teach=False, prev_run_id=-1,
                 prev_pose_id=-1, timestamp=None, gps_time=None, latitude=None, longitude=None, altitude=None):
        self.vertex_id = (int(run_id), int(pose_id))
        self.next_id = (int(next_run_id), int(next_pose_id))
        self.next_transform = next_transform
        self.prev_id = (int(prev_run_id), int(prev_pose_id))
        self.teach = teach
        self.timestamp = timestamp
        self.gps_time = gps_time
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude


class Graph:
    '''
    The Graph object represents a pose graph with a teach run and several repeat runs. Each run has 
    several sequentially connected vertices. Vertices can also be connected between different runs. 
    Args:
        teach_file (str): Path to file containing temporal transforms for teach run.
        repeat_files (list of str): Paths to files containing spatial transformations for repeat runs.
        im_time_files (list of str): Paths to files containing image time stamps for runs.
        gps_files (list of str): Paths to files contaning gps data for runs.
        gps_time_files (list of str): Paths to files containing gps time stamps for runs.
    '''    
    def __init__(self, teach_file, repeat_files=None, im_time_files=None, gps_files=None, gps_time_files=None):    
        if repeat_files is None:
            repeat_files = []
        if im_time_files is None:
            im_time_files = {}
        if gps_files is None:
            gps_files = {}
        if gps_time_files is None:
            gps_time_files = {}

        # Read teach run transforms from file
        transforms_temporal = np.loadtxt(teach_file, delimiter=",")
        self.teach_id = int(transforms_temporal[0, 0])
        self.vertices = {}
        self.matches = {}

        # Add teach run vertices
        for row in transforms_temporal:
            transform = Transform(np.array([row[4:7], row[8:11], row[12:15]]), np.array([row[7], row[11], row[15]]))
            self.add_vertex(row[0], row[1], row[2], row[3], transform, True)

        self.add_prev_ids()

        # If repeat run transforms, add those to graph
        for run_file in repeat_files:
            self.add_run(run_file)

        self.add_timestamps(im_time_files)
        self.add_gps(gps_files, gps_time_files)

    def get_vertex(self, vertex_id):
        """Returns a vertex in the graph.
        Args:
            vertex_id (tuple of ints): Id of the graph vertex object. The id is a tuple containing
                the run id and the the pose id.
        Returns:
            Vertex: The vertex object, None if the vertex does not exist.         
        """
        return self.vertices.get(vertex_id)

    def get_all_vertices(self):

        return self.vertices

    def is_vertex(self, vertex_id):
        """Returns whether a vertex object corresponding to the given vertex id exists in the graph.
        Args:
            vertex_id (tuple of ints): Id of the graph vertex object. The id is a tuple containing 
                the run id and the the pose id.
        Returns:
            bool: True if vertex exists in graph, False otherwise.         
        """
        if vertex_id in self.vertices:
            return True
        else:
            return False

    def add_vertex(self, run_id, pose_id, next_run_id, next_pose_id, next_transform, teach=False):
        """Adds a new vertex to the graph.
        Args:
            run_id (int): Id of the run the vertex belongs to.
            pose_id (int): Id of the pose for the vertex.
            next_run_id (int): Id of the run of the vertex that this vertex is connected to, 
                either spatially (from repeat to teach) or temporally (one teach vertex to another).
            next_pose_id (int): Id of the pose of the vertex that this vertex is connected to, 
                either spatially (from repeat to teach) or temporally (one teach vertex to another).
            next_transform (Transform): Transform from this to the connected vertex. 
        """
        v = Vertex(run_id, pose_id, next_run_id, next_pose_id, next_transform, teach)

        # Check if vertex already exists
        if self.is_vertex(v.vertex_id):
            print("Vertex {0} already exists in graph".format(v.vertex_id))
            return

        # If teach vertex, create an empty list of repeat vertices that have matched,
        # otherwise check that teach vertex exists and add this vertex to its list of matches
        if teach:
            self.matches[v.vertex_id] = []
            self.vertices[v.vertex_id] = v
        elif v.next_id in self.vertices:
            self.matches[v.next_id].append(v)
            self.vertices[v.vertex_id] = v
        else:
            print("Warning: teach vertex {0} not found in graph so vertex {1} was not added.".format(v.next_id, v.vertex_id))

    def add_prev_ids(self):
        """Iterates over all the teach vertices and adds the index of the previous vertex as an 
           attribute to each one.
        """
        for v_id in self.vertices:
            v = self.get_vertex(v_id)
            if v.teach and v.next_id in self.vertices:
                self.vertices[v.next_id].prev_id = v_id

    def add_run(self, run_file):
        """Add a new repeat run to the graph.
        Args:
            run_file (str): Path to file containing spatial transformations for the repeat run.
        
        """
        transforms_spatial = np.loadtxt(run_file, delimiter=",")
        for row in transforms_spatial:
            transform = Transform(np.array([row[4:7], row[8:11], row[12:15]]), np.array([row[7], row[11], row[15]]))
            self.add_vertex(row[0], row[1], row[2], row[3], transform, False)

    def add_timestamps(self, im_time_files):
        """Add image timestamps to vertices.
        Args:
            im_time_files (list of str): Paths to files containing image time stamps for runs.            
        """
        for run in im_time_files:
            run_times = np.loadtxt(im_time_files[run], delimiter=",")
            for row in run_times:
                if (run, int(row[0])) in self.vertices:
                    self.vertices[(run, row[0])].timestamp = float(row[1]) * 10**-9
                else:
                    print("Warning: attempted to add timestamp for vertex ({0}, {1}) not in graph.".format(run, int(row[0])))

    def add_gps(self, gps_files, gps_time_files):
        """Add GPS measurements and the associated timestamps to vertices.
        Args:
            gps_files (list of str): Paths to files contaning gps data for runs.
            gps_time_files (list of str): Paths to files containing gps time stamps for runs.            
        """
        for run in gps_files:
            run_times = np.loadtxt(gps_files[run], delimiter=",")
            for row in run_times:
                if (run, row[0]) in self.vertices:
                    self.vertices[(run, row[1])].latitude = row[2]
                    self.vertices[(run, row[1])].longitude = row[3]
                    self.vertices[(run, row[1])].altitude = row[4]
                else:
                    print("Warning: attempted to add GPS data for vertex ({0}, {1}) not in graph.".format(run, int(row[1])))

        for run in gps_time_files:
            run_times = np.loadtxt(gps_time_files[run], delimiter=",")
            for row in run_times:
                if (run, int(row[0])) in self.vertices:
                    self.vertices[(run, row[0])].gps_time = float(row[1]) * 10**-9

    def get_subgraph(self, start, end):
        """Returns a subgraph made from the teach vertices between (teach_id, start) and (
           teach_id, end), where teach_id refers to the run id of the teach run.
        Args:
            start (int): pose id of the vertex that starts the subgraph.
            end (int): pose id of the vertex that ends the subgraph. 
        Returns:
            Graph: subgraph made from teach vertices between start and end. Original graph if pose
                ids do not correspond to existing vertices.           
        """
        if self.is_vertex((self.teach_id, start)) and self.is_vertex((self.teach_id, end)):
            subgraph = copy.copy(self)
            subgraph.matches = {id: self.matches[id] for id in self.matches if start <= id[1] <= end}

            subgraph.vertices = {}
            for m in subgraph.matches:
                subgraph.vertices.update({m: self.get_vertex(m)})
                subgraph.vertices.update({repeat_vertex.vertex_id: repeat_vertex for repeat_vertex in subgraph.matches[m]})

            return subgraph
        else:
            print("Warning: invalid vertex chosen for subgraph, returning original graph.")
            return self

    def get_topological_dist(self, vertex_id1, vertex_id2):
        """Returns number of edges between two vertices in the graph.
        Args:
            vertex_id1 (tuple of ints): Id of the first vertex object. The id is a tuple containing 
                the run id and the the pose id.
            vertex_id2 (tuple of ints): Id of the second vertex object. The id is a tuple containing 
                the run id and the the pose id.
        Returns:
            int: The number of edges between the two vertices, -1 if no path exists between them.           
        """
        path, _ = self.get_path(vertex_id1, vertex_id2)
        return len(path) - 1

    def get_transform(self, vertex_id1, vertex_id2):
        """Returns the transform, T_21, from vertex 1 to vertex 2 in the graph.
        Args:
            vertex_id1 (tuple of ints): Id of the first vertex object. The id is a tuple 
                containing the run id and the the pose id.
            vertex_id2 (tuple of ints): Id of the second vertex object. The id is a tuple 
                containing the run id and the the pose id.
        Returns:
            Transform: The transform, T_21, from vertex 1 to vertex 2. Identity transform if no 
                path exists between the vertices.           
        """
        transform = Transform(np.eye(3), np.zeros((3,)))

        path, forward = self.get_path(vertex_id1, vertex_id2)

        if len(path) == 0:
            print("Warning: no path found between vertex {0} and vertex {1}, returning identity transform.".format(vertex_id1, vertex_id2))
            return transform
        if len(path) == 1:
            return transform

        # Always calculate transform in forward direction then flip if backward
        if not forward:
            path.reverse()

        # Compose transforms
        for vertex in path[:-2]:
            transform = self.get_vertex(vertex).next_transform * transform

        # Check if last vertex in path is a teach vertex
        if self.get_vertex(path[-2]).next_id == path[-1]:
            transform = self.get_vertex(path[-2]).next_transform * transform
        else:
            transform = self.get_vertex(path[-1]).next_transform.inv() * transform

        if forward:
            return transform
        else:
            return transform.inv()

    def get_path(self, vertex_id1, vertex_id2):
        """Returns a list of vertices connecting vertex 1 and vertex 2. 
           Traverses graph both ways and returns shorter path (topologically).
        Args:
            vertex_id1 (tuple of ints): Id of the first vertex object. The id is a tuple containing 
                the run id and the the pose id.
            vertex_id2 (tuple of ints): Id of the second vertex object. The id is a tuple containing 
                the run id and the the pose id.
        Returns:
            list of tuples of ints: List of vertex ids of vertices connecting vertex 1 and vertex 2. 
                List of length one if vertex 1 and 2 are the same. Empty list if no path is found 
                or if at least one of the vertices does not exist in the graph.
            bool: True if path is in the forward direction, otherwise False.            
        """
        if self.get_vertex(vertex_id1) is None or self.get_vertex(vertex_id2) is None:
            print("Warning: no path found. At least one of the vertices are invalid.")
            return [], False

        # Set to false if can't find path between vertices
        forward_valid = True
        backward_valid = True

        forward_path = [vertex_id1]

        # Check if vertex are the same
        if vertex_id1 == vertex_id2:
            return forward_path, True

        # Follow chain of vertices from vertex 1 to vertex 2
        start = self.get_vertex(vertex_id1)
        goal = self.get_vertex(vertex_id2)
        if not goal.teach:
            goal = self.get_vertex(goal.next_id)
        while start != goal:
            start = self.get_vertex(start.next_id)
            if start is None:
                forward_valid = False
                break
            else:
                forward_path.append(start.vertex_id)
        # If vertex 2 is repeat vertex add edge from teach path to vertex 2 at end
        if forward_path[-1] != vertex_id2:
            forward_path.append(vertex_id2)

        # Check other way around loop
        backward_path = [vertex_id2]
        start = self.get_vertex(vertex_id2)
        goal = self.get_vertex(vertex_id1)
        if not goal.teach:
            goal = self.get_vertex(goal.next_id)
        while start != goal:
            start = self.get_vertex(start.next_id)
            if start is None:
                backward_valid = False
                break
            else:
                backward_path.append(start.vertex_id)
        # If vertex 1 is repeat vertex add edge from teach path to vertex 1 at end
        if backward_path[-1] != vertex_id1:
            backward_path.append(vertex_id1)

        # Return shorter of the two. Returns boolean to indicate if path is in the forward direction
        if len(forward_path) <= len(backward_path) or not backward_valid and forward_valid:
            return forward_path, True
        elif backward_valid:
            backward_path.reverse()
            return backward_path, False
        else:
            print("Warning: no path found. Problem with graph.")
            return [], False

    def get_topo_neighbours(self, vertex_id, radius):
        """Returns a set of topological neighbour vertices within radius number of edges of the 
           given vertex.
        Args:
            vertex_id (tuple of ints): Id of the first vertex object. The id is a tuple containing 
                the run id and the the pose id.
            radius (int): Distance in number of edges used as the search radius.
        Returns:
            set: Set of vertex ids for neighbour vertices. Set of size 1 if radius is 0. 
                Empty set if radius is negative or if the given vertex does not exist in the graph.            
        """
        if not self.is_vertex(vertex_id):
            print("Warning: no topo neighbours found. Vertex {0} does not exist.".format(vertex_id))
            return set()

        if radius < 0:
            raise ValueError("Negative radius specified.")

        neighbours = {vertex_id}
        if radius == 0:
            return neighbours

        search_radius = 0

        v = self.get_vertex(vertex_id)
        if not v.teach:
            neighbours.add(v.next_id)
            left_bound = v.next_id
            right_bound = v.next_id
            search_radius += 1
        else:
            left_bound = vertex_id
            right_bound = vertex_id

        while search_radius < radius:
            neighbours.update(m.vertex_id for m in self.matches[left_bound])
            neighbours.update(m.vertex_id for m in self.matches[right_bound])
            left_bound = self.get_vertex(left_bound).prev_id
            right_bound = self.get_vertex(right_bound).next_id
            neighbours.add(left_bound)
            neighbours.add(right_bound)
            search_radius += 1

        return neighbours

    def get_metric_neighbours(self, vertex_id, radius):
        """Returns a set of metric neighbour vertices within radius meters of the given vertex.
        Args:
            vertex_id (tuple of ints): Id of the first vertex object. The id is a tuple containing 
                the run id and the the pose id.
            radius (float): Distance in meters used as the search radius.
        Returns:
            set: Set of vertex ids for neighbour vertices. Empty set if radius is negative 
                or if the given vertex does not exist in the graph.            
        """
        if not self.is_vertex(vertex_id):
            print("Warning: no metric neighbours found. Vertex {0} does not exist.".format(vertex_id))
            return set()

        if radius < 0:
            raise ValueError("Negative radius specified.")
            
        neighbours = {vertex_id}
        v = self.get_vertex(vertex_id)
        if not v.teach:
            teach_v = self.get_vertex(v.next_id)
            teach_T = v.next_transform
        else:
            teach_v = v
            teach_T = Transform(np.eye(3), np.zeros((3,)))

        # Search forward direction
        query_v = teach_v
        query_T = teach_T
        while True:
            # Test repeat vertices localized to current teach
            for m in self.matches[query_v.vertex_id]:
                T = m.next_transform.inv() * query_T
                if np.linalg.norm(T.r_ab_inb) < radius:
                    neighbours.add(m.vertex_id)

            # Update current teach
            if np.linalg.norm(query_T.r_ab_inb) < radius:
                neighbours.add(query_v.vertex_id)
                query_T = query_v.next_transform * query_T
                query_v = self.get_vertex(query_v.next_id)
            else:
                break

        # Search backward direction
        query_v = teach_v
        query_T = teach_T
        while True:
            # Update current teach. Order flipped to avoid checking closest teach vertex twice
            if np.linalg.norm(query_T.r_ab_inb) < radius:
                neighbours.add(query_v.vertex_id)
                query_v = self.get_vertex(query_v.prev_id)
                query_T = query_v.next_transform.inv() * query_T
            else:
                break
            # Test repeat vertices localized to current teach
            for m in self.matches[query_v.vertex_id]:
                T = m.next_transform.inv() * query_T
                if np.linalg.norm(T.r_ab_inb) < radius:
                    neighbours.add(m.vertex_id)

        return neighbours
