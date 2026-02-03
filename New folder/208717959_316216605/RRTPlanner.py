import numpy as np
import time
from RRTTree import RRTTree  # <-- Import the class directly

class RRTPlanner(object):
    def __init__(self, planning_env, ext_mode, goal_prob):
        self.planning_env = planning_env
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.tree = None

    def plan(self):
        start_time = time.time()
        # Create the RRTTree directly
        self.tree = RRTTree(self.planning_env)

        s, g = self.planning_env.start, self.planning_env.goal
        root_id = self.tree.add_vertex(s)  # Add the start vertex
        loop_time = time.time()
        max_iters = 10000
        new_id = root_id
        cnt_it = 0
        start_time_loop = 0
        for _ in range(max_iters):
            start_time_loop = time.time()
            cnt_it += 1
            rand_state = g if np.random.rand() < self.goal_prob else self.planning_env.sample_free()
            near_id, near_state = self.tree.get_nearest_state(rand_state)

            new_state = self.extend(near_state, rand_state)
            if self.planning_env.edge_validity_checker(near_state, new_state):
                edge_cost = np.linalg.norm(new_state - near_state)
                new_id = self.tree.add_vertex(new_state)
                self.tree.add_edge(near_id, new_id, edge_cost)
                if np.linalg.norm(new_state - g) < 1e-3:
                    break
            end_time_loop = time.time() - start_time_loop
            #print(f"loop time = {end_time_loop}")
        print(f"finished with {cnt_it}")
        plan = self.reconstruct_path(new_id)
        total_cost = self.compute_cost(plan)
        print(f"Total cost of path: {total_cost:.2f}")
        print(f"Total time: {time.time()-start_time:.2f}")
        return np.array(plan)

    def extend(self, near_state, rand_state):
        diff = rand_state - near_state
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return near_state
        direction = diff / dist
        if self.ext_mode == "E1":
            return rand_state
        elif self.ext_mode == "E2":
            eta = 10.0
            return rand_state if dist < eta else near_state + eta * direction
        else:
            raise ValueError("Unknown extension mode.")

    def reconstruct_path(self, goal_id):
        path = []
        current_id = goal_id
        while current_id != 0:
            path.append(self.tree.vertices[current_id].state)
            current_id = self.tree.edges[current_id]
        path.append(self.tree.vertices[0].state)
        path.reverse()
        return path

    def compute_cost(self, plan):
        return sum(np.linalg.norm(plan[i+1] - plan[i]) for i in range(len(plan)-1))
