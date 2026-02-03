import numpy as np
from RRTTree import RRTTree
import time

class RRTPlanner(object):
    def __init__(self, planning_env, ext_mode, goal_prob, nums_of_runs=10):
        # Set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # Set search parameters
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.nums_of_runs = nums_of_runs  # Statistical results across multiple runs

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        avg_cost = 0
        avg_time = 0
        runs_num = self.nums_of_runs

        for cnt in range(runs_num):
            start_time = time.time()
            plan = []
            env = self.planning_env
            start_state = env.start  # Initialize start state
            goal_state = env.goal  # Initialize goal state
            self.tree.add_vertex(start_state)  # Create the first node in the tree

            # Planning loop
            while not self.tree.is_goal_exists(state=goal_state):
                # Sample a random state with goal bias
                sampled_state = self.sample_random_state()

                # Find the nearest state in the tree and extend towards the sampled state
                nearest_neighbor_id, nearest_neighbor_state = self.tree.get_nearest_state(state=sampled_state)
                new_state = self.extend(nearest_neighbor_state, sampled_state)

                # Check if the new edge is valid (no collisions)
                if self.is_valid_edge(nearest_neighbor_state, new_state):
                    self.tree.add_vertex(state=new_state)
                    cost = env.compute_distance(start_state=nearest_neighbor_state, end_state=new_state)
                    nearest_new_id = self.tree.get_idx_for_state(state=new_state)
                    self.tree.add_edge(sid=nearest_neighbor_id, eid=nearest_new_id, edge_cost=cost)

            # Reconstruct the path from the goal to the start
            plan = self.reconstruct_path(goal_state)
            total_cost = self.compute_cost(plan)

            run_time = time.time() - start_time
            avg_cost += total_cost
            avg_time += run_time

            # Print stats for each run
            print(f"Run {cnt + 1} - Total cost: {total_cost:.2f}, Time: {run_time:.2f}")

        # Print average stats after all runs
        print(f"\nAverage cost of path: {avg_cost / runs_num:.2f}")
        print(f"Average time: {avg_time / runs_num:.2f}")

        return np.array(plan)

    def sample_random_state(self):
        """Sample a random state with goal bias"""
        env = self.planning_env
        if np.random.uniform() < self.goal_prob:
            return env.goal  # Goal bias
        else:
            # Sample a random point in the environment
            x_coord = np.random.uniform(env.xlimit[0], env.xlimit[1])
            y_coord = np.random.uniform(env.ylimit[0], env.ylimit[1])
            return np.array([x_coord, y_coord])

    def is_valid_edge(self, state1, state2):
        """Check if the edge between two states is free from collisions"""
        env = self.planning_env
        return env.state_validity_checker(state=state2) and env.edge_validity_checker(state1=state1, state2=state2)

    def extend(self, near_state, rand_state):
        """Extend from near_state towards rand_state"""
        env = self.planning_env
        delta = rand_state - near_state
        delta_norm = env.compute_distance(start_state=rand_state, end_state=near_state)
        normalized_direction_vector = delta / delta_norm
        eta = self.get_eta(delta_norm)

        if self.ext_mode == 'E1':
            return rand_state
        elif self.ext_mode == 'E2':
            # Extend with eta constraint
            return near_state + eta * normalized_direction_vector if delta_norm > eta else rand_state
        else:
            raise ValueError("Unknown extension mode.")

    def get_eta(self, delta_norm):
        """Return the eta value based on the distance"""
        # Dynamically determine eta or use a fixed value for testing
        eta_vec = [5, 10, 15]
        # For simplicity, choose the best eta based on empirical testing (e.g., eta=10)
        return eta_vec[1]  # Here you can dynamically choose based on distance if needed

    def reconstruct_path(self, goal_state):
        """Reconstruct the path from goal to start"""
        plan = []
        curr_state_id = self.tree.get_idx_for_state(state=goal_state)
        while curr_state_id != self.tree.get_root_id():
            curr_state = self.tree.vertices[curr_state_id].state
            plan.append(curr_state)
            curr_state_id = self.tree.edges[curr_state_id]
        # Insert the root and reverse the path to make it from start to goal
        plan.append(self.tree.vertices[curr_state_id].state)
        plan.reverse()
        return plan

    def compute_cost(self, plan):
        """Compute and return the plan cost (sum of distances between consecutive states)"""
        cost = sum(np.linalg.norm(plan[i+1] - plan[i]) for i in range(len(plan) - 1))
        return cost
