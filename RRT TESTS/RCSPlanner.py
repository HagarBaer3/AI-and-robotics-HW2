from collections import deque
import numpy as np


class RCSPlanner(object):    
    def __init__(self, planning_env):
        self.planning_env = planning_env
        self.expanded_nodes = [] 
        
        self.parent_dict = {} 
        self.resolution_dict = {}  
        self.rank_dict = {}

    def plan(self):
        '''
        Compute and return the plan using Resolution Complete Search.
        Returns a numpy array containing the states (positions) of the robot.
        '''
        start_state = np.array(self.planning_env.start)
        
        closed_list = set()
        # reverse the following lists:
        coarse_set = [(0,2), (-2,2), (-2, 0), (-2,-2), (0,-2), (2,-2), (2, 0), (2,2)]
        fine_set = [(0,1), (-1,1), (-1, 0), (-1,-1), (0,-1), (1,-1), (1, 0), (1,1)]
        
        self.resolution_dict[tuple(start_state)] = 'coarse'
        start_tuple = tuple(start_state)

        
        open_list = [(start_state, 0)]

        while open_list:

            current_state, current_state_rank =  min(open_list, key=lambda x: x[1])

            current_state_tuple = tuple(current_state)
            open_list.remove((current_state, current_state_rank))
            
            if not self.planning_env.state_validity_checker(current_state):
                continue
            current_tuple = tuple(current_state)
            if current_tuple in closed_list:
                continue
                
            self.expanded_nodes.append(current_state)            

            if np.array_equal(current_state, self.planning_env.goal):
                return self.reconstruct_path(current_state)   

            closed_list.add(current_tuple)
            for action in coarse_set:
                new_state = current_state + np.array(action)
                # use edge_validity_checker to check if the edge is valid
                if not self.planning_env.edge_validity_checker(current_state, new_state):
                    continue
                new_tuple = tuple(new_state)
                
                if new_tuple not in self.parent_dict:
                    self.parent_dict[new_tuple] = current_tuple
                    self.resolution_dict[new_tuple] = 'coarse'
                    open_list.append((new_state, current_state_rank + 1))
            
            parent_tuple = self.parent_dict.get(current_tuple)
            if parent_tuple is not None and self.resolution_dict[current_tuple] == 'coarse':
                parent_state = np.array(parent_tuple)
                for action in fine_set:
                    new_state = parent_state + np.array(action)
                    new_tuple = tuple(new_state)
                    
                    if new_tuple not in self.parent_dict:
                        self.parent_dict[new_tuple] = parent_tuple
                        self.resolution_dict[new_tuple] = 'fine'
                        
                        open_list.append((new_state, current_state_rank + 1))
        
        return np.array([])

    # def reconstruct_path(self, state):
    #     '''
    #     Reconstruct the path from goal to start using parent dictionary.
    #     '''
    #     path = []
    #     current_state = state
    #     print("I got to reconstruct_path")

    #     while tuple(current_state) in self.parent_dict:
    #         path.append(current_state)
    #         current_state = np.array(self.parent_dict[tuple(current_state)])
    #         if np.array_equal(current_state, self.planning_env.start):
    #             break
            
    #     path.append(self.planning_env.start) 
    #     return np.array(path[::-1])

    def reconstruct_path(self, state):
        '''
        Reconstruct the path and compute statistics.
        '''
        path = []
        current_state = state
        total_distance = 0
        coarse_steps = 0
        fine_steps = 0
        
        # Build path and collect statistics
        while tuple(current_state) in self.parent_dict:
            path.append(current_state)
            parent_state = np.array(self.parent_dict[tuple(current_state)])
            
            # Calculate distance
            step_distance = self.planning_env.compute_distance(current_state, parent_state)
            total_distance += step_distance
            
            # Count step types
            if self.resolution_dict[tuple(current_state)] == 'coarse':
                coarse_steps += 1
            else:
                fine_steps += 1
                
            current_state = parent_state
            
            if np.array_equal(current_state, self.planning_env.start):
                break
            
        path.append(self.planning_env.start)
        
        # Calculate statistics
        total_steps = coarse_steps + fine_steps
        coarse_percentage = (coarse_steps / total_steps * 100) if total_steps > 0 else 0
        fine_percentage = (fine_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Store statistics
        self.path_statistics = {
            'total_distance': round(total_distance, 2),
            'total_steps': total_steps,
            'coarse_steps': coarse_steps,
            'fine_steps': fine_steps,
            'coarse_percentage': round(coarse_percentage, 1),
            'fine_percentage': round(fine_percentage, 1)
        }
        
        # Print statistics
        print(f"\nPath Statistics:")
        print(f"Total path length: {self.path_statistics['total_distance']} units")
        print(f"Total steps: {self.path_statistics['total_steps']}")
        print(f"Coarse steps: {self.path_statistics['coarse_steps']} ({self.path_statistics['coarse_percentage']}%)")
        print(f"Fine steps: {self.path_statistics['fine_steps']} ({self.path_statistics['fine_percentage']}%)")
        
        return np.array(path[::-1])  
        
    # def reconstruct_path(self, node):
    #     '''
    #     Reconstruct the path from the goal to the start using parent pointers.
    #     # YOU DON'T HAVE TO USE THIS FUNCTION!!!
    #     '''
    #     import ipdb; ipdb.set_trace()
    #     path = []
    #     while node:
    #         path.append(node.state)  # Append the state
    #         node = node.parent  # Move to the parent
    #     path.reverse()
    #     print(path)
    #     return np.array(path)
    
    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        DO NOT MODIFY THIS FUNCTION!!!
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
