import numpy as np

class RCSPlanner(object):    
    def __init__(self, planning_env):
        self.planning_env = planning_env

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = [] 

    def plan(self):
        plan = []
        
        # Define sets for coarse and fine moves:
        coarseSet = [(2,2), (2,0), (2,-2), (0,2), (0,-2), (-2,2), (-2,0), (-2,-2)]
        fineSet   = [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]
        
        # Initialize root node and data structures:
        root = {
            'state': self.planning_env.start,
            'rank': 0,
            'resolution': 'coarse',
            'parent': None
        }
        OPEN = [root]
        CLOSED = set()
        
        while OPEN:
            # Pop the node with the smallest rank
            OPEN.sort(key=lambda x: x['rank'])
            current = OPEN.pop(0)
            current_state = current['state']
            
            # Check if current state is valid
            if not self.planning_env.state_validity_checker(current_state):
                continue
            
            # Check for duplicates
            state_tuple = tuple(current_state)
            if state_tuple in CLOSED:
                continue
            CLOSED.add(state_tuple)
            self.expanded_nodes.append(current_state)
            print(current_state)
            
            # Check if the goal is reached
            if np.linalg.norm(current_state - self.planning_env.goal) < 1e-6:
                # Reconstruct the path
                while current is not None:
                    plan.append(current['state'])
                    current = current['parent']
                plan.reverse()
                break
            
            # Expand neighbors based on resolution
            # if current['resolution'] == 'coarse':
            for action in coarseSet:
                new_state = current_state + np.array(action)
                if self.planning_env.state_validity_checker(new_state) and self.planning_env.edge_validity_checker(current_state, new_state):
                    new_node = {
                        'state': new_state,
                        'rank': current['rank'] + 1,
                        'resolution': 'coarse',
                        'parent': current
                    }
                    OPEN.append(new_node)
            
            # If not root and resolution was coarse, expand fine moves from parent
            if current['parent'] is not None and current['resolution'] == 'coarse':
                parent = current['parent']
                for action in fineSet:
                    new_state = parent['state'] + np.array(action)
                    if self.planning_env.state_validity_checker(new_state) and self.planning_env.edge_validity_checker(parent['state'], new_state):
                        new_node = {
                            'state': new_state,
                            'rank': parent['rank'] + 1,
                            'resolution': 'fine',
                            'parent': parent
                        }
                        OPEN.append(new_node)

        # After reconstructing the path
        if plan:
            path_length = sum(
                np.linalg.norm(np.array(plan[i + 1]) - np.array(plan[i])) for i in range(len(plan) - 1)
            )
            print("Path:")
            for i in range(len(plan)):
                print(f"Step {i+1}: {plan[i]}, Resolution: {'coarse' if i < len(plan)-1 and np.linalg.norm(np.array(plan[i+1]) - np.array(plan[i])) > 1.5 else 'fine'}")
            print(f"Total Path Length: {path_length}")
            print(f"Total Steps: {len(plan)}")
        else:
            print("No path found.")


        
        return np.array(plan)



    def reconstruct_path(self, node):
        '''
        Reconstruct the path from the goal to the start using parent pointers.
        # YOU DON'T HAVE TO USE THIS FUNCTION!!!
        '''
        path = []
        while node:
            path.append(node.state)  # Append the state
            node = node.parent  # Move to the parent
        path.reverse()
        print(path)
        return np.array(path)
    
    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        DO NOT MODIFY THIS FUNCTION!!!
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
