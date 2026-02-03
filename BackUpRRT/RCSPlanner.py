import numpy as np
import heapdict


class Node:
    def __init__(self, state, resolution, rank, parent):
        self.state = state
        self.resolution = resolution
        self.parent = parent
        self.rank = rank

    def __lt__(self, other):
        return self.rank <= other.rank  # Used by heapdict to prioritize by rank (cost)

class RCSPlanner(object):
    def __init__(self, planning_env):
        self.planning_env = planning_env

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []
        self.coarseMoves = self.generate_moves(-2, 2, 3, exclude_origin = True)  # Coarse moves
        self.fineMoves = self.generate_moves(-1, 1, 3, exclude_origin = True)                  # Fine moves

    def generate_moves(self, min_val, max_val, num_points, exclude_origin = False):
        #"Generate a grid of moves in the configuration space.
        moves = []
        x = np.linspace(min_val, max_val, num_points)
        for i in x:
            for j in x:
                if exclude_origin and (i == 0 and j == 0):
                    continue
                moves.append((i, j))
        return moves

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []
        env = self.planning_env
        
        #initialize 
        
        rootNode = Node(env.start, 'coarse', 0, None)
        closedList = set()
        openList = heapdict.heapdict()

        #Add the root node to the priority queue
        openList[str(env.start)] = rootNode
        while openList:
            # Pop the node with the smallest rank
            current_node_str, currentNode = openList.popitem()
            currentNode_state = currentNode.state

            # Check if currentNode state is valid
            if not env.state_validity_checker(currentNode_state):
                continue
            # Check for duplicates (to avoid re-expansion)
            if tuple(currentNode_state) in closedList:
                continue
            closedList.add(tuple(currentNode.state))
            self.expanded_nodes.append(currentNode.state)

            # Check if the goal is reached
            if np.all(currentNode.state == env.goal):
            # Reconstruct the path
                plan = self.reconstruct_path(currentNode)
                break

            # Expand neighbors based on resolution
            for action in self.coarseMoves:
                newNode_state = currentNode_state + np.array(action)
                if env.state_validity_checker(newNode_state) and env.edge_validity_checker(currentNode_state, newNode_state):
                    newNode = Node(newNode_state, 'coarse', currentNode.rank + 1, currentNode)
                    openList[str(newNode_state)] = newNode  # Add new state to the priority queue

            # Transition to fine resolution after a coarse expansion
            if currentNode.parent is not None and currentNode.resolution == 'coarse':
                for action in self.fineMoves:
                    currentNode_state = currentNode.state
                    newNode_state = currentNode_state + np.array(action)
                    if env.state_validity_checker(newNode_state) and env.edge_validity_checker(currentNode_state, newNode_state):
                        newNode = Node(newNode_state, 'fine', currentNode.rank + 1, currentNode)
                        openList[str(newNode_state)] = newNode
        # TODO: Task 4
        return np.array(plan)
        
    def reconstruct_path(self, node):
        '''
        Reconstruct the path from the goal to the start using parent pointers.
        # YOU DON'T HAVE TO USE THIS FUNCTION!!!
        '''
        path = []
        while node:
            path.append(node.state)  # Append the state
            node = node.parent       # Move to the parent
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
