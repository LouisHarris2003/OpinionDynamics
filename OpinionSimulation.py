import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import unittest


def defuant(threshold = 0.2, beta=0.2, use_network=False, num_nodes = 10, num_timesteps=100, show_network_delay=0.2):
    """
    Simulate the Deffuant model for opinion dynamics.

    :param threshold: Threshold value for opinion difference to initiate interaction
    :param beta: Weighting factor for updating opinions during interaction
    :param use_network: Boolean flag indicating whether to use a small-world network
    :param num_nodes: Number of nodes in the small-world network
    :param num_timesteps: Number of timesteps to simulate
    :param show_network_delay: Delay between network visualization updates
    :return: If not using a network, returns the population array with opinions, otherwise, returns None
    """
    if use_network == False:
        # Simulation without network
        # Create a 100x100 2D array filled with zeros
        population = np.zeros((100, 100))
        # Fill the first column with 100 random values between 0 and 1 representing 100 peoples random opinions
        population[:, 0] = np.random.rand(100)
        # copy first column to all other columns
        for i in range(len(population)):
            population[i, :] = population[i, 0]

        # Get the number of rows and columns in population
        num_rows = population.shape[0]
        # for allowed number of timesteps
        for timestep in range(num_timesteps-1):
            # randomly choose a person from the population``
            for i in range(250):
                random_person_index = random.randint(0,num_rows-1)
                # get left or right neighbour index randomly (circular)
                randomNeighbourIndex = (random_person_index + ((2*random.randint(0, 1))-1)) % num_rows
                # if two people have difference of opinion no larger than threshold
                if abs(population[random_person_index,timestep] - population[randomNeighbourIndex,timestep]) < threshold:
                    # tilt each person's and neighbour's opinions towards each other
                    population[random_person_index,timestep+1] = population[random_person_index,timestep] + beta * (population[randomNeighbourIndex,timestep] - population[random_person_index,timestep])  
                    population[randomNeighbourIndex,timestep + 1] = population[randomNeighbourIndex,timestep] + beta * (population[random_person_index,timestep] - population[randomNeighbourIndex,timestep])
                else:
                    population[random_person_index,timestep+1] = population[random_person_index,timestep]

        # Plot histogram of opinions at the last timestep
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(population[:, -1], bins=10, range=(0, 1))
        plt.xlabel('Opinion')
        plt.ylabel('Frequency')
        plt.title('Histogram of Opinion at Last Timestep')
        # Plot opinion over timestep for each person
        plt.subplot(1, 2, 2)
        for person in population:
            plt.plot(person)
        plt.xlabel('Timestep')
        plt.ylabel('Opinion')
        plt.title('Opinion Over Timestep')

        plt.suptitle(f'Coupling: {beta}, Threshold: {threshold}', fontsize=14, ha='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.show()

        return population

    else:
        # Simulation with small-world network
        # Define parameters for the Watts-Strogatz small-world graph
        p_rewiring = 0.2

        # Define array of mean opinion
        meanOpinion = np.zeros(num_timesteps)

        network = Network()
        network.make_small_world_network(num_nodes, p_rewiring)

        fig, ax = plt.subplots(figsize=(6, 6))

        # Calculate node positions in a circle
        radius = 5
        theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
        positions = np.column_stack([radius*np.cos(theta), radius*np.sin(theta)])

        for timestep in range(num_timesteps):
            ax.clear()
            for i in range(num_nodes):
                # Adding total opinions up in timestep
                neighbours = network.nodes[i].neighbors
                # Choose a random neighbor
                neighbour = random.choice(neighbours)
                if abs(network.nodes[i].value - network.nodes[neighbour].value) < threshold:
                    # Update opinions based on the threshold condition
                    network.nodes[i].value += beta * (network.nodes[neighbour].value - network.nodes[i].value)
                    network.nodes[neighbour].value += beta * (network.nodes[i].value - network.nodes[neighbour].value)
            for i in range(num_nodes):
                meanOpinion[timestep] += network.nodes[i].value
            node_colors = [network.nodes[i].value for i in range(num_nodes)]
            # plot nodes
            plt.scatter(positions[:, 0], positions[:, 1], s=100, c=node_colors, cmap=plt.cm.Reds)
            # Plot connections
            for i in range(num_nodes):
                x1, y1 = positions[i]
                for neighbor in network.nodes[i].neighbors:
                    x2, y2 = positions[neighbor]
                    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
            
            ax.set_title('Opinion Dynamics on Small-World Network (Timestep {})'.format(timestep))
            plt.axis('off')  # Turn off axis
            plt.pause(show_network_delay)

        # Finding mean of each timestep
        meanOpinion /= num_nodes

        # Plotting the mean opinion over all timesteps
        plt.close()
        plt.plot(range(num_timesteps), meanOpinion)
        plt.xlabel('Timestep')
        plt.ylabel('Mean Opinion')
        plt.title('Mean Opinion Over 100 Timesteps')
        plt.grid(True)
        plt.show()
        

# Function to parse command-line arguments
def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Model for opinion dynamics in populations")
    # Add the main flags
    parser.add_argument('--test_defuant', action='store_true', help='Add this flag to enable test_defuant')
    parser.add_argument('--defuant', action='store_true', help='Add this flag to enable defuant')
    parser.add_argument('--use_network', type=int, default=None, help='add this flag to enable small worlds network with a value for size of network')

    # Add optional flags with default values
    parser.add_argument('--beta', type=float, default=0.2, help='Value for beta (default: 0.2)')
    parser.add_argument('--threshold', type=float, default=0.2, help='Value for threshold (default: 0.2)')

    return parser.parse_args()

class Node:
    def __init__(self, value, number, neighbors=None):
        # Initialize a Node object with a value, index number, and neighbors list
        self.value = value
        self.index = number
        self.neighbors = neighbors if neighbors else []

class Network:
    def __init__(self, nodes=None):
        # Initialize a Network object with a list of nodes
        self.nodes = nodes if nodes is not None else []

    def make_small_world_network(self, N, re_wire_prob=0.2, max_connections=2):
        # Function to create a small-world network
        
        # Create N nodes and add them to the network
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            self.nodes.append(Node(value, node_number))

        # Connect each node to its neighbors in a ring-like structure
        for index, node in enumerate(self.nodes):
            prev_index_1 = (index - 1) % N
            next_index_1 = (index + 1) % N
            prev_index_2 = (index - 2) % N
            next_index_2 = (index + 2) % N
            node.neighbors.extend([prev_index_1, next_index_1, prev_index_2, next_index_2])

        # Rewire connections with a certain probability while ensuring maximum connections per node
        for index, node in enumerate(self.nodes):
            for neighbor_index in node.neighbors:
                if np.random.random() < re_wire_prob:
                    # Find available neighbors for rewiring
                    available_neighbors = [i for i in range(N) if i != index and i not in node.neighbors and i not in self.nodes[neighbor_index].neighbors]
                    if available_neighbors:
                        # Determine the number of connections to rewire
                        rewired_connections = min(max_connections, len(available_neighbors))
                        # Randomly select new neighbors to connect to
                        rewired_neighbors = np.random.choice(available_neighbors, rewired_connections, replace=False)
                        for new_neighbor_index in rewired_neighbors:
                            old_neighbor_index = node.neighbors.index(neighbor_index)
                            old_neighbor = self.nodes[neighbor_index]
                            # Remove old connection
                            old_neighbor.neighbors.remove(index)
                            node.neighbors.remove(neighbor_index)
                            # Connect to the new neighbor
                            node.neighbors.insert(old_neighbor_index, new_neighbor_index)
                            self.nodes[new_neighbor_index].neighbors.append(index)
                            # After connecting to a new neighbor, break the loop to prevent multiple connections
                            break


class TestDefuantSimulation(unittest.TestCase):
    
    def test_defuant(self):
        # Unit test for the defuant function
        print("Executing test_defuant_function")
        
        
        # Call defuant function with some threshold and beta values
        
        population = defuant(threshold = 0.3, beta = 0.4)
        
        # Assert that the population array is modified after calling defuant function
        self.assertFalse(np.all(population == 0))  # Check if all elements are not zero
        
        # Assert that opinions at the last timestep have been updated
        self.assertNotEqual(population[:, -1].tolist(), [0] * 100)  # Check if any opinion is not zero
        print("No errors detected")

def main():
    # Main function to parse arguments and call appropriate functions
    # get arguments
    args = parse_args()
    # Call functions based on flags
    if args.test_defuant:
        unittest.main(argv=[''], verbosity=2, exit=False)
    if args.defuant:
        if args.use_network is not None:
            defuant(args.threshold, args.beta, True, args.use_network)
        else:
            defuant(args.threshold, args.beta, False)

if __name__ == "__main__":
    main()
