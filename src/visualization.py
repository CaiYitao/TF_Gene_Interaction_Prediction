
from utils import *
import networkx as nx
import matplotlib.pyplot as plt
from GAE import GAEModel,predict_edge_labels
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np
from scipy import stats
import random

def plot_embedding_2D(embeddings, labels=None, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'TSNE':
        reducer = TSNE(n_components=2)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Unsupported method. Choose 'PCA' or 'TSNE'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    if labels is not None:
        if len(labels) != reduced_embeddings.shape[0]:
            raise ValueError("Number of labels does not match the number of embeddings.")
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    plt.title(f'2D Embedding using {method}')
    plt.colorbar()  # Only needed if labels are provided
    plt.show()
    #save plot
    plt.savefig(f'{method}_embedding.png')





# Plot function with subgraph sampling
def plot_subgraph_with_edge_labels(predicted_edge_labels, batch, gene_names, num_nodes=20):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    # Create edge weights from predicted_edge_labels
    edge_weights = predicted_edge_labels.detach().cpu().numpy()

    # Convert edge_index to list of edges with weights
    edges_with_weights = [
        (batch.edge_index[0, i].item(), batch.edge_index[1, i].item(), {"weight": edge_weights[i]})
        for i in range(batch.edge_index.size(1))
    ]

    # Create the main graph
    G = nx.DiGraph()  # Use a directed graph
    G.add_edges_from(edges_with_weights)

    # Sample a connected subgraph of num_nodes nodes
    subgraph, nodes = sample_neigh(G, num_nodes)
    
    # Filter gene names to include only nodes in the subgraph
    subgraph_gene_names = {node: gene_names[node] for node in subgraph.nodes if node in gene_names}
    
    # Create positions for the subgraph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, seed=42)  # Layout only for subgraph nodes
    
    # Create edge colors based on predicted labels
    edge_colors = []
    for u, v in subgraph.edges():
        # Find the index of the edge (u, v) in the original edge_index
        edge_idx = np.where((batch.edge_index[0] == u) & (batch.edge_index[1] == v))[0]
        if len(edge_idx) > 0:
            edge_color = 'green' if predicted_edge_labels[edge_idx].item() == 1 else 'orange'
        else:
            edge_color = 'black'  # Default color if edge is not found (shouldn't happen)
        edge_colors.append(edge_color)

    # Draw the subgraph with node labels and edge colors
    nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, arrowstyle='->', arrowsize=20, width=2)
    nx.draw_networkx_labels(subgraph, pos, labels=subgraph_gene_names, font_size=10, font_family='sans-serif')
    
    plt.title('Predicted Edge Labels in Subgraph')
    plt.show()

    #save the plotted subgraph as png

    # plt.savefig('~/plot/subgraph_with_predicted_edge_labels.png')


def main():
    
    #load model and test data
    dataset = load_data('data/expression_matrix.csv', 'data/suberites_presence_absence.csv')
    #load model
    train_data = dataset[5:100]
    test_data = dataset[0:5]
    data = test_data[0]
    model = GAEModel(in_channels=1, out_channels=32)
    model.load_state_dict(torch.load('model.pth'))
    # Assuming gene_names is a dictionary of node index to gene name
    gene_names = {i: f'sub2_g{i+1}' for i in range(data.x.size(0))}
    # predicted_edge_labels = predict_edge_labels(model, dataset[:5],classification='cluster')
    #load predicted edge labels from csv
    predicted_edge_labels = np.loadtxt('predicted_edge_labels.csv', delimiter=',').to_numpy()

    pel = predicted_edge_labels[:82068]
    embeddings = []

    for i in range(len(test_data)):
        z = model.encode(train_data[i].x, train_data[i].edge_index)
        embeddings.append(model.get_edge_embeddings(z,train_data[i].edge_index))
        
    embeddings = torch.cat(embeddings, dim=0).cpu().detach().numpy()
    # #save edge_index to csv
    # print('embeddings shape:', embeddings.shape)
    # print('predicted_edge_labels shape:', predicted_edge_labels.shape)
    # edge_index = data.edge_index.cpu().detach().numpy()
    # np.savetxt('edge_index.csv', edge_index, delimiter=',')
  
    plot_embedding_2D(embeddings[:82068], labels=predicted_edge_labels.cpu().detach().numpy()[:82068], method='TSNE')
   # plot the subgraph with predicted edge labels
    plot_subgraph_with_edge_labels(pel,data,gene_names)






if __name__ == '__main__':
    main()