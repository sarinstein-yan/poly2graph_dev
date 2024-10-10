import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ..poly2graph import binarized_Phi_image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def normalize_color(color, vmin=.3):
    color -= np.min(color)
    ptp = color.max()
    if ptp == 0: return np.full_like(color, vmin)
    else: return color/ptp*(1-vmin) + vmin

def visualize_attention_scores(G, pos, labels, node_att, edge_att, is_G, ax):
    node_options = {
        'node_size': 500*node_att,
        'node_color': 'C1' if is_G else 'C0',
    }
    label_options = {
        'labels': labels,
        'font_color': 'w',
        'font_weight': 'bold',
        'font_size': 10
    }
    edge_options = {
        'edge_color': edge_att,
        'edge_cmap': plt.cm.Blues if is_G else plt.cm.Greens,
        'edge_vmin': 0.,
        'width': 3*edge_att,
        'connectionstyle': "arc3,rad=0.08" if G.__class__ == nx.DiGraph else "arc3"
    }
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, **node_options)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, **label_options)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, **edge_options)
    ax.axis('off')
    return ax

def visualize_node_embeddings(x, sorted_idx, is_G, ax):
    # similarity = np.corrcoef(x[sorted_idx])
    similarity = cosine_similarity(x[sorted_idx])
    cmap = 'inferno' if is_G else 'viridis'
    if is_G: tick_labels = sorted_idx
    else: tick_labels = [chr(97+i) for i in sorted_idx]
    ax.imshow(similarity, cmap=cmap)
    ticks = np.arange(0, len(sorted_idx), 1)
    ax.set_xticks(ticks, labels=tick_labels)
    ax.set_yticks(ticks, labels=tick_labels)
    ax.grid(False)
    return ax

class ExplanationSummary():
    def __init__(self, model, dataset, grouped_graph_class_index, idx=None):
        self.model = model.cpu()
        self.dataset = dataset
        self.grouped_graph_class_index = grouped_graph_class_index
        self.embeddings = None
        if idx is not None:
            self.idx = idx
            self.get_data(idx)
            self.to_nxGraph()
    
    def get_data(self, idx):
        data_G, data_L = self.dataset[idx]
        data_G.batch = torch.zeros(data_G.num_nodes, dtype=torch.long)
        data_L.batch = torch.zeros(data_L.num_nodes, dtype=torch.long)
        self.model.eval()
        out, embeddings = self.model(data_G, data_L)
        probs = F.softmax(out, dim=1).squeeze().detach().cpu().numpy()
        y_pred = probs.argsort()[::-1]
        self.y_pred = y_pred
        self.y_pred_probs = probs[y_pred]
        self.y_true = data_G.y.item()
        self.poly_coeffs = data_G.full_coeffs.numpy()
        self.graph_iso_class = self.grouped_graph_class_index[idx]
        self.embeddings = embeddings
        self.pygG = data_G
        self.pygL = data_L

    def __call__(self, idx):
        self.idx = idx
        self.get_data(idx)
        self.to_nxGraph()
        return self

    def clear_embeddings(self):
        self.embeddings = None

    def _get_edge_att_TransformerConv(self, head):
        '''head is the index of the attention head, starting from 1'''
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        edge_att_G = self.embeddings['G_node']['att_w_conv1']
        edge_att_L = self.embeddings['L_node']['att_w_conv1']
        if head == 'all':
            return edge_att_G.sum(-1), edge_att_L.sum(-1)
        elif isinstance(head, int):
            return edge_att_G[:,head-1], edge_att_L[:,head-1]
        else:
            head = np.asarray(head)-1
            return edge_att_G[:,head].sum(-1), edge_att_L[:,head].sum(-1)

    def _get_edge_att_GATv2Conv(self, layer):
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        if isinstance(layer, int):
            if layer > self.model.num_layer_conv or layer < 2:
                raise ValueError(f'Layer must be an int between 2 and {self.model.num_layer_conv}.')
            edge_att_G = self.embeddings['G_node'][f'att_w_conv{layer}']
            edge_att_L = self.embeddings['L_node'][f'att_w_conv{layer}']
            return edge_att_G.ravel(), edge_att_L.ravel()
        elif layer == 'all':
            layer = range(2, self.model.num_layer_conv+1)
        edge_att_G = np.hstack([self.embeddings['G_node'][f'att_w_conv{i}'] for i in layer])
        edge_att_L = np.hstack([self.embeddings['L_node'][f'att_w_conv{i}'] for i in layer])
        return edge_att_G.sum(-1), edge_att_L.sum(-1)

    def _get_node_att(self,
        target_node_index_G, edge_att_G,
        target_node_index_L, edge_att_L,
        aggr=np.mean):
        '''Node attention is the average of incoming edge attention scores'''
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        node_att_G = [aggr(edge_att_G[target_node_index_G==i]) for i in range(self.pygG.num_nodes)]
        node_att_L = [aggr(edge_att_L[target_node_index_L==i]) for i in range(self.pygL.num_nodes)]
        return normalize_color(node_att_G), normalize_color(node_att_L)

    def get_node_embeddings(self, is_G=True):
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        if is_G or is_G == 'G':
            emb = [self.pygG.x.numpy()]
            emb_labels = ['G_x_input']
            for i in range(1, self.model.num_layer_conv+1):
                emb.append(self.embeddings['G_node'][f'x_conv{i}'])
                emb.append(self.embeddings['G_node'][f'x_gru{i}'])
                emb_labels.extend([f'G_x_conv{i}', f'G_x_gru{i}'])
            return emb, emb_labels
        elif not is_G or is_G == 'L':
            emb = [self.pygL.x.numpy()]
            emb_labels = ['L_x_input']
            for i in range(1, self.model.num_layer_conv+1):
                emb.append(self.embeddings['L_node'][f'x_conv{i}'])
                emb.append(self.embeddings['L_node'][f'x_gru{i}'])
                emb_labels.extend([f'L_x_conv{i}', f'L_x_gru{i}'])
            return emb, emb_labels
        else:
            raise ValueError('is_G must be a boolean or a string of "G" or "L".')

    
    def get_graph_embeddings(self):
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        emb_convG = self.embeddings['G_graph1']
        emb_convL = self.embeddings['L_graph1']
        emb_last = self.embeddings['GnL_graph-1']
        return emb_convG, emb_convL, emb_last
    
    def to_nxGraph(self, head='all', layer='all', create_using=nx.Graph):
        # get edge index
        edge_index_G = self.pygG.edge_index.numpy()
        edge_index_L = self.pygL.edge_index.numpy()

        # get edge attention scores
        edge_att_G, edge_att_L = [], []
        if head is not None:
            att_G, att_L = self._get_edge_att_TransformerConv(head)
            edge_att_G.append(att_G)
            edge_att_L.append(att_L)
        if layer is not None:
            att_G, att_L = self._get_edge_att_GATv2Conv(layer)
            edge_att_G.append(att_G)
            edge_att_L.append(att_L)
        edge_att_G = np.sum(edge_att_G, axis=0)
        edge_att_L = np.sum(edge_att_L, axis=0)

        # get node attention scores
        node_att_G, node_att_L = self._get_node_att(
                                    edge_index_G[1], edge_att_G,
                                    edge_index_L[1], edge_att_L)
        
        # Create networkx graphs
        nxG = create_using(); nxL = create_using()
        # Add nodes with positions and colors
        for i in range(self.pygG.num_nodes):
            nxG.add_node(i, pos=self.pygG.pos[i], att=node_att_G[i], label=i)
        for i in range(self.pygL.num_nodes):
            # alphabetic labels, lowercase
            nxL.add_node(i, pos=self.pygL.pos[i], att=node_att_L[i], label=chr(97+i))
        # Add edges with positions and colors
        for i, (s, e) in enumerate(edge_index_G.T):
            if nxG.has_edge(s, e):
                nxG[s][e]['att'] += edge_att_G[i]
            else:
                nxG.add_edge(s, e, att=edge_att_G[i])
        for i, (s, e) in enumerate(edge_index_L.T):
            if nxL.has_edge(s, e):
                nxL[s][e]['att'] += edge_att_L[i]
            else:
                nxL.add_edge(s, e, att=edge_att_L[i])
        # Normalize edge attributes for edge color
        edge_color_G = normalize_color(list(nx.get_edge_attributes(nxG, 'att').values()))
        edge_color_L = normalize_color(list(nx.get_edge_attributes(nxL, 'att').values()))
        # Set the normalized edge colors as attributes
        nx.set_edge_attributes(nxG, dict(zip(nxG.edges, edge_color_G)), 'att')
        nx.set_edge_attributes(nxL, dict(zip(nxL.edges, edge_color_L)), 'att')

        self.nxG = nxG
        self.nxL = nxL
        self.edge_color_G = edge_color_G
        self.edge_color_L = edge_color_L
        self.node_color_G = node_att_G
        self.node_color_L = node_att_L

        return nxG, nxL

    def summary_plot(self, path=None):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        ax = axes.flat
        emax = self.pygG.Emax.numpy()

        # plot 0: binarized phi image
        img = binarized_Phi_image(
            c=self.pygG.full_coeffs.numpy(),
            Emax=emax,
            Elen=500
        )
        ax[0].imshow(img, cmap='gray', extent=self.pygG.Emax.numpy())
        ax[0].set(title='Graph Skeleton', xlabel='Re(E)', ylabel='Im(E)')
        ax[0].grid(False)
        
        # plot 1: attention scores of G-channel
        self.to_nxGraph(head='all', layer='all')
        visualize_attention_scores(
            self.nxG,
            self.pygG.pos,
            {i: f'{i}' for i in range(self.pygG.num_nodes)},
            self.node_color_G,
            self.edge_color_G,
            is_G=True, ax=ax[1]
        )
        ax[1].set(
            title='Attention Scores of G',
            # xlim=emax[:2], ylim=emax[2:]
        )

        # plot 2: attention scores of L-channel
        visualize_attention_scores(
            self.nxL,
            self.pygL.pos,
            {i: f'{chr(97+i)}' for i in range(self.pygL.num_nodes)},
            self.node_color_L,
            self.edge_color_L,
            is_G=False, ax=ax[2]
        )
        ax[2].set(
            title='Attention Scores of L',
            xlim=ax[1].get_xlim(), ylim=ax[1].get_ylim()
        )

        # plot 3-5: node embeddings of G-channel
        emb_G, emb_labels_G = self.get_node_embeddings(is_G=True)
        sorted_idx_G = PCA(n_components=1).fit_transform(emb_G[-1]).ravel().argsort()
        for i, layer in enumerate([0,2,-1]):
            visualize_node_embeddings(emb_G[layer], sorted_idx_G, is_G=True, ax=ax[i+3])
            ax[i+3].set(title=emb_labels_G[layer])

        # plot 6-8: node embeddings of L-channel
        emb_L, emb_labels_L = self.get_node_embeddings(is_G=False)
        sorted_idx_L = PCA(n_components=1).fit_transform(emb_L[-1]).ravel().argsort()
        for i, layer in enumerate([0,2,-1]):
            visualize_node_embeddings(emb_L[layer], sorted_idx_L, is_G=False, ax=ax[i+6])
            ax[i+6].set(title=emb_labels_L[layer])

        plt.tight_layout()
        if path is not None:
            plt.savefig(path, tranparent=True)

        return fig, ax
    
    def summary_plot_per_layer(self, path=None):
        n_row = 1+self.model.num_layer_conv
        n_col = max(self.model.num_heads+2, 2*3)
        fig, axes = plt.subplots(n_row, n_col, figsize=(5*n_col, 5*n_row))

        emb_G, emb_labels_G = self.get_node_embeddings(is_G=True)
        sorted_idx_G = PCA(n_components=1).fit_transform(emb_G[-1]).ravel().argsort()
        emb_L, emb_labels_L = self.get_node_embeddings(is_G=False)
        sorted_idx_L = PCA(n_components=1).fit_transform(emb_L[-1]).ravel().argsort()

        # each head, TransformerConv
        for col in range(self.model.num_heads):
            self.to_nxGraph(head=col+1, layer=None)
            visualize_attention_scores(self.nxG, self.pygG.pos,
                {i: f'{i}' for i in range(self.pygG.num_nodes)},
                self.node_color_G, self.edge_color_G,
                is_G=True, ax=axes[0, col])
            axes[0, col].set(title=f'Transformer, Layer 1, Head {col+1}')
            visualize_attention_scores(self.nxL, self.pygL.pos,
                {i: f'{chr(97+i)}' for i in range(self.pygL.num_nodes)},
                self.node_color_L, self.edge_color_L,
                is_G=False, ax=axes[1, col])
            axes[1, col].set(title=f'Transformer, Layer 1, Head {col+1}')
        
        # each layer > 1, GATv2Conv
        for row in range(2, n_row):
            self.to_nxGraph(head=None, layer=row)
            visualize_attention_scores(self.nxG, self.pygG.pos,
                {i: f'{i}' for i in range(self.pygG.num_nodes)},
                self.node_color_G, self.edge_color_G,
                is_G=True, ax=axes[row, 0])
            axes[row, 0].set(title=f'GATv2, Layer {row}')
            visualize_attention_scores(self.nxL, self.pygL.pos,
                {i: f'{chr(97+i)}' for i in range(self.pygL.num_nodes)},
                self.node_color_L, self.edge_color_L,
                is_G=False, ax=axes[row, 3])
            axes[row, 3].set(title=f'GATv2, Layer {row}')
        
        # node embeddings, 1st layer
        for i in range(1, 3):
            visualize_node_embeddings(emb_G[i], sorted_idx_G, is_G=True, ax=axes[0, -i])
            axes[0, -i].set(title=emb_labels_G[i])
            visualize_node_embeddings(emb_L[i], sorted_idx_L, is_G=False, ax=axes[1, -i])
            axes[1, -i].set(title=emb_labels_L[i])
            
        # node embeddings, >1st layer
        for row in range(2, n_row):
            ax1 = visualize_node_embeddings(emb_G[2*row-1], sorted_idx_G, is_G=True, ax=axes[row, 1])
            ax2 = visualize_node_embeddings(emb_G[2*row], sorted_idx_G, is_G=True, ax=axes[row, 2])
            ax1.set(title=emb_labels_G[2*row-1])
            ax2.set(title=emb_labels_G[2*row])
            ax4 = visualize_node_embeddings(emb_L[2*row-1], sorted_idx_L, is_G=False, ax=axes[row, 4])
            ax5 = visualize_node_embeddings(emb_L[2*row], sorted_idx_L, is_G=False, ax=axes[row, 5])
            ax4.set(title=emb_labels_L[2*row-1])
            ax5.set(title=emb_labels_L[2*row])

        plt.tight_layout()
        if path is not None:
            plt.savefig(path, tranparent=True)

        return fig, axes