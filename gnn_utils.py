import torchmetrics.functional.classification as Mc

def to_pyg_data(Phi_graph, group_node_attrs=['o'], group_edge_attrs=['weight']):
    Phi_graph = Phi_graph.copy()
    for n in Phi_graph.nodes(data=True):
        if 'pts' in n[1]: del n[1]['pts']
        n[1]['o'] = n[1]['o'].astype(np.float32)
    for e in Phi_graph.edges(data=True):
        if 'pts' in e[2]: del e[2]['pts']
        e[2]['weight'] = e[2]['weight'].astype(np.float32)
    G = from_networkx(Phi_graph, group_node_attrs=group_node_attrs, group_edge_attrs=group_edge_attrs)
    return G

def to_pyg_data_L(Phi_graph, group_node_attrs=['weight'], group_edge_attrs=['o']):
    L = LG_undirected(Phi_graph)
    if L.number_of_edges() == 0:
        L = LG_undirected(Phi_graph, selfloops=True)
    for n in L.nodes(data=True):
        if 'pts' in n[1]: del n[1]['pts']
        n[1]['weight'] = n[1]['weight'].astype(np.float32)
    for e in L.edges(data=True):
        if 'pts' in e[2]: del e[2]['pts']
        e[2]['o'] = e[2]['o'].astype(np.float32)
    G = from_networkx(L, group_node_attrs=group_node_attrs, group_edge_attrs=group_edge_attrs)
    return G

def train(model, loader, val_loader, epochs=200, print_every=1, 
          top_k=1, task='multiclass', dim_out=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs+1):
        model.train()
        train_loss = 0; train_acc = 0
        train_roc = 0; train_ap = 0
        val_loss = 0; val_acc = 0

        # Train on batches
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            train_loss += loss / len(loader)
            if task == 'multiclass': kwarg = {'num_classes': dim_out}
            if task == 'multilabel': kwarg = {'num_labels': dim_out}

            train_acc += Mc.accuracy(out, data.y, task=task, top_k=top_k, **kwarg) / len(loader)
            train_roc += Mc.auroc(out, data.y, task=task, **kwarg) / len(loader)
            train_ap += Mc.average_precision(out, data.y, task=task, **kwarg) / len(loader)
            loss.backward()
            optimizer.step()

        # Print metrics every 20 epochs
        if(epoch % print_every == 0):
            val_loss, val_acc, val_roc, val_ap = test(model, val_loader, top_k, task, dim_out)
            print(f'Epoch {epoch:>3} | Train Loss: {train_loss:.2f} | Train Acc: {train_acc*100:>5.2f}% | Train ROC: {train_roc:.2f} | Train AP: {train_ap:.2f}')
            print(' '*9, f'|   Val Loss: {val_loss:.2f} |   Val Acc: {val_acc*100:.2f}% |   Val ROC: {val_roc:.2f} |   Val AP: {val_ap:.2f}')

    return model

@torch.no_grad()
def test(model, loader, top_k=1, task='multiclass', dim_out=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0; acc = 0; roc = 0; ap = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss += criterion(out, data.y) / len(loader)
        if task == 'multiclass': kwarg = {'num_classes': dim_out}
        if task == 'multilabel': kwarg = {'num_labels': dim_out}
        acc += Mc.accuracy(out, data.y, task=task, top_k=top_k, **kwarg) / len(loader)
        roc += Mc.auroc(out, data.y, task=task, **kwarg) / len(loader)
        ap += Mc.average_precision(out, data.y, task=task, **kwarg) / len(loader)
    return loss, acc, roc, ap

def group_isomorphic_graphs(graphs, labels, pyg_data_list):
    unique_graphs = []
    grouped_graphs_nx = []
    grouped_graphs_pyg = []
    grouped_labels = []

    for graph, label, pyg_data in zip(graphs, labels, pyg_data_list):
        added = False
        for i, unique_graph in enumerate(unique_graphs):
            if nx.is_isomorphic(graph, unique_graph):
                grouped_graphs_nx[i].append(graph)
                grouped_graphs_pyg[i].append(pyg_data)
                grouped_labels[i].append(label)
                added = True
                break
        if not added:
            unique_graphs.append(graph)
            grouped_graphs_nx.append([graph])
            grouped_graphs_pyg.append([pyg_data])
            grouped_labels.append([label])

    return len(unique_graphs), grouped_graphs_nx, grouped_graphs_pyg, grouped_labels

def evaluate_unique_graphs(model, grouped_graphs_pyg, grouped_labels):
    accs = []
    for pyg_graphs, labels in zip(grouped_graphs_pyg, grouped_labels):
        for pyg_data, label in zip(pyg_graphs, labels):
            pyg_data.y = torch.tensor([label], dtype=torch.long)
        loader = DataLoader(pyg_graphs, batch_size=64)
        loss, acc = test(model, loader, top_k=3)
        accs.append(acc)
    return np.array(accs)