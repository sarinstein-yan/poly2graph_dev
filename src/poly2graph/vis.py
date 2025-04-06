import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .skeleton2graph import skeleton2graph
from .spectral_graph import contract_close_nodes
from skimage.morphology import skeletonize

from numpy.typing import ArrayLike
from typing import Union, Sequence, Optional, Callable, Any, Iterable, TypeVar


def draw_spectral_graph(
    image: ArrayLike,
    ax: Optional[Any] = None,
    overlay_graph: Optional[bool] = False,
    contract_threshold: Optional[Union[int, float]] = None,
    ax_set_kwargs: Optional[dict] = {},
    s2g_kwargs: Optional[dict] = {}
) -> None:
    def to_graph(img, **kwargs):
        ske = skeletonize(img, method='lee')
        graph = skeleton2graph(ske, **kwargs)
        return graph

    if ax is None: ax = plt.gca()
    ax.imshow(image, cmap='bone')
    ax.set(xlabel='Re(E)', ylabel='Im(E)', **ax_set_kwargs)
    ax.axis('off')

    if overlay_graph:
        overlay_graph = to_graph(image, add_pts=True, **s2g_kwargs)
        if contract_threshold is not None:
            overlay_graph = contract_close_nodes(overlay_graph, contract_threshold)
        for (s, e, key, ps) in overlay_graph.edges(keys=True, data='pts'):
            # ps = overlay_graph[s][e][key]['pts']
            ax.plot(ps[:,1], ps[:,0], 'b-', lw=1, alpha=0.8)
        nodes = overlay_graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax.plot(ps[:,1], ps[:,0], 'r.')

# draw the graph on the skeleton image
def draw_graph_on_skeleton(img, graph, cn=255, ce=128):
    shape = img.shape
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for (s, e) in graph.edges():
        eds = graph[s][e]
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                pts = eds[i]['pts']
                img[np.dot(pts, acc)] = ce
        else: img[np.dot(eds['pts'], acc)] = ce
    for idx in graph.nodes():
        pts = graph.nodes[idx]['pts']
        img[np.dot(pts, acc)] = cn
    return img.reshape(shape)