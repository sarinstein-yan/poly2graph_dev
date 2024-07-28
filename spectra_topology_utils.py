import numpy as np
from numba import njit
import networkx as nx
import tensorflow as tf
import torch
from skimage.filters import laplace, gaussian
from skimage.filters import threshold_mean, threshold_triangle, threshold_li
from skimage.morphology import skeletonize
# from skimage.morphology import thin, medial_axis
import matplotlib.pyplot as plt

#################### skeleton_graph ########################

def _neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

@njit # my mark
def _mark(img, nbs): # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p]==0:continue
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2

@njit # trans index to r, c...
def _idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    rst -= 1
    return rst
    
@njit # fill a node (two or more points)
def _fill(img, p, num, nbs, acc, buf):
    img[p] = num
    buf[0] = p
    cur = 0; s = 1; iso = True;
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==2:
                img[cp] = num
                buf[s] = cp
                s+=1
            if img[cp]==1: iso=False
        cur += 1
        if cur==s:break
    return iso, _idx2rc(buf[:s], acc)

@njit # trace the edge and use a buffer, then buf.copy, if using [] numba doesn't work
def _trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0;
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, _idx2rc(buf[:cur+1], acc))
   
@njit # parse the image then get the nodes and edges
def _parse_struc(img, nbs, acc, iso, ring):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64) # 2**17 = 131072
    num = 10
    nodes = []
    for p in range(len(img)):
        if img[p] == 2:
            isiso, nds = _fill(img, p, num, nbs, acc, buf)
            if isiso and not iso: continue
            num += 1
            nodes.append(nds)
    edges = []
    for p in range(len(img)):
        if img[p] <10: continue
        for dp in nbs:
            if img[p+dp]==1:
                edge = _trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    if not ring: return nodes, edges
    for p in range(len(img)):
        if img[p]!=1: continue
        img[p] = num; num += 1
        nodes.append(_idx2rc([p], acc))
        for dp in nbs:
            if img[p+dp]==1:
                edge = _trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges
    
# use nodes and edges to build a networkx graph
def build_graph(nodes, edges, multi=False, full=True, DOS_image=None):
    os = np.array([i.mean(axis=0) for i in nodes])
    if full: os = os.round().astype(np.uint16)
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        if DOS_image is not None:
            node_dos = {'dos': DOS_image[os[i][0], os[i][1]]}
            # dos_list = [DOS_image[pt[0], pt[1]] for pt in nodes[i]]
            # node_dos = {'dos': np.mean(dos_list)}
        else: node_dos = {}
        graph.add_node(i, o=os[i], pts=nodes[i], **node_dos)
    for s,e,pts in edges:
        if full: pts[[0,-1]] = os[[s,e]]
        if DOS_image is not None:
            dos_list = [DOS_image[pt[0], pt[1]] for pt in pts]
            edge_dos = {'avg_dos': np.mean(dos_list)}
        else: edge_dos = {}
        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s,e, weight=l, pts=pts, **edge_dos)
    return graph

def mark_node(ske):
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = _neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    _mark(buf, nbs)
    return buf
    
def skeleton2graph(ske, multi=True, iso=False, ring=True, full=True, DOS_image=None):
    """
    Converts a skeletonized image into an NetworkX graph object. 

    Parameters:
    -----------
    ske : numpy.ndarray
        The input skeletonized image. This is typically a binary image where
        the skeletonized structures are represented by 1s and the background
        by 0s.
    
    multi : bool, optional, default: False
        If True, the function builds a multi-graph allowing multiple edges 
        between the same set of nodes. If False, only a single edge is 
        allowed between any pair of nodes.
    
    iso : bool, optional, default: True
        If True, isolated nodes (nodes not connected to any other node) 
        are included in the graph. If False, isolated nodes are ignored.
    
    ring : bool, optional, default: True
        If True, the function considers ring structures (closed loops) in 
        the skeleton. If False, ring structures are ignored.
    
    full : bool, optional, default: True
        If True, the graph nodes include the rounded coordinate arrays of the 
        original points. If False, the nodes include the full coordinates.

    DOS_image : numpy.ndarray, optional
        A 2D image of the density of states (DOS) values. If provided, the
        nodes and edges of the graph will include the DOS values as attributes.

    Returns:
    --------
    graph : networkx.Graph or networkx.MultiGraph
        A graph representation of the skeletonized image. Nodes correspond to 
        junction points and endpoints, and edges represent the skeleton lines 
        between them.

    Notes:
    ------
    - The function first pads the input skeleton image to handle edge cases 
      during processing.
    - Neighbors of each pixel in the padded image are calculated to facilitate 
      the conversion of indices.
    - The image is marked using a marking function to classify the points.
    - The marked image is parsed to extract nodes and edges.
    - A NetworkX graph is built from the parsed nodes and edges.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from spectra_topology_utils import skeleton2graph
    >>> ske = np.array([
            [0,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,0,0],
            [1,1,1,1,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0],
            [0,1,0,0,0,1,0,0,0],
            [1,0,1,0,0,1,1,1,1],
            [0,1,0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0,0,0]])
    >>> graph = skeleton2graph(ske, multi=True)
    >>> print(graph.nodes)
    >>> print(graph.edges)
    """
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = _neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    _mark(buf, nbs)
    nodes, edges = _parse_struc(buf, nbs, acc, iso, ring)
    return build_graph(nodes, edges, multi, full, DOS_image)
    
# draw the graph
def draw_skeleton_graph(img, graph, cn=255, ce=128):
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

############################################################

############### Spectra Potential Landscapes ###############

@tf.function
def poly_roots_tf_batch(c):
    """
    Calculate the roots of a monomial with coefficients `c`.
    The roots are the eigenvalues of the Frobenius companion matrix.

    Parameters
    ----------
    c : tf.Tensor, dtype=tf.complex64
        2-D tensor of polynomial coefficients ordered from low to high degree.
        The first axis is the batch axis.

    Returns
    -------
    roots : tf.Tensor, dtype=tf.complex64
        2-D tensor of roots of the polynomial.
    """
    n = c.shape[1]; batch_size = c.shape[0]
    if n < 2:
        return tf.constant([], dtype=c.dtype)
    if n == 2:
        return tf.constant(-c[:, 0] / c[:, 1], dtype=c.dtype)
    
    # Construct the Frobenius companion matrix
    lower_diagonal = tf.linalg.diag(tf.ones((batch_size, n - 2), dtype=c.dtype), k=-1)
    last_column = -c[:, :-1] / c[:, -1, None]
    last_column = tf.reshape(last_column, [batch_size, n - 1, 1])
    mat = tf.concat([lower_diagonal[..., :-1], last_column], axis=-1)
    mat = tf.reverse(mat, axis=[-2, -1]) # flip the matrix to reduce error
    
    # Calculate the eigenvalues of the companion matrix
    eigvals = tf.linalg.eigvals(mat)
    return eigvals

@torch.jit.script
def poly_roots_torch_batch(c):
    """
    Calculate the roots of a monomial with coefficients `c`.
    The roots are the eigenvalues of the Frobenius companion matrix.

    Parameters
    ----------
    c : torch.Tensor, dtype=torch.complex64
        2-D tensor of polynomial coefficients ordered from low to high degree.
        The first axis is the batch axis.

    Returns
    -------
    roots : torch.Tensor, dtype=torch.complex64
        2-D tensor of roots of the polynomial.
    """
    n = c.shape[1]
    batch_size = c.shape[0]
    if n < 2:
        return torch.empty(0, dtype=c.dtype, device=c.device)
    if n == 2:
        return -c[:, 0] / c[:, 1]
    
    # Construct the Frobenius companion matrix
    lower_diagonal = torch.diag_embed(torch.ones((batch_size, n - 2), dtype=torch.complex64, device=c.device), offset=-1)
    last_column = -c[:, :-1] / c[:, -1, None]
    last_column = last_column.view(batch_size, n - 1, 1)
    mat = torch.cat([lower_diagonal[..., :-1], last_column], dim=-1)
    mat = torch.flip(mat, dims=[-2, -1]) # flip the matrix to reduce error
    
    # Calculate the eigenvalues of the companion matrix
    eigvals = torch.linalg.eigvals(mat)
    return eigvals

def _normalize_image(image):
    """
    Normalize an image to the range [0, 1].

    Parameters
    ----------
    image : ndarray
        Input image to be normalized.

    Returns
    -------
    image : ndarray
        Normalized image with values scaled to the range [0, 1].
    """
    image -= np.min(image)
    image /= np.max(image)
    return image

def PosLoG(image, sigmas=[0,1], ksizes=[3],
           black_ridges=True, power_scaling=None):
    """
    Apply the Positive Laplacian of Gaussian (PosLoG) filter to an image.

    This filter detects continuous ridges, such as vessels, wrinkles, and rivers,
    and enhances the positive curvature regions while suppressing the negative 
    curvature regions.

    Parameters
    ----------
    image : ndarray
        Input image to be filtered.
    sigmas : iterable of floats, optional
        Sigmas used as scales for the Gaussian filter. Default is range(1, 6, 2).
    ksizes : iterable of ints, optional
        Sizes of the discrete Laplacian operator. Default is [3].
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when False, 
        it detects white ridges.
    power_scaling : float, optional
        If provided, raises the normalized image to the given power before 
        applying the filter. For spectral potential landscape, power scaling
        by 1/n (n = 3, 5, ...) would sharpen the spectral graph.

    Returns
    -------
    filtered_max : ndarray
        Filtered image with pixel-wise maximum response across all scales and 
        kernel sizes.

    Notes
    -----
    The PosLoG filter applies a Gaussian blur followed by the Laplace operator.
    Positive curvature regions are enhanced by taking the maximum of the Laplace 
    response and zero, normalizing it, and then taking the pixel-wise maximum 
    response across all specified scales and kernel sizes.
    """

    if black_ridges:
        image = -image

    image = _normalize_image(image)

    if power_scaling is not None:
        image = image**power_scaling
    
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        for ksize in ksizes:
            gauss = gaussian(image, sigma=sigma)
            lap = laplace(gauss, ksize=ksize)
            # remove negative curvature
            pos = np.maximum(lap, 0)
            # normalize to max = 1 unless all zeros
            max_val = pos.max()
            if max_val > 0: pos /= max_val
            filtered_max = np.maximum(filtered_max, pos)
            
    return filtered_max

def Phi_image(c, Emax=4, Elen=400, method=None):
    '''
    Generate the spectral potential landscape Phi(E) for a given polynomial.
    1-band only.

    Parameters
    ----------
    c : array_like
        Coefficients of the polynomial. Should be symmetric, 
        len(c) should be odd, the middle one is z^0 coefficient.
    Emax : float, optional
        Maximum energy range for the landscape. Default is 2.
    Elen : int, optional
        Number of points in the energy range. Default is 400.
    method : int or str, optional
        Method to calculate the TDL spectra:
        - 1 or 'spectral' : spectral potential landscape
        - 2 or 'diff_log' : difference of kappas corresponding
                            to the least 2 |z|'s
        - 3 or 'log_diff' : log difference of the least 2 |z|'s
        Default is None.

    Returns
    -------
    phi : ndarray
        The 2d image of the spectral potential landscape Phi(E).
    '''

    E_range = np.linspace(-Emax, Emax, Elen)
    E_pairs = np.meshgrid(E_range, E_range)
    E_complex = E_pairs[0] + 1j*E_pairs[1]

    p0 = (len(c)-1)//2
    cl = np.trim_zeros(c, 'f')
    E_pos = p0 - (len(c) - len(cl))

    c = np.trim_zeros(c)
    poly_len = len(c)
    q = poly_len - E_pos - 1 # p = E_pos
    coeff = np.zeros((E_complex.size, poly_len), dtype=np.complex64)
    for i in range(poly_len):
        coeff[:, i] = c[i]
    coeff[:, E_pos] -= E_complex.ravel()
    z = poly_roots_tf_batch(tf.constant(coeff, dtype=tf.complex64)).numpy()
    
    if method is None or method == 1 or method == 'spectral':
        # Method 1: spectral potential landscape
        betas = np.sort(np.abs(z), axis=1)[:, -q:]
        phi = np.log(np.abs(c[-1])) + np.sum(np.log(betas), axis=1)
    elif method == 2 or method == 'diff_log':
        # Method 2: kappa derived from least 2 |z|'s
        kappas = -np.log(np.sort(np.abs(z), axis=1))
        phi = kappas[:, 0] - kappas[:, 1]
    elif method == 3 or method == 'log_diff':
        # Method 3: log difference of least 2 |z|'s
        betas = np.sort(np.abs(z), axis=1)
        phi = np.log(betas[:, 1]-betas[:, 0])
    return phi.reshape(E_complex.shape)

def binarized_Phi_image(c, Emax=4, Elen=400, thresholder=threshold_mean):
    phi = Phi_image(c, Emax, Elen)
    ridge = PosLoG(phi)
    binary = ridge > thresholder(ridge)
    return binary

def Phi_graph(c, Emax=4, Elen=400, thresholder=threshold_mean, DOS_feature=True):
    phi = Phi_image(c, Emax, Elen)
    ridge = PosLoG(phi)
    binary = ridge > thresholder(ridge)
    ske = skeletonize(binary, method='lee')
    DOS_image = ridge if DOS_feature else None
    # multiplier = 10 * Emax/Elen if edge_weight_normalize else 1
    graph = skeleton2graph(ske, DOS_image=DOS_image)
    attrs = {'polynomial_coeff': c, 'Emax': Emax, 'Elen': Elen}
    graph.graph.update(attrs)
    return graph

def draw_image(image, ax=None, overlay_graph=False, ax_set_kwargs={}):
    def to_graph(img):
        ske = skeletonize(img, method='lee')
        return skeleton2graph(ske)

    if ax is None: ax = plt.gca()
    ax.imshow(image, cmap='bone')
    ax.set(xlabel='Re(E)', ylabel='Im(E)', **ax_set_kwargs)
    ax.axis('off')

    if overlay_graph:
        overlay_graph = to_graph(image)
        for (s, e, key) in overlay_graph.edges(keys=True):
            ps = overlay_graph[s][e][key]['pts']
            ax.plot(ps[:,1], ps[:,0], 'g-', lw=1, alpha=0.8)
        nodes = overlay_graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax.plot(ps[:,1], ps[:,0], 'r.')

############################################################

######################### Line Graph #######################
from functools import partial

def LG_undirected(G, selfloops=False, create_using=None, L_edge_attr_dim=1):
    """Returns the line graph L of the (multi)-graph G.

    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.

    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    L_edge_attr_dim : int
        Number of edge attributes to be included in the line graph. If 1, only
        the common node's attributes are included. If 3, all attributes from
        the three nodes connected by the edges a and b are included.

    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Graph specific functions for edges.
    get_edges = partial(G.edges, keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
    
    # Determine if we include self-loops or not.
    shift = 0 if selfloops else 1

    # Introduce numbering of nodes
    node_index = {n: i for i, n in enumerate(G)}

    # Lift canonical representation of nodes to edges in line graph
    edge_key_function = lambda edge: (node_index[edge[0]], node_index[edge[1]])

    if L_edge_attr_dim != 1 and L_edge_attr_dim != 3:
        raise ValueError("L_edge_attr_dim must be 1 or 3")

    edges = set()
    for u in G:
        # Label nodes as a sorted tuple of nodes in original graph.
        # Decide on representation of {u, v} as (u, v) or (v, u) depending on node_index.
        # -> This ensures a canonical representation and avoids comparing values of different types.
        nodes = [tuple(sorted(x[:2], key=node_index.get)) + (x[2],) for x in get_edges(u)]

        if len(nodes) == 1:
            # Then the edge will be an isolated node in L.
            edge = nodes[0]
            canonical_edge = (min(edge[0], edge[1]), max(edge[0], edge[1]), edge[2])
            L.add_node(canonical_edge, **G.get_edge_data(*edge[:3]))
  
        for i, a in enumerate(nodes):
            canonical_a = (min(a[0], a[1]), max(a[0], a[1]), a[2])
            L.add_node(canonical_a, **G.get_edge_data(*a[:3]))  # Transfer edge attributes to node
            for b in nodes[i + shift:]:
                canonical_b = (min(b[0], b[1]), max(b[0], b[1]), b[2])
                edge = tuple(sorted((canonical_a, canonical_b), key=edge_key_function))
                if edge not in edges:
                    if L_edge_attr_dim == 1:
                        # find the common node u. TODO: modify for self-loops
                        u = set(a[:2]).intersection(set(b[:2])).pop()
                        attr = G.nodes[u]
                    elif L_edge_attr_dim == 3:
                        # Combine attributes from all nodes connected by the edges a and b
                        attr = {}
                        for key in G.nodes[a[0]]:
                            attr[f"{key}_{a[0]}"] = G.nodes[a[0]][key]
                        for key in G.nodes[a[1]]:
                            attr[f"{key}_{a[1]}"] = G.nodes[a[1]][key]
                        for key in G.nodes[b[0]]:
                            attr[f"{key}_{b[0]}"] = G.nodes[b[0]][key]
                        for key in G.nodes[b[1]]:
                            attr[f"{key}_{b[1]}"] = G.nodes[b[1]][key]
                    L.add_edge(canonical_a, canonical_b, **attr)
                    edges.add(edge)
                    # print(f"Added edge: {canonical_a} -> {canonical_b} with attributes {attr}")
    return L






#################### Experimental ####################

def OBC_spectra_degree7_1band(c, N, pbc=False):
    '''
    Calculate the energy spectrum of the OBC Hamiltonian
    
    Parameters:
    -----------
    c: array_like
        Coefficients of the Hamiltonian. Should be symmetric, 
        len(c) should be 7, the middle one is z^0 coefficient.
    N: int
        Number of sites in the chain.
    pbc: bool, optional
        If True, use periodic boundary condition. Default is False.

    Returns:
    --------
    E_Re: ndarray
        Real part of the energy spectrum
    E_Im: ndarray
        Imaginary part of the energy spectrum
    '''
    H = np.eye(N, k=-3) * c[0]
    H += np.eye(N, k=-2) * c[1]
    H += np.eye(N, k=-1) * c[2]
    H += np.eye(N) * c[3]
    H += np.eye(N, k=1) * c[4]
    H += np.eye(N, k=2) * c[5]
    H += np.eye(N, k=3) * c[6]
    if pbc:
        H[-3:, :3] = H[:3, 3:6]
        H[:3, -3:] = H[3:6, :3]
    E = np.linalg.eigvals(H)
    return E.real, E.imag