import numpy as np
from numba import njit
import networkx as nx
import tensorflow as tf
import torch
import cv2
from skimage.filters import laplace, gaussian
from skimage.filters import threshold_mean, threshold_triangle, threshold_li
from skimage.morphology import skeletonize
# from skimage.morphology import thin, medial_axis
import matplotlib.pyplot as plt

from typing import Union

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
    # buf = np.zeros(1048576, dtype=np.int64) # 2**20 = 1048576
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
def build_graph(nodes, edges, multi=False, full=True, 
                Potential_image=None, DOS_image=None, add_pts=True):
    
    os = np.array([i.mean(axis=0) for i in nodes])
    if full: os = os.round().astype(np.uint16)

    graph = nx.MultiGraph() if multi else nx.Graph()

    for i in range(len(nodes)):
        if DOS_image is not None:
            node_dos = {'dos': DOS_image[os[i][0], os[i][1]]}
            # dos_list = [DOS_image[pt[0], pt[1]] for pt in nodes[i]]
            # node_dos = {'dos': np.mean(dos_list)}
        else: node_dos = {}

        if Potential_image is not None:
            node_pot = {'potential': Potential_image[os[i][0], os[i][1]]}
        else: node_pot = {}

        graph.add_node(i, o=os[i], **node_dos, **node_pot)

        # if add_pts: graph.nodes[i]['pts'] = nodes[i]

    for s,e,pts in edges:
        if full: pts[[0,-1]] = os[[s,e]]

        if DOS_image is not None:
            dos_list = [DOS_image[pt[0], pt[1]] for pt in pts]
            edge_dos = {'avg_dos': np.mean(dos_list)}
        else: edge_dos = {}

        if Potential_image is not None:
            pot_list = [Potential_image[pt[0], pt[1]] for pt in pts]
            edge_pot = {'avg_potential': np.mean(pot_list)}
        else: edge_pot = {}

        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()

        if add_pts:
            graph.add_edge(s,e, weight=l, pts=pts, **edge_dos, **edge_pot)
        else:
            graph.add_edge(s,e, weight=l, **edge_dos, **edge_pot)

    return graph

def mark_node(ske):
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = _neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    _mark(buf, nbs)
    return buf
    
def skeleton2graph(ske, multi=True, iso=False, ring=True, full=True, 
                   Potential_image=None, DOS_image=None, add_pts=True):
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

    Potential_image : numpy.ndarray, optional
        A 2D image of the spectral potential landscape values. If provided,
        the nodes and edges of the graph will include the potential values as
        attributes.

    DOS_image : numpy.ndarray, optional
        A 2D image of the density of states (DOS) values. If provided, the
        nodes and edges of the graph will include the DOS values as attributes.

    add_pts : bool, optional, default: False
        If True, the nodes and edges of the graph will include all original
        points (pixels) that make up the skeleton lines.

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
    return build_graph(nodes, edges, multi, full, 
                       Potential_image, DOS_image, add_pts)
    
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

def normalize_image(image):
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

    image = normalize_image(image)

    if power_scaling is not None:
        image = image**power_scaling
    
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        for ksize in ksizes:
            gauss = gaussian(image, sigma=sigma) if sigma > 0 else image
            lap = laplace(gauss, ksize=ksize)
            # remove negative curvature
            pos = np.maximum(lap, 0)
            # normalize to max = 1 unless all zeros
            max_val = pos.max()
            if max_val > 0: pos /= max_val
            filtered_max = np.maximum(filtered_max, pos)
            
    return filtered_max

def Phi_image(
    c: np.ndarray, 
    Emax: Union[int, float, list, np.ndarray],
    Elen: int = 400,
    method: Union[None, int, str] = None
) -> np.ndarray:
    '''
    Generate the spectral potential landscape Phi(E) for a given polynomial.
    1-band only.

    Parameters
    ----------
    c : array_like
        Coefficients of the polynomial. Should be symmetric, 
        len(c) should be odd, the middle one is z^0 coefficient.
    Emax : float, optional
        Maximum energy range for the landscape. 
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

    if not isinstance(c, np.ndarray): c = np.array(c)
    if isinstance(Emax, (int, float)):
        E_re_min, E_re_max, E_im_min, E_im_max = -Emax, Emax, -Emax, Emax
    else:
        E_re_min, E_re_max, E_im_min, E_im_max = Emax
    E_re_ls = np.linspace(E_re_min, E_re_max, Elen)
    E_im_ls = np.linspace(E_im_min, E_im_max, Elen)
    E_pairs = np.meshgrid(E_re_ls, E_im_ls)
    E_complex = E_pairs[0] + 1j*E_pairs[1]

    c_nonzero = c != 0
    l0 = np.argmax(c_nonzero) # first non-zero index
    m0 = len(c_nonzero)//2 # middle index
    r0 = np.argmax(c_nonzero.cumsum()) # last non-zero index
    z0 = m0 - l0 # position of z^0 after trimming zeros
    q = r0 - m0 # largest power of z

    c = c[l0:r0+1] # trim the zeros
    # create the coefficients
    coeff = np.zeros((E_complex.size, len(c)), dtype=np.complex64)
    for i in range(len(c)):
        coeff[:, i] = c[i]
    coeff[:, z0] -= E_complex.ravel()
    z = poly_roots_tf_batch(tf.constant(coeff, dtype=tf.complex64)).numpy()
    
    if method is None or method == 1 or method == 'spectral':
        # Method 1: spectral potential landscape
        betas = np.sort(np.abs(z), axis=1)[:, -q:]
        phi = np.log(np.abs(coeff[:, -1])) + np.sum(np.log(betas), axis=1)
    elif method == 2 or method == 'diff_log':
        # Method 2: kappa derived from least 2 |z|'s
        kappas = -np.log(np.sort(np.abs(z), axis=1))
        phi = kappas[:, 0] - kappas[:, 1]
    elif method == 3 or method == 'log_diff':
        # Method 3: log difference of least 2 |z|'s
        betas = np.sort(np.abs(z), axis=1)
        phi = np.log(betas[:, 1]-betas[:, 0])
    return phi.reshape(E_complex.shape)

def binarized_Phi_image(c, Emax, Elen=400, thresholder=threshold_mean):
    phi = Phi_image(c, Emax, Elen)
    ridge = PosLoG(phi)
    binary = ridge > thresholder(ridge)
    return binary

def Phi_graph(c, Emax, Elen=400, thresholder=threshold_mean, 
              Potential_feature=True, DOS_feature=True, s2g_kwargs={}):
    phi = Phi_image(c, Emax, Elen)
    ridge = PosLoG(phi)
    binary = ridge > thresholder(ridge)
    ske = skeletonize(binary, method='lee')
    Potential_image = phi if Potential_feature else None
    DOS_image = ridge if DOS_feature else None
    # multiplier = 10 * Emax/Elen if edge_weight_normalize else 1
    graph = skeleton2graph(ske, Potential_image=Potential_image, DOS_image=DOS_image, **s2g_kwargs)
    attrs = {'polynomial_coeff': c, 'Emax': Emax, 'Elen': Elen}
    graph.graph.update(attrs)
    return graph

def draw_image(image, ax=None, overlay_graph=False, ax_set_kwargs={}, s2g_kwargs={}):
    def to_graph(img, **kwargs):
        ske = skeletonize(img, method='lee')
        return skeleton2graph(ske, **kwargs)

    if ax is None: ax = plt.gca()
    ax.imshow(image, cmap='bone')
    ax.set(xlabel='Re(E)', ylabel='Im(E)', **ax_set_kwargs)
    ax.axis('off')

    if overlay_graph:
        overlay_graph = to_graph(image, add_pts=True, **s2g_kwargs)
        for (s, e, key, ps) in overlay_graph.edges(keys=True, data='pts'):
            # ps = overlay_graph[s][e][key]['pts']
            ax.plot(ps[:,1], ps[:,0], 'b-', lw=1, alpha=0.8)
        nodes = overlay_graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax.plot(ps[:,1], ps[:,0], 'r.')

############################################################

############ diagonalizing real space Hamiltonian ##########

def spectra_1band_direct_calculation(c, N, pbc=False, return_H=False):
    '''
    Calculate the energy spectrum of the OBC Hamiltonian for a given symmetric polynomial.
    
    Parameters:
    -----------
    c: array_like
        Symmetric coefficients of the Hamiltonian. len(c) should be odd, 
        with the middle one being the z^0 coefficient.
    N: int
        Number of sites in the chain.
    pbc: bool, optional
        If True, use periodic boundary conditions. Default is False.
    return_H: bool, optional
        If True, return the Hamiltonian matrix. If False, return the energy
        spectra. Default is False.

    Returns:
    --------
    E_Re: ndarray
        Real part of the energy spectrum.
    E_Im: ndarray
        Imaginary part of the energy spectrum.
    (or)
    H: ndarray
        Hamiltonian matrix.
    '''
    
    # Ensure the coefficients list is symmetric
    if len(c) % 2 == 0:
        raise ValueError("The length of coefficients 'c' must be odd."
                        " The middle coefficient is the z^0 term's.")
    
    mid_idx = len(c) // 2  # Middle index for the z^0 term
    
    # Create the Hamiltonian matrix
    H = np.zeros((N, N), dtype=np.float64)
    
    # Add the hopping terms based on the coefficients
    for i, coeff in enumerate(c):
        if coeff != 0:
            offset = i - mid_idx  # Determine the diagonal offset
            H += np.eye(N, k=offset) * coeff
            # Implement periodic boundary conditions if pbc is True
            if pbc:
                if offset > 0:
                    H += np.eye(N, k=offset-N) * coeff
                elif offset < 0:
                    H += np.eye(N, k=N+offset) * coeff

    if return_H: return H

    # Compute eigenvalues of the Hamiltonian
    E = np.linalg.eigvals(H)
    
    return E.real, E.imag

############################################################

######################### Line Graph #######################
from functools import partial

def _angle_between_vecs(v1, v2, origin=None):
    if origin is not None: v1 = v1 - origin; v2 = v2 - origin
    l1 = np.linalg.norm(v1); l2 = np.linalg.norm(v2)
    if l1 == 0 or l2 == 0: return 0
    else: return np.arccos(np.clip(np.dot(v1, v2)/(l1*l2), -1.0, 1.0))

def LG_undirected(G, selfloops=False, create_using=None, triplet_feature=False):
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
    triplet_feature : bool
        If `True`, calculate the angles between edges in the line graph.

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

    # if L_edge_attr_dim != 1 and L_edge_attr_dim != 3:
    #     raise ValueError("L_edge_attr_dim must be 1 or 3")

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
                    # find the common node u. TODO: modify for self-loops
                    u = set(a[:2]).intersection(set(b[:2])).pop()
                    attr = G.nodes[u]
                    if triplet_feature:
                        # Calculate the angle between edges
                        pos_u = attr['o']
                        v = a[0] if a[0] != u else a[1]
                        w = b[0] if b[0] != u else b[1]
                        angle = [_angle_between_vecs(G.nodes[v]['o'], G.nodes[w]['o'], origin=pos_u)]
                        if 'pts2' in G.get_edge_data(*a[:3]) and 'pts2' in G.get_edge_data(*b[:3]):
                            for pos in G.get_edge_data(*a[:3])['pts2']:
                                angle.append(_angle_between_vecs(G.nodes[w]['o'], pos, origin=pos_u))
                            for pos in G.get_edge_data(*b[:3])['pts2']:
                                angle.append(_angle_between_vecs(G.nodes[v]['o'], pos, origin=pos_u))
                        attr['angle'] = np.array(angle, dtype=np.float32)
                        attr['triplet_center'] = np.mean([G.nodes[v]['o'], G.nodes[w]['o'], pos_u], axis=0)
                        
                    # elif L_edge_attr_dim == 3:
                    #     # Combine attributes from all nodes connected by the edges a and b
                    #     attr = {}
                    #     for key in G.nodes[a[0]]:
                    #         attr[f"{key}_{a[0]}"] = G.nodes[a[0]][key]
                    #     for key in G.nodes[a[1]]:
                    #         attr[f"{key}_{a[1]}"] = G.nodes[a[1]][key]
                    #     for key in G.nodes[b[0]]:
                    #         attr[f"{key}_{b[0]}"] = G.nodes[b[0]][key]
                    #     for key in G.nodes[b[1]]:
                    #         attr[f"{key}_{b[1]}"] = G.nodes[b[1]][key]

                    L.add_edge(canonical_a, canonical_b, **attr)
                    edges.add(edge)

                    # print(f"Added edge: {canonical_a} -> {canonical_b} with attributes {attr}") # Debugging
    return L

################ Dataset Postprocessing ##############

def hash_labels(labels, n, dim=6):
    # treat as numbers in base n, convert to decimal
    base_vec = np.array([n**i for i in range(dim)])
    hash_value = base_vec @ labels.T
    unique_hash = np.unique(hash_value)
    hash_map = {hash_val: i for i, hash_val in enumerate(unique_hash)}
    reassigned_hash_value = np.array([hash_map[val] for val in hash_value])
    return reassigned_hash_value

def _average_attributes(node):
    # Check if attributes exist, otherwise initialize them
    if all(key in node for key in ['o', 'dos', 'potential']):
        sum_o = np.array(node['o'], dtype=float)
        sum_dos = node['dos']; sum_potential = node['potential']
        count = 1
    else:
        sum_o = np.zeros(2, dtype=float)  # Assuming 'o' is a 2D array based on the initial example
        sum_dos = 0.0; sum_potential = 0.0
        count = 0
    
    # If there is no 'contraction' field, return the current sums and count
    if 'contraction' not in node:
        return sum_o, sum_dos, sum_potential, count
    
    # Recursively process the contracted nodes
    for _, contracted_node in node['contraction'].items():
        o, dos, potential, n = _average_attributes(contracted_node)
        sum_o += np.array(o, dtype=float)
        sum_dos += dos; sum_potential += potential
        count += n
    
    # Avoid division by zero
    if count > 0:
        avg_o = sum_o / count
        avg_dos = sum_dos / count
        avg_potential = sum_potential / count
    else:
        avg_o = sum_o; avg_dos = sum_dos; avg_potential = sum_potential
    
    return avg_o, avg_dos, avg_potential, count

# TODO: bug, if 'dos' or 'potential' don't exist, contraction will destroy 'o'
# TODO: modify the function to handle any set of attributes
def process_contracted_graph(G):
    processed_graph = G.copy()
    for node, attr in processed_graph.nodes(data=True):
        avg_o, avg_dos, avg_potential, _ = _average_attributes(attr)
        processed_graph.nodes[node]['o'] = avg_o
        processed_graph.nodes[node]['dos'] = avg_dos
        processed_graph.nodes[node]['potential'] = avg_potential
        # Remove the contraction field as it's no longer needed
        if 'contraction' in attr:
            del processed_graph.nodes[node]['contraction']
    return processed_graph

def delete_iso_nodes(G, copy=True):
    del_G = G.copy() if copy else G
    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
    del_G.remove_nodes_from(isolated_nodes)
    return del_G

# def delete_iso_nodes(G, copy=True):
#     del_G = G.copy() if copy else G
#     return del_G.subgraph([n for n in G.nodes() if G.degree(n) > 0])

def contract_close_nodes(G, threshold):
    G = delete_iso_nodes(G) # remove isolated nodes
    contracted_graph = G.copy()
    while True:
        sorted_edges = sorted(contracted_graph.edges(data='weight'), key=lambda x: x[2])
        # Check if all edges are above the threshold
        if all(l >= threshold for _, _, l in sorted_edges):
            break
        for u, v, l in sorted_edges:
            if l < threshold:
                try:
                    temp_graph = process_contracted_graph(nx.contracted_nodes(contracted_graph, u, v, self_loops=False, copy=True))
                    # print(temp_graph.nodes(data=True)) # Debugging
                    if temp_graph.number_of_edges() == 0:
                        return G  # Return the original graph if contraction leads to no edges
                    contracted_graph = temp_graph
                except:
                    pass
    return delete_iso_nodes(contracted_graph) # remove isolated clusters



#################### Experimental ####################
