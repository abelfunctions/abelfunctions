r"""Skeleton :mod:`abelfunctions.skeleton`
=================================================

This module defines the skeleton of the Riemann surface. That is, a means of
not only computing the a- and b-cycles of the first homology group of the
Riemann surface but also mechanisms for travelling from one sheet to another on
the Riemann surface.

"""
import numpy
import networkx as nx

from abelfunctions.utilities import Permutation


def find_cycle(pi, j):
    """Returns the cycle (as a list) of the permutation pi containing j.

    The ordering of a cycle is important for the homology functions since
    cycles are used to index dictionaries. For example, although "(0 7 4)" and
    "(7 4 0)" are the same cycle, this function outputs the cycles sith the
    smallest element of the cycle first.

    Parameters
    ----------
    pi : Permutation
        A permutation on {0,...,n-1}.
    j : integer
        An element in {0,...,n-1}.

    Returns
    -------
    element_cycle : tuple
        A tuple representing the ordered cycle.
    """
    if isinstance(pi, list):
        pi = Permutation(pi)

    cycles = pi._cycles
    for cycle in cycles:
        if j in cycle:
            element_cycle = tuple(reorder_cycle(cycle, min(cycle)))
            return element_cycle

def smallest(l):
    """Returns the smallest sheet number appearing in a cycle.

    The cycles of the homology are written with their smallest sheet number
    first. This function finds the smallest sheet number in the cycle l =
    (sheet, branch point, sheet, branch point, ...)

    Parameters
    ----------
    l : tuple
        A cycle on the elements {0,...,n-1}.

    Returns
    -------
    sheet_number : int
        The smallest sheet number in the cycle.
    """
    a = l[:]

    # If the first element of the cycle is a branch point then just shift
    # the cycle by one.
    if not isinstance(a[0], int):
        a = a[1:] + [a[0]]

    # Return the smallest sheet number appearing in the cycle
    sheet_number = min([a[2*i] for i in range(len(a)/2)])
    return sheet_number


def reorder_cycle(c, j=None):
    """Reorder a cycle.

    Returns a cycle (as a list) with the element "j" occuring first. If "j"
    isn't provided then assume sorting by the smallest element

    Parameters
    ----------
    c : tuple
        A cycle represented by a tuple.
    j : int, optional
        An element of {0,...,n-1}. If not provided will reorder cycle starting
        from smallest element appearing in the cycle.

    """
    n = len(c)
    try:
        if j is None:
            j = smallest(c)
        i = c.index(j)
    except ValueError:
        raise ValueError("%d does not appear in the cycle %s"%(j,c))

    reordered_cycle = [c[k%n] for k in range(i,i+n)]
    return reordered_cycle

def frobenius_transform(A, g):
    """Perform the Frobenius transform on the matrix `A`.

    This procedure brings any intersection matrix a to its canonical form
    :math:`b` by a transformation .. math::

        \alpha \times a \times \alpha^T = b.

    If :math:`2g = rank(a)` and `d` is the size of the square matrix :math:`a`,
    then b has :math:`d-2g` null rows and :math:`d-2g` null columns. These are
    moved to the lower right corner. On its diagonal, b has 2 :math:`g \times
    g` null blocks. Above the diagonal is a :math:`g \times g` identity block.
    Below the diagonal is a :math:`g \times g` negative identity block. The
    output of the procedure is the transformation matrix alpha.

    """
    if not isinstance(A, numpy.matrix):
        A = numpy.matrix(A, dtype=numpy.int)
    K = A
    dim = K.shape[0]

    # the rank of an antisymmetric matrix is always even (equal to 2g)
    T = numpy.eye(dim, dtype=numpy.int)

    # create the block below the diagonal. make zeros everywhere else in the
    # first g columns
    for i in range(g):
        counter = dim - 1

        # make sure column i has a suitable pivot by swapping rows
        # and columns
        while numpy.all( K[(g+i):,i] == numpy.zeros(dim-(g+i)) ):
            T[[i,counter],:] = T[[counter,i],:]
            K[:,[i,counter]] = K[:,[counter,i]]
            K[[i,counter],:] = K[[counter,i],:]
            counter -= 1

        # if the pivot element is zero then swap rows to make it non-zero
        if K[i+g,i] == 0:
            k = i+g+1
            while K[i+g,i] == 0:
                if K[k,i] != 0:
                    pivot = -1/K[k,i];

                    T[k,:]      *= pivot         # scale row
                    T[[k,i+g],:] = T[[i+g,k],:]  # swap rows

                    K[k,:]      *= pivot         # scale row
                    K[[k,i+g],:] = K[[i+g,k],:]  # swap rows
                    K[:,k]      *= pivot         # scale column
                    K[:,[k,i+g]] = K[:,[i+g,k]]  # swap columns

                # move to next row
                k += 1

        # otherwise, if the pivot element is non-zero then scale it so it's
        # equal to -1. this is automatically done in the zero case
        else:
            pivot = -1/K[i+g,i]
            T[i+g,:] *= pivot
            K[i+g,:] *= pivot
            K[:,i+g] *= pivot

        # use the pivot to create zeros in the rows above it and below it
        for j in list(range(i, i + g)) + list(range(i + g + 1, dim)):
            pivot = -K[j,i]/K[i+g,i]
            T[j,:] += pivot * T[i+g,:]
            K[j,:] += pivot * K[i+g,:]
            K[:,j] += pivot * K[:,i+g]

    # the block above the diagonal is already there. use it to create zeros
    # everywhere else in the second block of g columns. automatically all other
    # columns are then equal to zero, because the rank of the intersection
    # matrix K is only 2g
    for i in range(g):
        for j in range(i+g+1, dim):
            pivot = -K[j,i+g]
            T[j,:] = T[j] + pivot*T[i,:]
            K[j,:] = K[j,:] + pivot*K[i,:]
            K[:,j] = K[:,j] + pivot*K[:,i]

    # sanity check: did the Frobenius transform produce the correct
    # result?  T * K * T.T = J where J has the gxg identity I in the
    # top right block and -I in the lower left block (the Jacobian
    # matrix)
    J = numpy.dot(numpy.dot(T, numpy.matrix(A)), T.T)
    for i in range(g):
        for j in range(g):
            if j==i+g and i<g:   val = 1
            elif i==j+g and j<g: val = -1
            else:                val = 0

            if J[i,j] != val:
                raise ValueError('Could not compute Frobenuis transform of '
                                 'intersection matrix.')
    return T

def tretkoff_graph(base_point, base_sheets, monodromy_group):
    """Construct the Tretkoff graph from a monodromy group.

    There are two types of nodes:

    * sheets: (integer) these occur on the even levels

    * branch places: (complex, permutation) the first elements is the
      projection of the place in the complex x-plane. the second element is a
      cycle appearing in the monodromy element. (places above a branch point
      are in 1-1 correspondence with the cycles of the permuation) these occur
      on the odd levels

    .. note::

        This is one of the first codes I wrote and is in desperate need of
        cleaning up. Look at these nested statements!

    Parameters
    ----------
    monodromy_group : list
        The monodromy data.

    Returns
    -------
    C : networkx graph
        A graph representing the skeleton of the Riemann surface.

    """
    branch_points, monodromy = monodromy_group

    # initialize graph with base point: the zero sheet
    C = nx.Graph()
    node = (0,)
    C.add_node(node)
    C.nodes[node]['value'] = 0
    C.nodes[node]['label'] = "$0$"
    C.nodes[node]['final'] = False
    C.nodes[node]['level'] = 0
    C.nodes[node]['nrots'] = 0

    # keep track of sheets and branch places that we've already visited.
    # initialize with the zero sheet and all branch places with a stationary
    # cycle (a cycle with one element)
    covering_number = len(base_sheets)
    t = len(branch_points)
    visited_sheets = [0]
    visited_branch_places = [
        (branch_points[i], find_cycle(monodromy[i],j))
        for j in range(covering_number)
        for i in range(t)
        if len(find_cycle(monodromy[i],j)) == 1
    ]

    level = 0
    endpoints = [node]
    while endpoints:
        # obtain the endpoints on the previous level that are not
        # final and sort by their "succession" order".
        endpoints = sorted([n for n,d in C.nodes_iter(data=True)
                            if d['level'] == level and not d['final']])

        for node in endpoints:
            # determine the successors for this node. we use a different method
            # depending on what level we're on:
            #
            # if on an even level (on a sheet): the successors are branch
            # places. these are the places other than the one that is the
            # predecessor to this node.
            #
            # if on an odd level (on a branch place): the successors are
            # sheets. these sheets are simply the sheets found in the branch
            # place whose order is determined by the predecessor sheet.
            ###################################################################
            if level % 2 == 0:
                current_sheet = C.nodes[node]['value']

                # determine which branch points to add. in the initial case,
                # add all branch points. for all subsequent sheets add all
                # branch points other than the one that brought us to this
                # sheet
                if current_sheet == 0:
                    branch_point_indices = list(range(t))
                else:
                    pred = C.neighbors(node)[0]
                    bpt, pi = C.nodes[pred]['value']
                    ind = branch_points.index(bpt)
                    branch_point_indices = list(range(ind+1, t)) + list(range(ind))

                # for each branch place connecting the curent sheet to other
                # sheets, add a final edge if we've already visited the place
                # or connect it to the graph, otherwise.
                ctr = 0
                for idx in branch_point_indices:
                    succ = tuple(list(node) + [ctr])
                    bpt = branch_points[idx]
                    pi = find_cycle(monodromy[idx],current_sheet)
                    value = (bpt,pi)

                    if value in visited_branch_places:
                        final = True
                    else:
                        final = False
                        visited_branch_places.append(value)

                    if len(pi) > 1:
                        C.add_edge(node,succ)
                        C.nodes[succ]['value'] = value
                        C.nodes[succ]['label'] = "$b_{%d},%s$"%(idx,pi)
                        C.nodes[succ]['final'] = final
                        C.nodes[succ]['level'] = level+1
                        ctr += 1

            ###################################################################
            else:
                current_place = C.nodes[node]['value']
                bpt,pi = current_place

                # C is always a tree. obtain the previous node (which is the
                # source sheet) since we order cycles with the source sheet
                # appearing first.
                #
                # we also try to minimize the number of rotations performed by
                # allowing reverse rotations.
                n = len(pi)
                pred = C.neighbors(node)[0]
                previous_sheet = C.nodes[pred]['value']
                pi = reorder_cycle(pi,previous_sheet)
                ctr = 0
                for idx in range(1,n):
                    succ = tuple(list(node) + [ctr])
                    value = pi[idx]

                    if value in visited_sheets:
                        final = True
                    else:
                        final = False
                        visited_sheets.append(value)

                    C.add_edge(node,succ)
                    C.nodes[succ]['value'] = value
                    C.nodes[succ]['label'] = "$%d$"%(value)
                    C.nodes[succ]['final'] = final
                    C.nodes[succ]['level'] = level+1
                    C.nodes[succ]['nrots'] = idx if idx <= n/2 else idx-n
                    ctr += 1

        # we are done adding succesors to endpoints at this level. level up!
        level += 1

    return C


def final_edges(C):
    """Returns a list of final edges from the homology graph.

    The final edges are those that define the c-cycles on the Riemann surface.
    Note that the edges returned are such that the nodes of the edge are _both_
    final nodes.

    The final edges are ordered such that the sheet number appears first in the
    edge.

    Parameters
    ----------
    C : networkx graph
        The homology graph.

    Returns
    -------

    - list of (ordered) tuples representing the final edges

    """
    final_nodes = [n for n in C.nodes() if C.nodes[n]['final']]
    edges = []
    while len(final_nodes) > 0:
        node = final_nodes.pop()
        pred = C.neighbors(node)[0]
        pred_val = C.nodes[pred]['value']
        other = [n for n in final_nodes if C.nodes[n]['value'] == pred_val and
                 C.nodes[C.neighbors(n)[0]]['value'] == C.nodes[node]['value']]
        other = other[0]

        final_nodes.remove(other)

        # order is important: the nodes with final vertices "don't actually
        # exist" in the homology graph. they're only there to help determine
        # replative ordering of cycles. We choose final edges such that the
        # predecessors of the nodes give the correct ordering
        if isinstance(C.nodes[node]['value'],tuple):
            edges.append((other,node))
        else:
            edges.append((node,other))

    return edges


def intersection_matrix(final_edges, g):
    r"""Returns the intersection matrix from a list of final edges.

    Compute the intersection matrix of the c-cycles from the Tretkoff graph and
    final edge data output by `tretkoff_graph()`.

    Parameters
    ----------
    final_edges : list
        Each edge corresponds to a c-cycle on the Riemann surface
    g : int
        The expected genus of the riemann surface as given by
        singularities.genus()

    Returns
    -------
    K : matrix
        The intersection matrix.

    """
    def intersection_number(ei,ej):
        r"""Returns the intersection number of two edges of the Tretkoff graph.

        Note: Python is smart and uses lexicographical ordering on lists which
        is exactly what we need.
        """
        ei_start,ei_end = ei
        ej_start,ej_end = ej

        # the intersection number changes sign when a single edge is
        # reversed. normalize the edges such that the starting node of
        # each edge occurs before the ending node and that ei's starting
        # node occurs before ej's. (intersection is anti-symmetic)
        if ei_start > ei_end:
            return (-1)*intersection_number((ei[1],ei[0]),ej)
        elif ej_start > ej_end:
            return (-1)*intersection_number(ei,(ej[1],ej[0]))
        elif ei_start > ej_start:
            return (-1)*intersection_number(ej,ei)

        # after the above transformations, there is only one
        # configuration resulting in a non-zero intersection number. (24
        # total intersection possibilities / 2**3 = 3, because of three
        # binary transformations)
        if ej_start < ei_end < ej_end:
            return 1
        return 0

    # the intersection matrix is anti-symmetric, so we only determine the
    # intersection numbers of the upper triangle
    num_final_edges = len(final_edges)
    K = numpy.zeros((num_final_edges, num_final_edges), dtype=numpy.int)
    for i in range(num_final_edges):
        ei = final_edges[i]
        for j in range(i+1,num_final_edges):
            ej = final_edges[j]
            K[i,j] = intersection_number(ei,ej)

    # obtain the intersection numbers below the diagonal
    K = K - K.T

    # sanity_check: make sure the intersection matrix predicts the same genus
    # that the genus formula otuputs
    rank = numpy.linalg.matrix_rank(K)
    if rank/2 != g:
        raise ValueError('Found inconsistent genus in homolgy '
                         'intersection matrix.')
    return K


def compute_c_cycles(tretkoff_graph, final_edges):
    """Returns the c-cycles of the Riemann surface.

    Parameters
    ----------
    tretkoff_graph : networkx graph
        The Tretkoff graph

    final_edges : list
        A list of the final edges of the Tretkoff graph

    Returns
    -------
    c_cycles : list
        A list of the form

        `[s_0, (b_{i_0}, n_{i_0}), s_1, (b_{i_1}, n_{i_1}), ...]`

        where `s_k` is a sheet number, `b_{i_k}` is the `{i_k}`th branch point,
        and `n_{i_k}` is the number of times and direction to go about branch
        point `b_{i_k}`.

    """
    root = tuple([0])
    C = tretkoff_graph
    c_cycles = []

    # recall that the edges have a direction: edge[0] is the starting node and
    # edge[1] is the ending node. This determines the direction of the c-cycle.
    for final_edge in final_edges:
        # obtain the vertices on the Tretkoff graph starting from the base
        # place, going through the edge, and then back to the base_place
        #
        # see the comment in homology:final_edges() for an explanation on the
        # ordering / direction of the cycle.
        edge = [C.neighbors(n)[0] for n in final_edge]
        path_to_edge = nx.shortest_path(C,root,edge[0])
        path_from_edge = nx.shortest_path(C,edge[1],root)
        path = path_to_edge + path_from_edge
        path_values = [C.nodes[n]['value'] for n in path]

        # convert branch places (branch point, permutation) to
        # point-rotations pairs (branch point, number and direction of
        # rotations)
#         for n in range(1,len(path),2):
#             branch_place = path_values[n]

#             if n <= len(path_to_edge):
#                 next_sheet = path[n+1]
#                 nrots = C.nodes[next_sheet]['nrots']
#             else:
#                 next_sheet = path[n-1]
#                 nrots = - C.nodes[next_sheet]['nrots']
#             path_values[n] = (branch_place[0], nrots)

        # go the the sheet number in the final edge, recording number of
        # rotations normally
        for n in range(1, len(path), 2):
            bi,pi = path_values[n]
            prev_sheet = C.nodes[path[n-1]]['value']
            next_sheet = C.nodes[path[n+1]]['value']

            nrots = pi.index(next_sheet) - pi.index(prev_sheet)
            if nrots > len(pi)/2: nrots -= len(pi)

            path_values[n] = (bi, nrots)

        c_cycles.append(path_values)

    return c_cycles

def reverse_cycle(cycle):
    """Returns the reversed cycle. Note that rotation numbers around branch points
    are correctly computed.

    Parameters
    ----------
    cycle : list
        A cycle.

    Returns
    -------
    rev_cycle : list
        The reversed cycle.

    """
    rev_cycle = list(reversed(cycle))
    for n in range(1,len(cycle),2):
        rev_cycle[n] = (rev_cycle[n][0], -rev_cycle[n][1])
    return rev_cycle


def compress_cycle(cycle, tretkoff_graph):
    """
    Given a cycle, the Tretkoff graph, and the monodromy graph, return a
    shortened equivalent cycle.

    Parameters
    ----------
    cycle : list
        A cycle.
    tretkoff_graph : networkx graph
        The Tretkoff graph.

    Returns
    -------
    cycle : list
        The compressed cycle.
    """
    # Compression #1: add rotation numbers of successive cycle
    # elements if the branch points are equal
    N = len(cycle)
    n = 1
    while n < (N-2):
        curr_place = cycle[n]
        next_place = cycle[n+2]

        # if two successive branch points are the same then delete one of them
        # and sum the number of rotations.
        if curr_place[0] == next_place[0]:
            cycle[n] = (curr_place[0], curr_place[1] + next_place[1])
            cycle.pop(n+1)
            cycle.pop(n+1)
            N -= 2
        else:
            n += 2

    # Compression #2: delete cycle elements with zero rotations
    N = len(cycle)
    n = 0
    while n < (N-1):
        branch = cycle[n+1]

        if branch[1] == 0:
            cycle.pop(n)
            cycle.pop(n)
            N -= 2
        else:
            n += 2

    return cycle


def compute_ab_cycles(c_cycles, linear_combinations, g, tretkoff_graph):
    """
    Returns the a- and b-cycles of the Riemann surface given the
    intermediate 'c-cycles' and linear combinations matrix.

    Input:

    - c_cycles

    - linear_combinations: output of the Frobenius transform of the
    """
    lincomb = linear_combinations
    M,N = lincomb.shape

    a_cycles = []
    b_cycles = []

    for i in range(g):
        a = []
        b = []
        for j in range(N):
            cij = lincomb[i,j]
            c = c_cycles[j] if cij >= 0 else reverse_cycle(c_cycles[j])
            a.extend(abs(cij)*c[:-1])

            cij = lincomb[i+g,j]
            c = c_cycles[j] if cij >= 0 else reverse_cycle(c_cycles[j])
            b.extend(abs(cij)*c[:-1])

    a = a + [0]
    b = b + [0]
    a = compress_cycle(a, tretkoff_graph)
    b = compress_cycle(b, tretkoff_graph)

    a_cycles.append(a)
    b_cycles.append(b)
    return a_cycles, b_cycles


class Skeleton(object):
    """Defines the basic y-path structure of the Riemann surface.

    In particular, this class offers methods for determining which *y-paths*,
    given by a list of branch points in the complex x-plane and rotation
    numbers, to take order to define homology basis cycles as well as sheet
    switching paths.

    .. note::

        This class is a light wrapper around legacy code. This legacy code
        should eventually be made part of this class. What's implemented here
        is a temporary hack.

    Attributes
    ----------
    C : networkx.Graph
        A graph encoding the skeleton of the Riemann surface.
    genus : int
        The genus of the Riemann surface.


    Methods
    -------
    .. autosummary::

      a_cycles
      b_cycles
      c_cycles
      y_path_sheet_swap

    """

    def __init__(self, base_point, base_sheets, monodromy_group, genus):
        """Initializes the Y-Skeleton by computing the monodromy graph and
        homology cycles of the Riemann surface.

        Parameters
        ----------
        monodromy_group : dict
            The monodromy group of the curve as given by
            :py:func:`RiemannSurfacePathFactory.monodromy_group`
        genus : int
            The genus of the Riemann surface.

        """
        self.genus = numpy.int(genus)
        self.C = tretkoff_graph(base_point, base_sheets, monodromy_group)

        # compute the a-, b-, and c-cycles by calling self.homology()
        self._a_cycles, self._b_cycles, self._c_cycles, \
            self._linear_combinations = self.homology()

    def _value(self, node):
        """Gets the value associated with `node` on the y-skeleton `self.C`.

        """
        return self.C.nodes[node]['value']

    def _node(self, value):
        """Converts `value` to its associated node on the y-skeleton `self.C`.

        """
        nodes = [n for n,d in self.C.nodes(data=True)
                 if numpy.all(d['value'] == value) and not d['final']]
        return nodes[0]

    def _values(self, ypath, rotations=False):
        """Converts a ypath from value data to node data.

        See :py:meth:`self._value`. This method can return the rotation
        information, as opposed to the permutation, as an option.

        .. note::

            In order to provide rotation data the ypath must contain a
            starting and ending sheet. Also, it is assumed that the
            ypath is starting at / closer to the base place and ending
            further away.

        Parameters
        ----------
        ypath : list
            A list of nodes on the y-skeleton `self.C`.

        """
        values = [self._value(node) for node in ypath]
        if rotations:
            for i in range(1, len(values), 2):
                bi, pi = values[i]
                prev_sheet = values[i-1]
                next_sheet = values[i+1]

                # compute the number of rotations needed to move between
                # sheets.  take the shorest of the forward / reverse
                # path options.
                nrots = pi.index(next_sheet) - pi.index(prev_sheet)
                if nrots > len(pi)/2: nrots -= len(pi)
                values[i] = (bi, nrots)

        return values

    def _nodes(self, ypath):
        """Converts a ypath from node data to value data.

        """
        return [self._node(value) for value in ypath]

    def _trim_ypath(self, ypath):
        """Trims off the sheet data from `ypath`.

        Given a ypath in `(..., sheet, (branch_point, rotations), ...)`
        form return the same path but with only the branch points and
        rotations information.

        """
        return ypath[1::2]

    def base_node(self):
        """Returns the root node of the yskeleton."""
        return (0,)

    def ypath_from_base_to_sheet(self, sheet):
        """Returns a ypath from the base sheet of the Riemann surface to
        `sheet`.

        Parameters
        ----------
        sheet : int
            The index of the target sheet.
        """
        # convert sheet into a node
        if numpy.issubdtype(sheet, numpy.integer):
            sheet = self._node(sheet)

        base = self.base_node()
        path_to_sheet = nx.shortest_path(self.C, base, sheet)
        values = self._values(path_to_sheet, rotations=True)
        values = self._trim_ypath(values)
        return values

    def ypath_from_sheet_to_base(self, sheet):
        """Returns a ypath from `sheet` to the base sheet of the Riemann
        surface.

        .. note::

            This is simply a reversal of
            :py:meth:`ypath_from_base_to_sheet`.

        Parameters
        ----------
        sheet : int
            The index of the target sheet.

        """
        ypath = self.ypath_from_base_to_sheet(sheet)
        ypath_rev = self.ypath_values_reverse(ypath)
        return ypath_rev

    def ypath_values_reverse(self, ypath):
        """Returns a ypath representing the reverse of `ypath`.

        Reversing a ypath means not only visiting the branch points in
        reverse order but also rotating about them in reverse.

        .. note::

            Only accepts trimmed ypaths in `(bpt, nrots)` notation.
        """
        for n in range(len(ypath)):
            bpt, nrots = ypath[n]
            ypath[n] = (bpt, -nrots)
        return ypath

    def homology(self):
        """Computes the first homology group of the Riemann surface.

        Returns
        -------
        a_cycles, b_cycles, c_cycles, linear_combinations
            y-paths corresponding to the a-, b-, and c-cycles and a
            matrix giving the linear combination of c-cycles for each a-
            and b-cycle.

        .. note::

            Move compress cycle to :py:class:`RiemannSurfacePathFactory`?

        .. note::

            Delete legacy behavior of including sheet numbers in y-paths.

        """
        g = self.genus
        edges = final_edges(self.C)
        K = intersection_matrix(edges, g)
        T = frobenius_transform(K, g)

        c_cycles = compute_c_cycles(self.C, edges)
        a_cycles, b_cycles = compute_ab_cycles(c_cycles, T, g, self.C)

        # TODO: cycles are returned in sheet number / (branch point,
        # rotations) pairs. refactor so that they're just returned
        # without the sheet numbers
        for k in range(g):
            a_cycles[k] = a_cycles[k][1::2]
            b_cycles[k] = b_cycles[k][1::2]
        for k in range(len(c_cycles)):
            c_cycles[k] = c_cycles[k][1::2]

        linear_combinations = T[:2*g,:]
        return a_cycles, b_cycles, c_cycles, linear_combinations

    def a_cycles(self):
        """Returns the y-paths of the a-cycles of the Riemann surface.

        Returns
        -------
        list
            A list of the a-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        return self._a_cycles

    def b_cycles(self):
        """Returns the y-paths of the b-cycles of the Riemann surface.
        """
        return self._b_cycles

    def c_cycles(self):
        """Returns the y-paths of the c-cycles of the Riemann surface and the
        linear combination matrix defining the a- and b-cycles from the
        c-cycles.

        The a- and b- cycles of the Riemann surface are formed from linear
        combinations of the c-cycles. These linear combinations are obtained
        from the :py::meth:`linear_combinations` method.

        .. note::

            It may be computationally more efficient to integrate over
            the (necessary) c-cycles and take linear combinations of the
            results than to integrate over the a- and b-cycles
            separately. Sometimes the column rank of the linear
            combination matrix (that is, the number of c-cycles used to
            construct a- and b-cycles) is lower than the size of the
            homology group and sometimes the c-cycles are simpler and
            shorter than the homology cycles.

        """
        return self._c_cycles, self._linear_combinations

    def plot(self):
        """Plots the y-skeleton of the Riemann surface.
        """
        # get the edges and final edges of the graph.
        C = self.C
        final_nodes = [n for n,d in C.nodes(data=True) if d['final']]
        final_edges = [e for e in C.edges()
                       if e[0] in final_nodes or e[1] in final_nodes]
        edges = [e for e in C.edges() if e not in final_edges]

        # custom node position makes it clear which branch points lead
        # to which sheets. corresponding nodes are positioned in a line
        # above each "level" of the graph.
        pos = {(0,):(0,0)}
        labels = {(0,):"$0$"}
        level = 0
        level_points = [(0,)]  # a list of points at each level
        while len(level_points) > 0:
            level += 1
            level_points = sorted([n for n,d in C.nodes(data=True)
                                   if d['level'] == level])
            num_level_points = len(level_points)
            n = 0
            for point in level_points:
                pos[point] = (level,n-num_level_points/2.0)
                labels[point] = C.nodes[point]['label'] + \
                    '\n $' + str(point) + '$'
                n += 1

        # draw it with separate coloring for the final edges
        nx.draw_networkx_nodes(C, pos, node_color='w')
        nx.draw_networkx_nodes(C, pos, nodelist=final_nodes, node_color='w')
        nx.draw_networkx_edges(C, pos, edgelist=edges)
        nx.draw_networkx_edges(C, pos, edgelist=final_edges,
                               edge_color='b', style='dashed')
        nx.draw_networkx_labels(C, pos, labels=labels, font_size=13)
        plt.xticks([])
        plt.yticks([])
        return plt.gcf()
