"""
Monodromy

Module for computing the monodromy group of the set of branch points
of a complex plane algebraic curve.
"""

import numpy
import scipy
import sympy
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.cbook import flatten

from utilities import (
    cached_function,
    )

import pdb

def prim_fringe(G, weight_function=lambda e: 1, starting_vertex=None):
    """
    Computes a minimal spanning tree of the graph G. Variant of Prim's
    algorithm with optional starting vertex. Based on Sage's implementation.
    """
    if starting_vertex is None:
        v = G.nodes_iter().next()
    else:
        v = starting_vertex
    tree = set([v])
    edges = []

    # Initialize fringe_list with v's neighbors. Fringe_list
    # contains fringe_vertex: (vertex_in_tree, weight) for each
    # fringe vertex.
    fringe_list = dict([u, (weight_function((v, u)), v)] for u in G[v].keys())
    cmp_fun = lambda x: fringe_list[x]
    for i in range(G.order() - 1):
        # find the smallest-weight fringe vertex
        u = min(fringe_list, key=cmp_fun)
        edges.append((fringe_list[u][1], u))
        tree.add(u)
        fringe_list.pop(u)
        # update fringe list
        for neighbor in [v for v in G[u].keys() if v not in tree]:
            w = weight_function((u, neighbor))
            if neighbor not in fringe_list or fringe_list[neighbor][0] > w:
                fringe_list[neighbor] = (w, u)
    return edges

#@cached_function
def discriminant_points(f,x,y):
    """
    Returns the discriminant points of a plane algebraic curve `f =
    f(x,y)` in no particular order.

    Note: a specific order is established when `monodromy_graph()` is
    called.
    """
    p    = sympy.Poly(f,[x,y])
    res  = sympy.Poly(sympy.resultant(p,p.diff(y),y),x)

    # Compute the numerical roots. Since Sympy has issues
    # balancing between precision and speed we compute the roots
    # of each factor of the resultant.
    dps = sympy.mpmath.mp.dps
    rts = []
    for factor,degree in res.factor_list_include():
        rts.extend(factor.nroots(n=dps+3))
    rts = map(lambda z: z.as_real_imag(), rts)
    disc_pts = map(lambda z: sympy.mpmath.mpc(*z), rts)

    # Pop any roots that appear to be equal up to the set
    # multiprecision.  Geometrically, this may cause two roots to
    # be interpreted as one thus possibly reducing the genus of
    # the curve. XXX Think of better ways to deal with this
    # scenario.
    N = len(disc_pts)
    i = 0
    while i < N:
        k = 0
        while k < N:
            eq = sympy.mpmath.almosteq(disc_pts[i],disc_pts[k])
            if (k != i) and eq:
                disc_pts.remove(disc_pts[k])
                N -= 1
            else:
                k += 1
        i += 1
    return disc_pts

#@cached_function
def monodromy_graph(f, x, y, kappa=3.0/5.0):
    """
    Constructs a NetworkX graph describing the path connectedness of the
    monodromy paths. Each node has the following data attached to it:

    - pos: a tuple (b_re, b_im) containing the real and imaginary
           parts of the discriminant point.

    - value: the discriminant point (as a point in the complex plane)

    - radius: the radius of the monodromy path circle about the
              discriminant point

    - type: the 'type' of the vertex ('simple', 'node', 'vpoint',
            'vpoint node') using the definitions of [FSK]

    - root: the index of the root vertex. This is the vertex that is
            closest to the base point

    - base point: the base point of the monodromy group. Given to
                  each node for ease of access / convenience

    - string: a piece of data used internally to determine which
              vertices to conjugate by when determining the monodromy
              from the initial monodromy.

    - conjugates: a list of vertices / discriminant points to conjugate
                  by when constucting the monodromy path from the
                  base point to the discriminant point and back

    Input:

    - `f,x,y`: a complex plane algebraic curve

    - `kappa`: a scaling factor for the radii of the monodromy path
      circles. If `kappa==1` then the circles are as large as possible
      without intersecting any other monodromy path circles.

    Output:

    A NewtorkX graph containing the above data.

    Note:

    This graph is modified by `monodromy()`
    """
    disc_pts = discriminant_points(f,x,y)

    # sort the discriminant points by angle.
    # notation:
    #     - b0 = base point
    #     - bd = discriminant point closest to base point
    #     - bd_index = index of bd. its ranking in the sorted
    #                  discriminant points
    key = lambda z: 100*sympy.mpmath.re(z) - sympy.mpmath.im(z)
    def cmp(z,b0=None):
        """
        Compare by angle, first, and then by distance from the
        base point.
        """
        return (sympy.mpmath.arg(z-b0),sympy.mpmath.absmax(z-b0))

    bd  = min(disc_pts, key=key)
    base_radius = min( [sympy.mpmath.absmax(bd - bi) for bi in disc_pts
                        if bi != bd] + [10] ) * kappa / 2.0
    b0 = bd - base_radius
    disc_pts = sorted(disc_pts, key=lambda z: cmp(z,b0=b0))
    bd_index = disc_pts.index(bd)

    # XXX UGLY...but I can't find a better way to convert a polynomial
    # to something that outputs mpc type roots
    p = sympy.poly(f.subs(x,b0), y)
    coeffs = p.all_coeffs()
    coeffs_b0 = map(lambda c: c.evalf(subs={x:b0},
                                      n = sympy.mpmath.mp.dps), coeffs)
    coeffs_b0 = map(lambda z: sympy.mpmath.mp.mpc(*(z.as_real_imag())),
                    coeffs_b0)
    base_lift = sympy.mpmath.polyroots(coeffs_b0)

    #
    # XXX coerce everything to numpy data types
    #
    b0 = numpy.complex(b0)
    bd = numpy.complex(bd)
    disc_pts = numpy.array(disc_pts, dtype=numpy.complex)
    base_lift = numpy.array(base_lift, dtype=numpy.complex)

    # Compute minimal spanning tree. the weights are the distances
    # between the nodes. the spanning tree algorihtm is from a
    # Sage implementation of Prim with a designated starting node
    # since
    n = len(disc_pts)
    G = nx.complete_graph(n)
    weight_function = lambda e: \
        sympy.mpmath.fabs(disc_pts[e[1]]-disc_pts[e[0]])
    spanning_tree = prim_fringe(G, weight_function=weight_function,
                                starting_vertex=bd_index)
    G = nx.DiGraph()
    G.add_edges_from(spanning_tree)


    # Compute path radii for each discriminant point and store
    # radii and position data to the graph along with other vertex data.
    for i in range(len(disc_pts)):
        disc_pt = disc_pts[i]
        rho  = min( [abs(disc_pt - disc_pts[j])
                     for j in range(len(disc_pts)) if j != i] )
        radius = rho * kappa / 2.0

        # store useful data to graph
        G.node[i]['pos']        = (disc_pt.real,
                                   disc_pt.imag)
        G.node[i]['value']      = disc_pt
        G.node[i]['radius']     = radius
        G.node[i]['type']       = 'simple'
        G.node[i]['root']       = bd_index
        G.node[i]['basepoint']  = b0
        G.node[i]['baselift']   = base_lift
        G.node[i]['string']     = []
        G.node[i]['conjugates'] = []

    # Compute graph indices: these determine which sides of the
    # monodromy circles the edges of the graph are connected to.
    for (i,k) in G.edges():
        bi = G.node[i]['value']
        bk = G.node[k]['value']
        d = (bk-bi).real
        R = max( [G.node[i]['radius'], G.node[k]['radius']] )
        if d > R:    G[i][k]['index'] = (1,-1)
        elif d < -R: G[i][k]['index'] = (-1,1)
        else:        G[i][k]['index'] = (-1,-1)

    # Comptues the vertex types as defined in [FKS]:
    #
    # * 'node'     -- a parent vertex of the graph with multiple children
    # * 'v-point'  -- a vertex of special type described in Section 3 of
    #                 [FKS]. Travel to successor vertices don't necessarily
    #                 require going "underneath" a v-point
    # * 'endpoint' -- a childless vertex
    for vertex in G.nodes():
        path = nx.shortest_path(G, source=bd_index, target=vertex)
        for j in range(len(path)-2):
            e_left  = [path[j],path[j+1]]
            e_right = [path[j+1],path[j+2]]


            # The vertex in question is the one joining the two
            # edges of the path.
            v = e_left[1]

            # check for 'v-point': when a path contains a sequence of
            # edges of the form [..., bj^(I)], [bj^(I), ...] where I
            # is a matching index.
            if e_left[1] == e_right[0]:
                left_index  = G.edge[e_left[0]][e_left[1]]['index']
                right_index = G.edge[e_right[0]][e_right[1]]['index']

                # the middle vertex is a v-point. Mark it as such.
                # If it has previously been marked as a node or a
                # v-point node then mark it as a v-point node.
                if left_index[1] == right_index[0]:
                    if G.node[v]['type'] in ['node','v-point node']:
                        G.node[v]['type'] = 'v-point node'
                    else:
                        G.node[v]['type'] = 'v-point'

            # check for 'node': a discriminant point where several
            # branches meet. i.e. where the number of successors is
            # greater than 1. v-points can be nodes as well.
            n_succ = len(G.successors(v))
            if n_succ > 1:
                # if the vertex is already a v-point and if it has more
                # successors than the v-point and a string then
                # label it as a v-point node
                if G.node[v]['type'] in ['v-point', 'v-point node']:
                    G.node[v]['type'] = 'v-point node'
                else:
                    G.node[v]['type'] = 'node'

    # treat the root vertex as a node type
    if G.node[bd_index]['type'] == 'v-point':
        G.node[bd_index]['type'] = 'v-point node'
    else:
        G.node[bd_index]['type'] = 'node'

    #
    # Comptue the graph node conjugates so as to establisht he proper
    # ordering of the monodromy paths.
    #
    def angle(w, origin_vertex=None):
        """
        Compute the angle between vertex v and w using monodromy
        graph indices.
        """
        arg = numpy.angle #sympy.mpmath.arg
        v = origin_vertex

        # compute "reference point": if v is a node then this is
        # just -Rx. Otherwise, it's the return point...[XXX hard
        # to explain...]
        x = G.node[v]['value']
        Rv = G.node[v]['radius']
        if G.node[v]['type'] == 'node':
            z0 = -Rv
        else:
            u = G.predecessors(v)[0] # G is a tree so only one predecessor
            Ru = G.node[u]['radius']
            ind = G[u][v]['index']
            z0 = (G.node[u]['value'] + Ru*ind[0]) - (x + Rv*ind[1])


        # return the angle made with the "reference angle" by
        # rotating.  when v == w just return the smallest possible
        # value (-pi) since this is used as reference [XXX
        # again...hard to explain.
        if v == w:
            if G.node[v]['type'] == 'node':
                z = z0
            else:
                u   = G.predecessors(v)[0]
                ind = G[u][v]['index']
                z   = -Rv*ind[1]
        else:
            y  = G.node[w]['value']
            Rw = G.node[w]['radius']
            try:
                ind = G[v][w]['index']
            except:
                raise ValueError("Couldn't determine initial monodromy "+ \
                                 "point ordering: %d is not a "%(v)     + \
                                 "successor of %d."%(w))
            z = (y + Rw*ind[1]) - (x + Rv*ind[0])

        return arg(-z/z0)

    # (0) Gather special nodes
    root = G.node[0]['root']
    nodes = [n for n,data in G.nodes(data=True)
             if data['type']=='node']
    vpoints = [n for n,data in G.nodes(data=True)
               if data['type']=='v-point']
    endpoints = [n for n,deg in G.out_degree_iter() if deg == 0]
    vpoint_nodes = [n for n,data in G.nodes(data=True)
                    if data['type']=='v-point node']

    # caching...
    disc_pts = [data['value'] for node,data in G.nodes(data=True)]
    if len(G.node[root]['string']) == len(disc_pts):
        return G.node[root]['string']

    # (1) Initialize strings using endpoints
    #
    # construct initial strings: these are the strings following
    # the end points that stop at a node or vpoint
    for v in endpoints:
        string = [v]
        # as long as the successors are simple, keep appending to
        # the string. since graph is tree there should be only on
        # predecessor per node
        u = G.predecessors(v)[0]
        while G.node[u]['type'] == 'simple':
            v = u
            u = G.predecessors(v)[0]
            string.append(v)

        # add the string to the stack and attach data to graph
        G.node[v]['string'] = string

    # (2) create a stack of v-points, nodes, and v-point nodes
    #     and exhaust to construct ordering tree
    #
    # in each iteration, we check the following
    # 1. find a node / v-point that can be resolved. This is possible
    #    when all successors have a full string attached in the node
    #    data field 'string'. if none can be found then move to next
    #    special point.
    # 2. when a special vertex has enough filled strings,
    special_vertices = [n for n,data in G.nodes(data=True)
                        if data['type']!='simple']
    N = len(special_vertices)
    n = 0;
    while N > 0:
        # if n == N then we haven't found a node / v-point with
        # all strings available. raise error
        if n == N:
            raise Error("Couldn't determine initial monodromy " + \
                        "point ordering.")

        # grab the current special point
        v = special_vertices[n]

        # check if all attached strings are available
        has_strings = True
        for w in G.successors(v):
            if G.node[w]['string'] == []: has_strings = False

        # if all strings are available, apply ordering based on
        # special vertex type. otherwise, proceed to next special
        # point and loop
        if has_strings:
            # HACK: some unclean programming is done in the vpoint node
            # case. first, we need to resolve all of non-vee side
            # nodes and treat them as
            if G.node[v]['type'] == 'v-point node':
                G.node[v]['type'] = 'node'

                # XXX probably don't need the conditional. need to
                # make sure that v is included in the string since
                # if we need to conjugate by v then we need to
                # also conjugate
                if G.node[v]['string'] == []:
                    G.node[v]['string'] = [v]

                # get the "node side" successors of v: these are
                # the vertices that don't form a v-point.
                u = G.predecessors(v)[0]
                uv_ind = G[u][v]['index']
                succ = [w for w in G.successors(v)
                        if G[v][w]['index'][0] != uv_ind[1]]
                pts = succ + [v]

                # sort "node side points" (include v-point itself)
                key = lambda w: angle(w, origin_vertex=v)
                pts.sort(key=key)
                string = [G.node[pt]['string'] for pt in pts]
                G.node[v]['string'] = list(flatten(string))

                # now set v to a v-point type and get v-point side
                # point. Let rest of algorithm take care of
                # sorting
                G.node[v]['type'] = 'v-point'
                succ = [w for w in G.successors(v)
                        if G[v][w]['index'][0] == uv_ind[1]]
                pts = succ + [v]

            else:
                G.node[v]['string'] = [v]
                succ = G.successors(v)
                pts  = succ + [v]


            # sort and get strings
            key = lambda w: angle(w, origin_vertex=v)
            pts.sort(key=key)
            string = [G.node[pt]['string'] for pt in pts]

            # extend string to next v-point or node (unless at root)
            # XXX
            w = v
            if v != root:
                u = G.predecessors(w)[0]
                while (G.node[u]['type'] == 'simple'):
                    string.append(u)
                    w = u
                    u = G.predecessors(u)[0]
            G.node[w]['string'] = list(flatten(string))

            # reset to top of stack
            n = 0
            special_vertices.remove(v)
            N -= 1
        else:
            n += 1

    # finally, use the string data to compute the "position tree" and
    # then, from this data, compute the conjugates of each node in the
    # graph.
    N = len(disc_pts)
    position_tree = [v for v in G.node[root]['string']]
    while N > 0:
        # grab the largest element in the tree
        m = max(position_tree)
        i = position_tree.index(m)
        if i == (N-1):
            position_tree.remove(m)
            N -= 1
        else:
            # k is the element of the tree appearing to the right
            # of the element m. conjugate phi_m by phi_k
            k = position_tree[i+1]

            # swap m and k in the position tree
            position_tree[i+1] = m
            position_tree[i] = k

            # store conjugation information in self
            G.node[m]['conjugates'].append(k)

    return G

def show_paths(G):
    """
    Plot all of the paths in the complex x-plane of the monodromy
    group.
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    # plot the location of the base_point
    a = G.node[0]['basepoint']
    ax.plot(sympy.re(a), sympy.im(a), color='r',
            marker='o', markersize=10)

    for i in G.nodes():
        # plot the circle about the discriminant point
        radius = G.node[i]['radius']
        pos    = G.node[i]['pos']
        circle = Circle(pos, radius, edgecolor='b', facecolor='w')
        ax.add_patch(circle)

        # mark the node number
        ax.text(pos[0], pos[1], '%d'%i,
                horizontalalignment='center', verticalalignment='center')

        # for each connected discriminant point, draw the line
        # connecting the two circles. (using the path index.)
        for k in G.successors(i):
            index       = G[i][k]['index']
            next_pos    = G.node[k]['pos']
            next_radius = G.node[k]['radius']
            x    = (pos[0] + index[0]*radius,
                    next_pos[0] + index[1]*next_radius)
            y    = (pos[1], next_pos[1])
            line = Line2D(x,y)
            ax.add_line(line)

    ax.axis('tight', aspect=1.0)
    fig.show()


class XSkeleton(object):
    """Defines the basic x-path structure of the Riemann surface.

    A component of :class:`RiemannSurfacePathFactory`, `XSkeleton`
    computes the 'x-highways` of the Riemann surface: the paths taken in
    the x-plane to navigate the branch points of the curve. A
    `networkx.Graph` object encapsulates most of this information.

    Attributes
    ----------
    RS : RiemannSurface
    kappa : double
        A scaling factor between 0.5 and 1 used to determine the radii
        of the circles around the branch points of the curve.
    G : networkx.Graph
        A graph object defining the skeleton of the x-part of the
        Riemann Surface.

    Methods
    -------
    base_point
        Returns the base point of the monodromy group of the curve.
    branch_points
        Returns the branch points of the curve.
    xpath_to_branch_point
        Returns a list of tuples representing a path in the complex
        x-plane starting at the base point and leading to the given
        branch point.
    xpath_circle_branch_point
        Returns a list of tuples representing a path in the complex
        x-plane encircling a branch point a given number of times.
    xpath_monodromy_path
        Returns a list of tuples representing a path in the complex
        x-plane representing the monodromy path of the given branch
        point.
    xpath_monodromy_path
        Returns a list of tuples representing a path in the complex
        x-plane starting at the base point and encircling infinity a
        given number of times.

    .. todo::

        Wrap remaining functions (as necessary) in this module into this
        class.

    """
    def __init__(self, RS, kappa=3./5., base_point=None):
        """Initialize the XSkeleton.

        All of the information given by :class:`XSkeleton` is computed
        at initialization. The primary attribute is a `networkx.Graph`
        object which stores the radii and branch point connectivity.

        Arguments
        ---------
        RS : RiemannSurface
        kappa : double, optional
            A scaling factor between 0.5 and 1 used to determine the
            radii of the circles around the branch points of the curve.
        base_point : complex (optional)
            A custom base point can be given. This is often used for
            testing and comparison purposes.

        """
        self.RS = RS

        f = RS.f
        x = RS.x
        y = RS.y
        self.kappa = kappa
        self.hurwitz_system = None

        # ensure that the monodromy graph is ready at instantiation
        self.G = monodromy_graph(f, x, y, kappa=kappa)

        # set the base points and base sheets
        self._custom_base_point = True
        self._base_point = base_point
        if not base_point:
            self._base_point = self.G.node[0]['basepoint']
            self._custom_base_point = False

    def _index(self, b):
        """Returns the index of the branch point in the ordered list of branch
        points.

        In some cases the ordering of the branch points in the complex
        plane matters (where the ordering is the angle made between the
        branch point and the base point).

        Arguments
        ---------
        bpt : complex

        .. note::

            Should some approximation proceedures be in place here?

        """
        # pts = self.discriminant_points()
        # return numpy.argmin(numpy.abs(pts-b))
        return self.branch_points().tolist().index(b)

    def base_point(self):
        return self._base_point

    def branch_points(self):
        """Returns a list of the discriminant points of the X-skeleton."""
        return numpy.array(
            [data['value'] for node,data in self.G.nodes(data=True)],
            dtype=numpy.complex)

    def root_branch_point(self):
        """Returns the branch point closest to / attached to the base point.
        """
        root_index = self.G.node[0]['root']
        return self.G.node[root_index]['value']

    def branch_points_on_path(self, bi):
        """Returns the branch points lying on the path from the base point,
        through the root node, and to the branch point.

        Arguments
        ---------
        bi : complex
            A branch point of the curve.

        Returns
        -------
        list
            A list of the branch points lying on the x-path from the
            base point to the target branch point.

        """
        bi_index = self._index(bi)
        root_index = self.G.node[0]['root']
        path_vertices = nx.shortest_path(self.G, source=root_index,
                                         target=bi_index)

        return [self.G.node[n]['value'] for n in path_vertices]

    def conjugates(self, bi):
        """Returns the branch points that are "conjugate" to the given branch
        point.

        The conjugate branch points are those where the path around `bi`
        needs to lie above the point when defining the monodromy path.

        Arguments
        ---------
        bi : complex
            A branch point of the curve.

        Returns
        -------
        list
            A list of the conjugate branch points to `b`.

        """
        bi_index = self._index(bi)
        return self.G.node[bi_index]['conjugates']

    def radius(self, bi):
        """The radius of the circle around the branch point `bi`.

        Arguments
        ---------
        bi : complex
            A branch point of the curve.

        Returns
        -------
        double
            The radius of the circle around `b`.

        """
        bi_index = self._index(bi)
        return self.G.node[bi_index]['radius']

    def _get_edge_index(self, bi, bj):
        """Returns the "edge index" of the path from `bi` to `bj`.

        The edge index encodes which sides of the branch points' circles
        are connected. `-1` = left side. `+1` = right side. So, for
        example, an edge index of `(1,-1)` between points `bi` and `bj`
        means a the straight line path connects the right side (`+1`) of
        the circle around `bi` to the left side (`-1`) of the circle
        around `bj`.

        This method is used primarily in :py:meth:`is_semicircle_on_path`
        and :py:meth:`semicircle_on_path`.

        """
        bi_index = self._index(bi)
        bj_index = self._index(bj)
        root_index = self._index(self.root_branch_point())
        edgeij_index = self.G[bi_index][bj_index]['index']

        # the root point always connects to the left side of a connected
        # branch point. the side on the root point circle doesn't
        # matter.
        if bi_index == root_index:
            #return (0,-1)
            return edgeij_index
        else:
            return edgeij_index

    def xpath_to_branch_point(self, bi):
        """Returns data defining the path starting from the base x-point
        and ending "at" the branch point `bi`. (See below.)

        Technically, the ending point is :math:`b_i \pm R_i`. Whether or
        not we end on the left or right side of the branch point depends
        on its connectivity with the other branch points of the
        X-Skeleton structure.

        Arguments
        ---------
        bi : complex
            A finite branch point.

        Returns
        -------
        list
            A list of tuples defining either lines or semicircles in the
            complex x-plane.

        """
        # xpath stores the tuples defining the Riemann surface paths
        b = self.branch_points_on_path(bi)
        conjugates = self.conjugates(bi)
        xpath = []

        # if there is a custom base point, add the line from the
        # base point to the root branch point
        if self._custom_base_point:
            root = self.root_branch_point()
            R = self.radius(root)
            xpath.append( (self.base_point(), root-R) )

        # for each branch point on the path to the target branch point
        # point construct the semicircle going aroung the branch point
        # (if necessary) and the line going to the next branch point.
        bj = b[0]
        for bk in b[1:]:
            # the semicircle around `bj` should go above `bj` if it is a
            # conjugate branch point to `bi`
            is_conjugate = True if bj in conjugates else False
            if self.has_semicircle_on_path(bj, bk):
                xpath.append(self.semicircle_on_path(
                    bj, bk, is_conjugate=is_conjugate))

            xpath.append(self.line_on_path(bj, bk))
            bj = bk  # update loop

        return xpath

    def xpath_circle_branch_point(self, bi, nrots=1):
        """Returns a list of tuples, each encoding semicircles, representing a
        rotation around branch point `bi` `nrots` times.

        Arguments
        ---------
        bi : complex
            The branch point to encircle.
        nrots : int, optional
            The number of times to rotate around the branch point. If
            negative, rotate `abs(nrots)` times in the negative
            (clockwise) direction. (default = 1)

        Returns
        -------
        list
            A list of tuples of the form `(R, w, arg_start, arg_end,
            dir)` where `R` is the radius, `w` is the center,
            `arg_start` is the starting argument, `arg_end` is the
            ending argument, and `dir` is the direction of a semicircle
            making up a number of rotations around the branch point
            `bi`.

        .. note:

            The output, here, can be used immediately to construct
            :class:RiemannSurfacePathPrimitive objects as opposed to the
            output of :py:meth:`semicircle_on_path` where the direction
            of the semicircle still needs to be specified.

        """
        b = self.branch_points_on_path(bi)

        # if we're encircling the root branch point (the point closest
        # to the base point) then we're always starting on the left-side
        # of the circle
        if len(b) == 1:
            edge_index = (0,-1)
        else:
            b_prev = b[-2]
            edge_index = self._get_edge_index(b_prev, bi)

        # construct complete circle
        R = self.radius(bi)
        w = bi
        theta = numpy.pi if edge_index[1] == -1 else 0
        dtheta = numpy.pi if nrots > 0 else -numpy.pi
        circle = [(R, w, theta, dtheta),
                  (R, w, theta + dtheta, dtheta)]

        # rotate abs(nrots) times
        return circle * int(abs(nrots))

    def xpath_monodromy_path(self, bi, nrots=1):
        """Returns data defining the path starting from the base x-point, going
        around the branch point `bi` `nrots` times, and returning to the
        base x-point.

        Arguments
        ---------
        bi : complex
            A branch point.
        nrots : integer (default `1`)
            A number of rotations around this branch point.

        Returns
        -------
        list
            A list of tuples defining either lines or semicircles in the
            complex x-plane.

        """
        # special case when going around infinity.
        if bi == sympy.oo:
            return self.xpath_around_infinity(nrots=nrots)

        xpath_to_bi = self.xpath_to_branch_point(bi)
        xpath_around_bi = self.xpath_circle_branch_point(bi, nrots=nrots)
        xpath_from_bi = self.xpath_reverse(xpath_to_bi)
        xpath = xpath_to_bi + xpath_around_bi + xpath_from_bi
        return xpath

    def xpath_around_infinity(self, nrots=1):
        """Returns data defining the path starting from the base x-point, going
        around infinity `nrots` times, and returning to the base point.

        Computes and returns a path circling infinity in the positive
        direction. (In the finite :math:`\mathbb{C}_x` plane this is a
        clockwise circle around all of the finite branch point.) This
        path is used to determine the monodromy group element around
        infinity.

        Arguments
        ---------
        nrots : integer, (default `1`)
            The number of rotations around this branch point.

        Returns
        -------
        RiemannSurfacePath
            The path of the monodromy group circling the branch point.

        """
        xpath = []

        # determine the radius R of the circle, centered at the origin,
        # encircling all of the branch points and their "protection"
        # circles
        b = self.branch_points()
        R = 0.0
        for bi in b:
            radius = self.radius(bi)
            arg = numpy.angle(bi)
            Ri = numpy.abs(bi + arg*radius)
            R = Ri if Ri > R else R

        # the path begins with a line starting the base point and ending
        # at -R.
        xpath.append((self.base_point(), -R))

        # the positive direction around infinity is equal to the
        # negative direction around the origin
        dtheta = -numpy.pi if nrots > 0 else numpy.pi
        for _ in range(abs(nrots)):
            xpath.append((R, 0, numpy.pi, dtheta))
            xpath.append((R, 0, 0, dtheta))

        # return to the base point
        xpath.append((-R, self.base_point()))
        return xpath

    def xpath_reverse(self, xpath):
        """Reverses an x-path.

        Useful for building the return path from a branch point to the
        base point.

        Arguments
        ---------
        xpath : list
            A list of tuples defining either lines or semicircles in the
            complex x-plane.

        Returns
        -------
        list
            A list of tuples defining either lines or semicircles in the
            complex x-plane representing the reverse of `xpath`.

        """
        reverse_xpath = []
        for data in xpath[::-1]:
            if len(data) == 2:
                z0, z1 = data
                data = (z1, z0)
            else:
                R, w, theta, dtheta = data
                theta += dtheta
                dtheta = -dtheta
                data = (R, w, theta, dtheta)

            reverse_xpath.append(data)
        return reverse_xpath

    def has_semicircle_on_path(self, bj, bk):
        """Returns `True` if the path from branch point `bj` to `bk` requires
        encircling `bj`. Returns `False` if no semicircle is necessary.

        Arguments
        ---------
        bj, bk : complex
            Adjacent discriminant points lying on the x-skeleton path.

        Returns
        -------
        boolean
            `True` if the path from `bj` to `bk` requires a semicircle
            around `bk`.

        """
        edgejk_index = self._get_edge_index(bj, bk)

        # special case when bj is the root branch point:
        root = self.root_branch_point()
        if numpy.abs(bj-root) < 1e-14:
            # if the line from the root branch point `bj` to the child
            # `bk` connects to the left side of the root branch point
            # then a semicircle around `bj` is unnecessary
            if edgejk_index[0] == -1:
                return False
            else:
                return True
        else:
            # otherwise, determine the branch point `bi` preceeding `bj`
            # on the path to `bk`
            b = self.branch_points_on_path(bk)
            bj_path_index = b.index(bj)
            bi = b[bj_path_index - 1]
            edgeij_index = self._get_edge_index(bi, bj)

        # a semicircle is unnecessary when the side of the line segment
        # connecting point `bi` to point `bj` on the `bj` circle is the
        # same as of the segment connecting `bj` to `bk`. visually, this
        # pattern looks like a "V" touching the side of a circle.
        if edgeij_index[1] == edgejk_index[0]:
            return False
        else:
            return True

    def semicircle_on_path(self, bj, bk, is_conjugate=False):
        """Returns a tuple defining the semicircle that needs to be navigated
        when traveling around branch point `bj`, to `bk`.

        Raises an error if no such semicircle exists. See
        :py:meth:`has_semicircle_on_path`

        Arguments
        ---------
        bj, bk : complex
            Adjacent discriminant points lying on the x-skeleton path.
        is_conjugate : boolean (default `False`)
            If `False`, returns path data defining a semicircle going
            *below* `bj`. If `True`, the semicircle travels above `bj`.

        Returns
        -------
        (R, w, arg_start, arg_end)
            A tuple encoding the semicircle information. `R` is the
            radius, `w` is the center, `arg_start` is the starting
            argument on the semicircle, and `arg_end` is the ending
            argument.

        """
        if not self.has_semicircle_on_path(bj, bk):
            raise ValueError('No semicircle is necessary when navigating '
                             'from %s to %s'%(bj, bk))

        edgejk_index = self._get_edge_index(bj, bk)
        bj_index = self._index(bj)
        R = self.G.node[bj_index]['radius']
        w = bj

        # special case when bj is the root branch point:
        root = self.root_branch_point()
        if numpy.abs(bj-root) < 1e-14:
            # we've already checked that the semicircle is needed. the
            # semicircle around the root branch point always starts on
            # the left side
            theta = numpy.pi
        else:
            # otherwise, determine the branch point `bi` preceeding `bj`
            # on the path to `bk` and get its connection properties
            b = self.branch_points_on_path(bk)
            bj_path_index = b.index(bj)
            bi = b[bj_path_index - 1]
            edgeij_index = self._get_edge_index(bi, bj)

            # start on the left side of the `bj` circle if the edge
            # connecting `bi` to `bj` meets the left (-1) side
            theta = numpy.pi if edgeij_index[1] == -1 else 0

        dtheta = -numpy.pi if is_conjugate else numpy.pi
        return (R, w, theta, dtheta)

    def line_on_path(self, bj, bk):
        """Returns a tuple representing the endpoints of the line from
        branch points `bj` to `bk`.

        Arguments
        ---------
        bj, bk : complex
            Adjacent branch points.

        Returns
        -------
        (z0, z1)
            A tuple representing the line segment with the complex
            starting point `z0` and complex ending point `z1`.

        """
        bj_index = self._index(bj)
        bk_index = self._index(bk)
        bj_radius = self.radius(bj)
        bk_radius = self.radius(bk)
        edgejk_index = self._get_edge_index(bj, bk)

        # if we're on the left (-1) side of the circle then we subtract
        # the radius. otherwise, we're on the right (+1) side so we add
        # the radius.
        z0 = bj + edgejk_index[0]*bj_radius
        z1 = bk + edgejk_index[1]*bk_radius
        return (z0, z1)

    def show_paths(self):
        show_paths(self.G)
