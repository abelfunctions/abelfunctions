"""
Monodromy
"""

import numpy
import scipy
import sympy
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.cbook import flatten

from utilities import cached_function, cached_property

from riemannsurface_path import polyroots

import pdb

class Permutation(object):
    """
    A permutation.

    Examples::
    We create the permutation ``p = 1->1, 2->4, 3->2, 4->3``.
    
        >>> p = Permutation([0,3,1,2])

    We can multiply permutations together. Let ``q`` be the transposition
    ``0->1, 1->0``. The product ``qp`` represents the permutation obtained
    by applying ``p`` to the identity ``[0,1,2,3]`` first and then ``q``.

        >>> q = Permutation([2,1])
        >>> q*p
        [1, 3, 0, 2]
        >>> p*q
        [3, 0, 1, 2]

    Permutations can act on lists.
    
        >>> p.action(['a','b','c','d'])
        ['a', 'd', 'b', 'c']
    """
    def __init__(self,l):
        """
        Construct a Permutation object. If the input "l" is a list of
        ints then set self.list to "l". Otherwise, cycle notation is assumed 
        if l[0] is itself a list.
        """
        if isinstance(l,list):
            if isinstance(l[0],list):
                l = self._list_from_cycle(l)
            self._list = l                
        else:
            # try to turn object into list
            self._list = list(l)
            self.__init__(l)

        self._hash = None


    def _list_from_cycle(self,cycles):
        """
        Create a permutation list ``i \to l[i]`` from a cycle notation
        list.

        Example:

        >>> p = Permutation([[0,1],[2],[3]])
        >>> p._list
        [1, 0, 2]

        >>> q = Permutation([[2,4],[1,3]])
        >>> q._list
        [2, 3, 0, 1]
        """
        degree = max([0] + [max(cycle + [0]) for cycle in cycles]) + 1
        l = range(degree)
        for cycle in cycles:
            if not cycle:
                continue
            first = cycle[0]
            for i in range(len(cycle)-1):
                l[cycle[i]] = cycle[i+1]
            l[cycle[-1]] = first

        return l

    def _cycle_from_list(self,l):
        pass

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return self._list.__repr__()

    def __hash__(self):
        if self._hash is None:
            self._hash = str(self._list).__hash__()
        return self._hash

    def __len__(self):
        return self._list.__len__()

    def __getitem__(self, key):
        return self._list.__getitem__(key)
    
    def __contains__(self, item):
        return self._list.__contains__(item)

    def __mul__(self, other):
        return self.__rmul__(other)

    def __rmul__(self, other):
        # pad the permutations if they are of different lengths
        new_other = other[:] + [i+1 for i in range(len(other), len(self))]
        new_p1 = self[:] + [i+1 for i in range(len(self), len(other))]
        return Permutation([new_p1[i-1] for i in new_other])

    def __call__(self, i):
        """
        Returns the image of the integer i under this permutation.
        """
        if isinstance(i,int) and 0 <= i < len(self):
            return self[i]
        else:
            raise TypeError, "i (= %s) must be an integer between %s and %s" %(i,0,len(self)-1)


    def index(self, key):
        return self._list.index(key)

    def action(self, a):
        """
        Returns the action of the permutation on a list.

        **Examples**

            >>> p = Permutation([0,3,1,2])
            >>> p.action(['a','b','c','d'])
            ['a', 'd', 'b', 'c']
        """
        if len(a) != len(self):
            raise ValueError, "len(a) must equal len(self)"
#        return map(lambda i: a[self[i]-1], range(len(a)))
        return map(lambda i: a[self[i]], range(len(a)))

    def inv(self):
        """
        Returns the inverse permutation.
        """
        l = range(len(self))
        for i in range(len(self)):
            l[self(i)] = i

        return Permutation(l)
        


def matching_permutation(a, b):
    """
    Returns the permutation ``p`` that matches the elements of the list 
    `a` to those of `b` as closely as possible. That is 
    `b ~= p.action(a)``. The elements of the two lists nedd not be the
    same but will try to match them as closely as possible.
    
    EXAMPLES::
    If the two lists contain the same elements then ``matching_permutation``
    simply returns permutation defining the rearrangement.::
        
        >>> a = [6,-5,9]
        >>> b = [9,6,-5]
        >>> p = matching_permutation(a,b); p
        [3, 1, 2]
        
    ``matching_permutation`` will attempt to find such a permutation even if 
    the elements of the two lists are not exactly the same.::
        
        >>> a = [1.1,7.2,-3.9]
        >>> b = [-4,1,7]
        >>> p = matching_permutation(a,b); p
        [3, 1, 2]
        >>> p.action(a)
        [-4, 1, 7]
    """
    N = len(a)
    if N != len(b):
        raise ValueError, "Lists must be of same length."

    perm = [-1]*N
    eps  = 0.5*min([abs(a[i]-a[j]) for i in range(N) for j in range(i)])
    
    for i in xrange(N):
        for j in xrange(N):
            dist = abs(a[i] - b[j])
            if dist < eps:
                perm[j] = i
                break

    if -1 in perm:
        raise ValueError, "Could not compute matching permutation between %s and %s." %(a,b)
            
    return Permutation(perm)




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


@cached_function
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


@cached_function
def monodromy_graph(f,x,y,kappa=3.0/5.0):
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
    cmp = lambda z: sympy.mpmath.arg(z-b0)
    bd  = min(disc_pts, key=key)
    base_radius = min( [sympy.mpmath.absmax(bd - bi) for bi in disc_pts 
                        if bi != bd] + [10] ) * kappa / 2.0

    b0 = bd - base_radius
    disc_pts = sorted(disc_pts, key=cmp)
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
        G.node[i]['pos']        = (sympy.mpmath.re(disc_pt), 
                                   sympy.mpmath.im(disc_pt))
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
        d = sympy.mpmath.re(bk-bi)
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


def initial_monodromy_path(G,i,Npts=4):
    """
    Returns a list of interpolating points on the initial monodromy
    path encircling the 'i'th discriminant point. 'Npts' is the number
    of interpolating points per "section" of the path where the path
    is divided into semi-circle and line segment sections.
    """
    # Functions for computing interpolating points on semi-circles
    # and line segments.
    t_pts = sympy.mpmath.linspace(0, 1, Npts, endpoint=False)
    circle = lambda R, z0, arg: \
        [R*sympy.mpmath.exp(sympy.mpmath.j*(sympy.mpmath.pi*t + arg)) + z0
         for t in t_pts]
    line = lambda start, end: [start*(1-t) + end*t for t in t_pts]

    root = G.node[i]['root']
    path_points = []

    # Compute interpolating points for circle / line pairs. That
    # is, for each node in the shortest path to the target node
    # determine the interpolating points on the circle and the
    # interpolating points on the line connecting to the next
    # discriminant point.
    path_vertices = nx.shortest_path(G, source=root, target=i)
    prev_node = root
    for idx in range(len(path_vertices)-1):
        curr_node   = path_vertices[idx]
        curr_radius = G.node[curr_node]['radius']
        curr_value  = G.node[curr_node]['value']

        next_node   = path_vertices[idx+1]
        next_radius = G.node[next_node]['radius']
        next_value  = G.node[next_node]['value']

        # Determine if semi-circle is needed. This is done by
        # checking the path index of the previous edge with the
        # path index of the next edge
        if prev_node == root: 
            prev_edge_index = (0,-1)  # in the root node case
        else: 
            prev_edge_index = G[prev_node][curr_node]['index']
        curr_edge_index = G[curr_node][next_node]['index']
        if prev_edge_index[1] != curr_edge_index[0]:
            # add semicircle going in positive direction
            arg = sympy.mpmath.pi if prev_edge_index[1] == -1 else 0
            circ_pts = circle(curr_radius, curr_value, arg)
            path_points.extend(circ_pts)

        # draw line to next discriminant point
        start = curr_value +  curr_edge_index[0]*curr_radius
        end   = next_value +  curr_edge_index[1]*next_radius
        line_pts = line(start, end)
        path_points.extend(line_pts)

        # update previous point
        prev_node = curr_node

    # we have now arrived at the circle encircling the desired
    # discriminant point. There's a special case for when we just
    # encircle the root node.
    if len(path_vertices) == 1:
        next_value  = G.node[root]['value']
        next_radius = G.node[root]['radius']
        curr_edge_index = (-1,-1)

    arg = sympy.mpmath.pi if curr_edge_index[1] == -1 else 0
    circ_pts = circle(next_radius, next_value, arg) + \
        circle(next_radius, next_value, arg + sympy.mpmath.pi)
    circ_pts.append(circ_pts[0])

    # combine the path to the circle, the final circle points, 
    # and the reverse path points
    path_points += circ_pts + path_points[::-1]

    return path_points


def plot_initial_monodromy_path(f,x,y,i,kappa=3.0/5.0,Npts=4,eps=0.1):
    """
    Plots the initial monodromy path. Used for testing and
    debugging purposes.
    """
    # plot the monodromy graph
    G = monodromy_graph(f,x,y,kappa=kappa)
    show_paths(G)
    fig = plt.gcf()
    ax  = fig.gca()

    # compute the path points
    path_points = initial_monodromy_path(f,x,y,G,i,Npts=Npts)
    N = len(path_points)

    # plot forward path
    topos = lambda pp: (sympy.re(pp), sympy.im(pp))
    for n in xrange(N/2):
        pos = topos(path_points[n])
        ax.text(pos[0], pos[1], '%d'%n, size='x-small',
                verticalalignment='top')

    # plot return path
    for n in xrange(N/2,N):
        pos = topos(path_points[n])
        ax.text(pos[0], pos[1], '%d'%n, size='x-small',
                verticalalignment='bottom')

    ax.axis('tight')
    fig.show()


def initial_monodromy(f,x,y,G,i,kappa=3.0/5.0,Npts=8,lift_paths=False):
    """
    Returns the initial monodromy corresponding to disciminant
    point `i`.  That is, the permutation of sheets as we go around
    discriminant point `i` using the initial monodromy paths.
    """
    # compute mpmath-optimized versions of f and its derivatives
    dfdx = sympy.diff(f,x)
    dfdy = sympy.diff(f,y)

    _f   = sympy.lambdify((x,y), f, "mpmath")
    dfdx = sympy.lambdify((x,y), dfdx, "mpmath")
    dfdy = sympy.lambdify((x,y), dfdy, "mpmath")
    
    deg = sympy.degree(f,y)

    # obtain interpolating points in path
    path = initial_monodromy_path(G,i,Npts=Npts)
    N = len(path)
    maxiter = 32
    eps = sympy.mpmath.mp.eps

    # note that the order of base_lift is fixed at creation of the
    # class / monodromy group.
    base_point = G.node[0]['basepoint']
    base_lift = G.node[0]['baselift']
    prev_rts = [yi for yi in base_lift]

    if lift_paths:
        lift_points = [base_lift]

    # analytically continue along the path starting with the base lift
    # at the base point.
    yi   = [0]*deg
    yim1 = base_lift
    xim1 = base_point
    for i in xrange(1,N):
        xi = path[i]
        dx = xi-xim1

        with sympy.mpmath.extraprec(4):
            for j in xrange(deg):
                # compute numerical approximation of the next set of
                # roots using Taylor series. This allows us to use
                # fewer interpolating points. Use generators for fast
                # creation.
                yim1j = yim1[j]
                dyj = - dx * dfdx(xim1,yim1j) / dfdy(xim1,yim1j)
                yij_approx = yim1j + dyj

                # Newton iterate to next point. (Note: this is done
                # instead of "sympy.mpmath.polyroots" for speed and
                # instead of sympy.mpmath.findroot for performance.)
                yij = yij_approx
                for k in xrange(maxiter):
                    step = _f(xi,yij) / dfdy(xi,yij)
                    yij -= step

                yi[j] = yij

        if lift_paths: 
            lift_points.append([yij for yij in yi])

        xim1 = xi
        yim1 = yi

    # return the lift points, if requested. Otherwise, just return
    # the initial monodromy permutation.
    if lift_paths: return lift_points
    else:          return matching_permutation(base_lift, yi)


def plot_initial_monodromy_lift(f,x,y,i,kappa=3.0/5.0,Npts=8):
    """
    Plots the lift [y1,...,yn] on the complex plane as x varies 
    Along an initial monodromy path.
    """
    G = monodromy_graph(f,x,y,kappa=kappa)
    lift = initial_monodromy(f,x,y,G,i,Npts=Npts,lift_paths=True)
    deg = len(lift[0])
    clrs = ['b','g','r','k','c','m','y']*int(deg/7+1)

    plt.ion()

    # scan ahead in order to determine proper plot bounds
    axis = [-1,1,-1,1]
    for j in xrange(len(lift)):
        for i in xrange(deg):
            yji = lift[j][i]

            if yji.real < axis[0]: axis[0] = float(yji.real)
            if yji.real > axis[1]: axis[1] = float(yji.real)
            if yji.imag < axis[2]: axis[2] = float(yji.imag)
            if yji.imag > axis[3]: axis[3] = float(yji.imag)

    xeps = (axis[1]-axis[0])/10.0
    yeps = (axis[3]-axis[2])/10.0
    axis[0] -= xeps
    axis[1] += xeps
    axis[2] -= yeps
    axis[3] += xeps

    plt.axis(axis)

    # initialize plot with base lift
    xdata = {}
    ydata = {}
    lines = range(deg)
    for i in xrange(deg):
        y = lift[0][i]
        xdata[i] = [y.real]
        ydata[i] = [y.imag]
        lines[i], = plt.plot(y.real, y.imag, color=clrs[i],
                             linestyle='--', linewidth=4*(i+1), alpha=0.3)

    # for each complex x-point plot the ordered lift at that point
    for j in xrange(len(lift)):
        for i in xrange(deg):
            y = lift[j][i]

            xdata[i].append(y.real)
            ydata[i].append(y.imag)

            lines[i].set_xdata(xdata[i])
            lines[i].set_ydata(ydata[i])
        plt.draw()


def _construct_position_tree(G):
    """
    Returns the "relative position tree", a list giving the relative
    ordering of the branch points, using the monodromy graph data.

    See Section 3 of [FCS].
    """
    def angle(w, origin_vertex=None):
        """
        Compute the angle between vertex v and w using monodromy
        graph indices.
        """
        arg = sympy.mpmath.arg
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

    return [v for v in G.node[root]['string']]


@cached_function
def monodromy(f,x,y,kappa=3.0/5.0,Npts=8):
    """
    Returns information about the monodromy group of the Riemann
    surface associated with the plane complex algebraic curve ``f(x,y)
    = 0``. 

    In particular, `monodromy()` returns the Hurwitz system of
    ``f(x,y) = 0``, which consistes of the data
    
    - base point: the base point of the monodromy group

    - base sheets: the ordered sheets / lift ``y = \{
      y_0,\ldots,y_{n-1}`` lying above the base point, where ``n =
      deg_y(f)``.

    - branch points: the branch points of the algebraic curve lying in
      the complex ``x``-plane

    - monodromy: the elements of the monodromy group. The ``j``th
      element of this list corresponds to the ``j``th branch point

    - connectivity graph: a network_x graph that encodes all of the
      path-finding information of the monodromy group. See
      `monodromy_graph()` for more information.

    Input:

    - `f,x,y`: a complex plane algebraic curve

    - `kappa`: (default: 3.0/5.0) a scaling factor for the radii of
      the monodromy path circles. If `kappa==1` then the circles are
      as large as possible without intersecting any other monodromy
      path circles.

    - `Npts`: (default: 8) the number of interpolating points per path
      segment to use when computing the initial monodromy
    """
    G = monodromy_graph(f,x,y)
    disc_pts = [data['value'] for node,data in G.nodes(data=True)]
    N = len(disc_pts)

    monodromy = [initial_monodromy(f,x,y,G,i,Npts=Npts) for i in range(N)]
    position_tree = _construct_position_tree(G)

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
            phi_m = monodromy[m]
            phi_k = monodromy[k]
            monodromy[m] = phi_k * phi_m * phi_k.inv()

            # swap m and k in the position tree
            position_tree[i+1] = m
            position_tree[i] = k

            # store conjugation information in self
            G.node[m]['conjugates'].append(k)

    # return the Hurwitz system and monodromy graph
    base_point = G.node[0]['basepoint']
    base_sheets = G.node[0]['baselift']
    branch_points = disc_pts    

    return base_point, base_sheets, branch_points, monodromy, G



    

if __name__=='__main__':
    from sympy.abc import x,y

    f0 = y**3 - 2*x**3*y - x**8  # Klein curve
    f1 = (x**2 - x + 1)*y**2 - 2*x**2*y + x**4
    f2 = -x**7 + 2*x**3*y + y**3
    f3 = (y**2-x**2)*(x-1)*(2*x-3) - 4*(x**2+y**2-2*x)**2
    f4 = y**2 + x**3 - x**2
    f5 = (x**2 + y**2)**3 + 3*x**2*y - y**3
    f6 = y**4 - y**2*x + x**2   # case with only one finite disc pt
    f7 = y**3 - (x**3 + y)**2 + 1
    f8 = (x**6)*y**3 + 2*x**3*y - 1
    f9 = 2*x**7*y + 2*x**7 + y**3 + 3*y**2 + 3*y
    f10= (x**3)*y**4 + 4*x**2*y**2 + 2*x**3*y - 1
    
    f = f10
   
#     import cProfile, pstats
#     cProfile.run('mon = M.monodromy()','monodromy.profile')
#     p = pstats.Stats('monodromy.profile')
#     p.strip_dirs()
#     p.sort_stats('time').print_stats(25)
#     p.sort_stats('cumulative').print_stats(25)
#     p.sort_stats('calls').print_stats(25)
