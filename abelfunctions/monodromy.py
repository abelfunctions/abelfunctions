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





class Monodromy(object):
    """
    Class defining the monodromy group of a complex plane algebraic
    curve.
    """
    def __init__(self, f, x, y, kappa=3.0/5.0):
        """
        The monodromy group corresponding to the complex plane algebraic curve
        `f = f(x,y)`.

        Inputs:
        
        -- f,x,y: a Sympy/Sage plane algebraic curve

        -- kappa: a relaxation factor used to determine the radius of the
           monodromy path circles about the branch points. If `kappa = 1`
           then the radius of the circles are 1/2 the distance to the 
           nearest branch points.
        """
        self.f = f
        self.x = x
        self.y = y
        
        dfdx = sympy.diff(f, x).simplify()
        dfdy = sympy.diff(f, y).simplify()

        self._f   = sympy.lambdify((x,y), f, "mpmath")
        self.dfdx = sympy.lambdify((x,y), dfdx, "mpmath")
        self.dfdy = sympy.lambdify((x,y), dfdy, "mpmath")
        
        self.deg = sympy.degree(f,y)
        
        self.kappa = kappa

        self._base_point = None
        self._base_lift  = None
        self._discriminant_points = None
        self._monodromy_graph = None
        self._monodromy_graph = self.monodromy_graph()
        self._monodromy = None


    def __repr__(self):
        return "Monodromy group of the Riemann surface defined by the algebraic curve %s." %(self.f)



    def __str__(self):
        return self.__repr__()



    def base_point(self):
        """
        Returns the base point (b0) of the monodromy group.

        This is a fixed point that lies to the left of all
        discriminant points of the algebraic curve.
        """    
        return self._base_point


    def base_lift(self):
        """
        Returns the ordered lift of the base point. That is, the first
        element corresponds to sheet 0, the next sheet 1, etc.
        """
        return self._base_lift


    def base_sheets(self):
        """
        Same as Monodromy.base_lift(). 
        """
        return self.base_lift()


    def branch_points(self):
        """
        Returns the branch points of the algebraic curve. These are the
        discriminant points that have non-identity monodromy elements.
        """
        return self.discriminant_points()


    def discriminant_points(self):
        """
        Returns the list of discriminant points of a plane algebraic
        curve `f = f(x,y)` in order.
        """
        # self.monodromy_graph requires that we compute the discriminant
        # points, unordered, first. However, once we order the points
        # we set the result as self._discriminant_points
        if self._discriminant_points:
            return self._discriminant_points

        x, y = self.x, self.y
        p    = sympy.Poly(self.f,[x,y])
        res  = sympy.Poly(sympy.resultant(p,p.diff(y),y),x)
        
        # compute the numerical roots
        dps = sympy.mpmath.mp.dps
        rts = [rt for rt,ord in res.all_roots(multiple=False, radicals=False)]
        rts = map(lambda z: sympy.N(z,n=dps).as_real_imag(), rts)
        disc_pts = map(lambda z: sympy.mpmath.mpc(*z), rts)

        return disc_pts



    def monodromy_graph(self):
        """
        Constructs a NetworkX graph describing the path connectedness of the
        monodromy paths. Each node has the following data attached to it:
        
        * pos: a tuple (b_re, b_im) containing the real and imaginary
               parts of the discriminant point.

        * value: the discriminant point (as a point in the complex plane)

        * radius: the radius of the monodromy path circle about the 
                  discriminant point

        * type: the 'type' of the vertex ('simple', 'node', 'vpoint', 
                'vpoint node') using the definitions of [FSK]

        * root: the index of the root vertex. This is the vertex that is
                closest to the base point

        * base point: the base point of the monodromy group. Given to
                      each node for ease of access / convenience

        * string: a piece of data used to determine which vertices to 
                  conjugate by when determining the monodromy from the 
                  initial monodromy.
                  
        * conjugates: a list of vertices / discriminant points to conjugate
                      by when constucting the monodromy path from the
                      base point to the discriminant point and back

        """
        if self._monodromy_graph:
            return self._monodromy_graph

        disc_pts = self.discriminant_points()

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
                            if bi != bd] + [10] ) * self.kappa / 2.0
        
        b0 = bd - base_radius
        disc_pts = sorted(disc_pts, key=cmp)
        bd_index = disc_pts.index(bd)

        # store sorted discriminant points and base point data
        self._discriminant_points = disc_pts
        self._base_point = b0


        # XXX UGLY...but I can't find a better way to convert a polynomial
        # to something that outputs mpc type roots
        p         = sympy.poly(self.f.subs(self.x,b0), self.y)
        coeffs    = p.all_coeffs()
        coeffs_b0 = map(lambda c: c.evalf(subs={self.x:b0}, 
                                          n = sympy.mpmath.mp.dps), coeffs)
        coeffs_b0 = map(lambda z: sympy.mpmath.mp.mpc(*(z.as_real_imag())),
                        coeffs_b0)
        self._base_lift  = sympy.mpmath.polyroots(coeffs_b0)


        # Compute minimal spanning tree. the weights are the distances
        # between the nodes. the spanning tree algorihtm is from a
        # Sage implementation of Prim with a designated starting node
        # since
        n = len(disc_pts)
        G = nx.complete_graph(n)
        weight_function = lambda e: sympy.mpmath.fabs(disc_pts[e[1]]-disc_pts[e[0]])
        spanning_tree = prim_fringe(G, weight_function=weight_function, 
                                    starting_vertex=bd_index)
        G = nx.DiGraph()
        G.add_edges_from(spanning_tree)

        #
        # Compute path radii for each discriminant point and store
        # radii and position data to the graph along with other vertex data.
        #
        for i in range(len(disc_pts)):
            disc_pt = disc_pts[i]
            rho  = min( [abs(disc_pt - disc_pts[j])
                         for j in range(len(disc_pts)) if j != i] )
            radius = rho * self.kappa / 2.0
            
            # store useful data to graph
            G.node[i]['pos']        = (sympy.mpmath.re(disc_pt), 
                                       sympy.mpmath.im(disc_pt))
            G.node[i]['value']      = disc_pt
            G.node[i]['radius']     = radius
            G.node[i]['type']       = 'simple'
            G.node[i]['root']       = bd_index
            G.node[i]['basepoint']  = b0
            G.node[i]['string']     = []
            G.node[i]['conjugates'] = []

        #
        # Compute graph indices: these determine which sides of the
        # monodromy circles the edges of the graph are connected to.
        #
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



    @cached_function
    def _special_vertices(self):
        """
        Returns the nodes, v-points, and v-point nodes of the initial
        monodromy graph.
        """
        G = self.monodromy_graph()
        return [n for n,data in G.nodes(data=True) if data['type']!='simple']

    @cached_function
    def _vpoints(self):
        """
        Returns the v-points of the monodromy graph.
        """
        G = self.monodromy_graph()
        return [n for n,data in G.nodes(data=True) if data['type']=='v-point']

    @cached_function
    def _nodes(self):
        """
        Returns the nodes of the monodromy graph.
        """
        G = self.monodromy_graph()
        return [n for n,data in G.nodes(data=True) if data['type']=='node']

    @cached_function
    def _vpoint_nodes(self):
        """
        Returns the v-points of the monodromy graph that are also nodes.
        """
        G = self.monodromy_graph()
        return [n for n,data in G.nodes(data=True) if data['type']=='v-point node']

    @cached_function
    def _endpoints(self):
        """
        Returns the endpoints of the monodromy graph.
        """
        G = self.monodromy_graph()
        return [n for n,deg in G.out_degree_iter() if deg == 0]


    def show_paths(self):
        """
        Plot all paths of the monodromy group.
        """
        G   = self.monodromy_graph()
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        
        # plot the location of the base_point
        a = self.base_point()
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

            
        
    def _initial_monodromy_path(self, i, Npts=4):
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

        G    = self.monodromy_graph()
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



    def plot_initial_monodromy_path(self, i, Npts=4, eps=0.1):
        """
        Plots the initial monodromy path. Used for testing and
        debugging purposes.
        """
        # plot the monodromy graph
        self.show_paths()
        fig = plt.gcf()
        ax  = fig.gca()

        # compute the path points
        path_points = self._initial_monodromy_path(i, Npts=Npts)
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


    def initial_monodromy(self, i, Npts=4, lift_paths=False):
        """
        Returns the initial monodromy corresponding to disciminant
        point `i`.  That is, the permutation of sheets as we go around
        discriminant point `i` using the initial monodromy paths.
        """
        n = self.deg

        # compute derivatives. optimize for return type
        f = self._f
        dfdx = self.dfdx
        dfdy = self.dfdy

        # obtain interpolating points in path
        path = self._initial_monodromy_path(i, Npts=Npts)

        # note that the order of base_lift is fixed at creation of the
        # class / monodromy group.
        base_point = self.base_point()
        base_lift  = self.base_lift()
        prev_rts   = [yi for yi in base_lift]

        if lift_paths:
            lift_points = [base_lift]

        yi   = [0]*n
        yim1 = base_lift
        xim1 = base_point
        for xi in path[1:]:
            # compute numerical approximation of the next set of roots
            # using Taylor series. This allows us to use fewer
            # interpolating points. Use generators for fast creation.
            dx = xi - xim1
            f_xi  = lambda y: f(xi,y)
            df_xi = lambda y: - dx * dfdx(xi,y) / dfdy(xi,y)
            for j in xrange(n):
                yp = - dfdx(xim1,yim1[j]) / dfdy(xim1,yim1[j])
                dy = yp * dx + sympy.mpmath.eps       # in case yp == 0
                yi_approx = yim1[j] + dy
                guess = (yi_approx - dy, yi_approx, yi_approx + dy)
                yi[j] = sympy.mpmath.findroot(f_xi, guess, df=df_xi,
                                              solver='muller')
            
            # for plotting purposes, we optionally store the lift
            if lift_paths:
                lift_points.append([yij for yij in yi])
            
            yim1 = yi
            xim1 = xi

        # return the lift points, if requested. Otherwise, just return
        # the initial monodromy permutation.
        if lift_paths: return lift_points
        else:          return matching_permutation(base_lift, yi)


    
    def plot_initial_monodromy_lift(self, i, Npts=4):
        """
        Plots the lift [y1,...,yn] on the complex plane as x varies 
        Along an initial monodromy path.
        """
        lift = self.initial_monodromy(i, Npts=Npts, lift_paths=True)
        clrs = ['b','g','r','k','c','m','y']*int(self.deg/7+1)

#         for i in xrange(self.deg):
#             yi = [lift[n][i] for n in xrange(len(lift))]
#             yi_re = [sympy.re(yij) for yij in yi]
#             yi_im = [sympy.im(yij) for yij in yi]
#             ax.plot(yi_re, yi_im, color=clrs[i], 
#                     linestyle='--', linewidth=3*(i+1), alpha=0.4)

#         plt.show()

        plt.ion()


        # scan ahead to figure out the axes
        axis = [-1,1,-1,1]
        for j in xrange(len(lift)):
            for i in xrange(self.deg):
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

        xdata = {}
        ydata = {}
        lines = range(self.deg)
        for i in xrange(self.deg):
            y = lift[0][i]
            xdata[i] = [y.real]
            ydata[i] = [y.imag]
            lines[i], = plt.plot(y.real, y.imag, color=clrs[i],
                                 linestyle='--', linewidth=4*(i+1), alpha=0.3)


        for j in xrange(len(lift)):
            for i in xrange(self.deg):
                y = lift[j][i]

                xdata[i].append(y.real)
                ydata[i].append(y.imag)

                lines[i].set_xdata(xdata[i])
                lines[i].set_ydata(ydata[i])
            plt.draw()


    def _construct_position_tree(self):
        """
        Returns the "relative position tree", a list giving the relative
        ordering of the branch points.

        See Section 3 of [FCS].
        """
        G = self.monodromy_graph()

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


        #
        # (0) Gather special nodes
        #
        root = G.node[0]['root']
        endpoints = self._endpoints()
        vpoints = self._vpoints()
        nodes = self._nodes()
        vpoint_nodes = self._vpoint_nodes()


        # caching...
        if len(G.node[root]['string']) == len(self.discriminant_points()):
            return G.node[root]['string']

        #
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


        #
        # (2) create a stack of v-points, nodes, and v-point nodes
        #     and exhaust to construct ordering tree
        #
        # in each iteration, we check the following
        # 1. find a node / v-point that can be resolved. This is possible
        #    when all successors have a full string attached in the node
        #    data field 'string'. if none can be found then move to next
        #    special point.
        # 2. when a special vertex has enough filled strings,
        special_vertices = self._special_vertices()
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


    def monodromy(self, Npts=4):
        """
        Returns the monodromy group.

        Note: see page 541 of [F...
        """
        if self._monodromy:
            return self._monodromy

        G = self.monodromy_graph()
        N = len(self.discriminant_points())

        monodromy = [self.initial_monodromy(i, Npts=Npts) for i in range(N)]
        position_tree = self._construct_position_tree()

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

        # cache results
        self._monodromy = monodromy

        return monodromy        

    def hurwitz_system(self):
        """
        Returns the Hurwitz system:

            [base_point, base_sheets, branch_points, monodromy]

        where
        
        * base_point: the base point of the monodromy group in the
                      complex x-plane
        * base_sheets: the ordered sheets lying above the base point
        * branch_points: the locations of the ordered branch points in 
                         the complex x-plane
        * monodromy: the corresponding monodromy group permutation elts
        """
        mon = self.monodromy()
        return self.base_point(), self.base_sheets(), self.branch_points(), mon



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
    
    f  = f2
    M = Monodromy(f,x,y)
   
#     import cProfile, pstats
#     cProfile.run('mon = M.monodromy()','monodromy.profile')
#     p = pstats.Stats('monodromy.profile')
#     p.strip_dirs()
#     p.sort_stats('time').print_stats(25)
#     p.sort_stats('cumulative').print_stats(25)
#     p.sort_stats('calls').print_stats(25)
