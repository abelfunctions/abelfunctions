"""
Monodromy
"""

import pdb

import functools

import numpy
import scipy
import sympy
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from utilities import cached_function, cached_property



class Permutation(object):
    """
    A permutation.

    Examples::
    We create the permutation ``p = 1->1, 2->4, 3->2, 4->3``.
    
        >>> p = Permutation([1,4,2,3])

    We can multiply permutations together. Let ``q`` be the transposition
    ``1->2, 2->1``. The product ``qp`` represents the permutation obtained
    by applying ``p`` to the identity ``[1,2,3,4]`` first and then ``q``.

        >>> q = Permutation([2,1])
        >>> q*p
        [2, 4, 1, 3]
        >>> p*q
        [4, 1, 2, 3]

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

        >>> p = Permutation([[1,2],[3],[4]])
        >>> p._list
        [2, 1, 3]

        >>> q = Permutation([[2,4],[1,3]])
        >>> q._list
        [3, 4, 1, 2]
        """
        degree = max([1] + [max(cycle + [1]) for cycle in cycles])
        l = range(1,degree+1)
        for cycle in cycles:
            if not cycle:
                continue
            first = cycle[0]
            for i in range(len(cycle)-1):
                l[cycle[i]-1] = cycle[i+1]
            l[cycle[-1]-1] = first

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
        if isinstance(i,int) and 1 <= i <= len(self):
            return self[i-1]
        else:
            raise TypeError, "i (= %s) must be an integer between %s and %s" %(i,1,len(self))        

    def index(self, key):
        return self._list.index(key)

    def action(self, a):
        """
        Returns the action of the permutation on a list.

        **Examples**

            >>> p = Permutation([1,4,2,3])
            >>> p.action(['a','b','c','d'])
            ['a', 'd', 'b', 'c']
        """
        if len(a) != len(self):
            raise ValueError, "len(a) must equal len(self)"
        return map(lambda i: a[self[i]-1], range(len(a)))

    def inv(self):
        """
        Returns the inverse permutation.
        """
        l = range(len(self))
        for i in range(len(self)):
            l[self(i+1)-1] = i+1

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
                perm[j] = i+1
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
    def __init__(self, f, x, y, 
                 dtype=numpy.complex, kappa=4.0/5.0, use_mpmath=False):
        """
        The monodromy group corresponding to the complex plane algebraic curve
        `f = f(x,y)`.

        Inputs:
        
        -- f,x,y: a Sympy/Sage plane algebraic curve

        -- dtype: (default: numpy.complex) a Numpy dtype where all numerical
        computations are to take place
        """
        self.f = f
        self.x = x
        self.y = y
        self.deg = sympy.degree(f,y)
        
        self.kappa = kappa
        self.dtype = dtype

        self._discriminant_points = None
        self._path_points         = None
        self._use_mpmath          = use_mpmath
        self._monodromy_graph     = self.monodromy_graph()



    def __repr__(self):
        return "Monodromy group of the Riemann surface defined by the algebraic curve %s." %(self.f)



    def __str__(self):
        return self.__repr__()



    @cached_function
    def branch_points(self):
        """
        Returns the branch points of the algebraic curve. These are the
        discriminant points that have non-identity monodromy elements.
        """
        pass



    @cached_function
    def permutations(self):
        """
        Returns the monodromy group: a list of branch points and their 
        corresponding permutations.
        """
        pass



    def discriminant_points(self):
        """
        Computes a list of the  discriminant points of a plane algebraic 
        curve `f = f(x,y)` with accuract
        """
        # self.monodromy_graph requires that we compute the discriminant
        # points, unordered, first. However, once we order the points
        # we set the result as self._discriminant_points
        if self._discriminant_points:
            return self._discriminant_points

        x, y = self.x, self.y
        p    = sympy.Poly(self.f,[x,y])
        res  = sympy.Poly(sympy.resultant(p,p.diff(y),y),x)
        rts  = res.all_roots(multiple=False,radicals=False)

        # XXX
        # the following is a strange hack since Add objects cannot be
        # converted to MPF/MPCs for some reason
        #
        # also: precision needs to be explicity given
        # XXX
        if self._use_mpmath:
            prec = sympy.mpmath.mp.dps
            disc_pts = map(sympy.mpmath.mpmathify,
                           [str(rt.n(prec)).replace('*I','j') for rt,_ in rts])
        else:
            disc_pts = [self.dtype(rt) for rt,_ in rts]

        return disc_pts



    def base_point(self):
        """
        Returns the base point (b0) of the monodromy group.

        This is a fixed point that lies to the left of all
        discriminant points of the algebraic curve.
        """    
        return self._monodromy_graph.node[0]['base point']



    @cached_function
    def monodromy_graph(self):
        """
        Constructs a NetworkX graph describing the path connectedness of the
        monodromy paths. Call several subroutines to compute:

        * paths -- compute path indicators
        * interpolating_points -- computes interpolating points along each
          edge and at each node (both upper and lower points on circle)
        * 
        """
        disc_pts = self.discriminant_points()
        
        # sort the discriminant points by angle.
        # notation:
        #     - b0 = base point
        #     - bd = discriminant point closest to base point
        #     - bd_index = index of bd. its ranking in the sorted
        #                  discriminant points
        if self._use_mpmath:
            key = lambda z: 100*sympy.mpmath.re(z) - sympy.mpmath.im(z)
            cmp = lambda z: sympy.mpmath.arg(z-b0)
        else:
            key = lambda z: 100*numpy.real(z) - numpy.imag(z)
            cmp = lambda z: numpy.angle(z-b0)

        bd  = min(disc_pts, key=key)
        base_radius = min( [numpy.abs(bd - bi) for bi in disc_pts 
                            if bi != bd] ) * self.kappa / 2.0
        b0 = bd - base_radius
        disc_pts = sorted(disc_pts, key=cmp)
        bd_index = disc_pts.index(bd)

        # store sorted discriminant points
        self._discriminant_points = disc_pts

        # Compute minimal spanning tree. the weights are the distances
        # between the nodes. the spanning tree algorihtm is from a
        # Sage implementation of Prim with a designated starting node
        # since
        n = len(disc_pts)
        G = nx.complete_graph(n)
        if self._use_mpmath:
            weight_function = lambda e: sympy.mpmath.fabs(disc_pts[e[1]] - disc_pts[e[0]])
        else:
            weight_function = lambda e: numpy.abs(disc_pts[e[1]] - disc_pts[e[0]])
        spanning_tree = prim_fringe(G, weight_function=weight_function, 
                                    starting_vertex=bd_index)
        G = nx.DiGraph()
        G.add_edges_from(spanning_tree)
 
        # compute path radii for each discriminant point and store
        # radii and position data to the graph
        min_rho = 2**(-numpy.nbytes[self.dtype]*4) # minimal bits of precision
        for i in range(len(disc_pts)):
            disc_pt = disc_pts[i]
            rho  = min( [abs(disc_pt - disc_pts[j])
                         for j in range(len(disc_pts)) if j != i] )
            radius = rho * self.kappa / 2.0

            # if mp_math is not being used then check if we're below
            # the floating point calculating threshold
            if not self._use_mpmath and rho < min_rho:
                warnings.warn("Cannot accurately compute monodromy: "  + \
                              "discriminant points are too close and " + \
                              "may cause numerical errors.",
                              RuntimeWarning)
            
            # store useful data to graph
            if self._use_mpmath: 
                G.node[i]['pos'] = (sympy.re(disc_pt), sympy.im(disc_pt))
            else:
                G.node[i]['pos'] = (disc_pt.real, disc_pt.imag)
            G.node[i]['value']  = disc_pt
            G.node[i]['radius'] = radius
            G.node[i]['type']   = 'simple'
            G.node[i]['root']   = bd_index
            G.node[i]['base point'] = b0


        # compute additional graph data
        G = self._compute_path_indices(G)
        G = self._compute_vertex_types(G,source=bd_index)
        return G



    def plot_monodromy_graph(self):
        """
        Plots the monodromy graph.
        """
        G = self.monodromy_graph()
        nodes = G.nodes(data=True)
        pos = [node_data['pos'] for node, node_data in nodes]
        nx.draw(G,pos=pos)        



    def _compute_path_indices(self, G):
        """
        Determine the "path indices" of the monodromy graph. For each
        edge (i,k) assign a tuple (j,l) with values describing the 
        paths connecting one discriminant point to another.
        
        For example, (j,l) = (1,-1) means that a path is followed from 
        the right side of 
        """
        for (i,k) in G.edges():
            bi = G.node[i]['value']
            bk = G.node[k]['value']
            if self._use_mpmath: d = sympy.mpmath.re(bk-bi)
            else:                d = numpy.real(bk-bi)
            R = max( [G.node[i]['radius'], G.node[k]['radius']] )
            if d > R:    G[i][k]['index'] = (1,-1)
            elif d < -R: G[i][k]['index'] = (-1,1)
            else:        G[i][k]['index'] = (-1,-1)

        return G



    def _compute_vertex_types(self, G, source):
        """
        Comptues the vertex types as defined in [FKS]:
        
        * 'node' -- a parent vertex of the graph with multiple children
        * 'v-point' -- a vertex of special type described in Section 3 of [FKS]
        * 'endpoint' -- a childless vertex

        Note: the "children" of a vertex are neighbors that are further away 
        from the starting vertex.

        Note: requires that path indices are computed first.
        """
        edges = list(nx.dfs_edges(G,source=source))
        for j in range(len(edges)-1):
            e_left  = edges[j]
            e_right = edges[j+1]

            # check for 'v-point': when a path contains a sequence of
            # edges of the form [..., bj^(I)], [bj^(I), ...] where I
            # is a matching index.
            if e_left[1] == e_right[0]:                
                left_index  = G.edge[e_left[0]][e_left[1]]['index']
                right_index = G.edge[e_right[0]][e_right[1]]['index']
                if left_index[1] == right_index[0]:
                    G.node[e_left[1]]['type'] = 'v-point'
            # check for 'node': a discriminant point where several
            # branches meet. i.e. where the number of successors is
            # greater than 1
            if len(G.successors(e_left[0])) > 1:
                G.node[e_left[0]]['type'] = 'node'

        return G



    @cached_function
    def vpoints(self):
        """
        Returns the v-points of the monodromy graph.
        """
        G = self.monodromy_graph()
        return [n for n,data in G.nodes(data=True) if data['type']=='v-point']



    @cached_function
    def nodes(self):
        """
        Returns the nodes of the monodromy graph.
        """
        G = self.monodromy_graph()
        return [n for n,data in G.nodes(data=True) if data['type']=='node']



    def show_paths(self):
        """
        Plot all paths of the mmonodromy group.
        """
        G = self.monodromy_graph()
        fig = plt.figure()
        ax  = fig.add_subplot(111)

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

            
        
    def initial_monodromy_path(self, i, Npts=8):
        """
        Returns a list of points on the initial monodromy path.
        """
        # choose interpolating functions based on underlying type
        if self._use_mpmath:
            t_pts = sympy.mpmath.linspace(0, 1, Npts, endpoint=False)
            circle = lambda R, z0, arg: [R*sympy.mpmath.exp(sympy.mpmath.j*(sympy.mpmath.pi*t + arg)) + z0 for t in t_pts]
            line = lambda start, end: [start*(1-t) + end*t for t in t_pts]
        else:
            t_pts = numpy.linspace(0, 1, Npts, endpoint=False).tolist()
            circle = lambda R, z0, arg: [R*numpy.exp(1.0j*(numpy.pi*t + arg)) + z0 for t in t_pts]
            line = lambda start, end: [start*(1-t) + end*t for t in t_pts]


        G    = self.monodromy_graph()
        root = G.node[i]['root']
        path_points = []

        # compute interpolating points for circle / line pairs.  that
        # is, for each node in the shortest path to the target node
        # determine the interpolating points on the circle and the
        # interpolating points on the line connecting to the next
        # discriminant point.
        path_vertices = nx.shortest_path(G, source=root, target=i)
        prev_node = root
        for idx in range(len(path_vertices)-1):
            curr_node   = path_vertices[idx]
            curr_radius = G.node[curr_node]['radius']
            curr_pos    = G.node[curr_node]['pos']

            next_node   = path_vertices[idx+1]
            next_radius = G.node[next_node]['radius']
            next_pos    = G.node[next_node]['pos']

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
                z0  = curr_pos[0] + 1.0j*curr_pos[1]
                arg = numpy.pi if prev_edge_index[1] == -1 else 0
                circ_pts = circle(curr_radius, z0, arg)
                path_points.extend(circ_pts)


            # draw line to next discriminant point
            start = curr_pos[0] + 1.0j*curr_pos[1] + \
                curr_edge_index[0]*curr_radius
            end   = next_pos[0] + 1.0j*next_pos[1] + \
                curr_edge_index[1]*next_radius
            line_pts = line(start, end)
            path_points.extend(line_pts)

            # update previous point
            prev_node = curr_node


        # we have now arrived at the circle encircling the desired
        # discriminant point. There's a special case for when we just
        # encircle the root node.
        if len(path_vertices) == 1:
            next_pos    = G.node[root]['pos']
            next_radius = G.node[root]['radius']
            curr_edge_index = (-1,-1)

        z0  = next_pos[0] + 1.0j*next_pos[1]
        arg = numpy.pi if curr_edge_index[1] == -1 else 0
        circ_pts = circle(next_radius, z0, arg) + \
            circle(next_radius, z0, arg + numpy.pi)
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
        path_points = self.initial_monodromy_path(i, Npts=Npts)
        N = len(path_points)

        # plot forward path
        if self._use_mpmath:
            topos = lambda pp: (sympy.re(pp), sympy.im(pp))
        else:
            topos = lambda pp: (numpy.real(pp), numpy.imag(pp))
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



    def initial_monodromy(self, i, Npts=8, lift_paths=False, *args, **kwds):
        """
        Returns the initial monodromy corresponding to disciminant
        point `i`.  That is, the permutation of sheets as we go around
        discriminant point `i` using the initial monodromy paths.
        """
        # compute derivatives. optimize for return type
        dfdx = sympy.diff(self.f, self.x)
        dfdy = sympy.diff(self.f, self.y)
        if self._use_mpmath:
            dfdx = sympy.lambdify([x,y], dfdx, "mpmath")
            dfdy = sympy.lambdify([x,y], dfdy, "mpmath")
        else:
            dfdx = sympy.lambdify([x,y], dfdx, "numpy")
            dfdy = sympy.lambdify([x,y], dfdy, "numpy")

        # lift function: for each x = xi compute the roots yi_j lying
        # above xi
        lift  = lambda a: sympy.nroots(self.f.subs({self.x : a}), *args, **kwds)

        # obtain interpolating points in path
        path = self.initial_monodromy_path(i, Npts=Npts)

        # rewrite this part so it's computed at creation of monodromy
        base_point = self.base_point()
        base_lift  = lift(base_point)
        prev_rts   = [yi for yi in base_lift]

        if lift_paths:
            lift_points = [base_lift]

        yim1 = base_lift
        xim1 = base_point
        for xi in path:
            # compute numerical approximation of the next set of roots
            # using Taylor series. This allows us to use fewer
            # interpolating points
            dx = xi - xim1
            yi = lift(xi)
            yi_approx = [yim1[j] - dx * dfdx(xim1,yim1[j]) / dfdx(xim1,yim1[j])
                         for j in xrange(self.deg)]

            rho = matching_permutation(yi, yi_approx)
            yi = rho.action(yi)
            
            # for plotting purposes, we optionally store the lift
            if lift_paths:
                lift_points.append(yi)
            
            yim1 = yi
            xim1 = xi


        # return the lift points, if requested. Otherwise, just return
        # the initial monodromy permutation.
        if lift_paths:
            if self._use_mpmath:
                return lift_points
            else:
                return numpy.array(lift_points, dtype=self.dtype)
        else:
            return matching_permutation(base_lift, yi)


    
    def plot_initial_monodromy_lift(self, i, Npts=8, *args, **kwds):
        """
        Plots the lift [y1,...,yn] on the complex plane as x varies 
        along an initial monodromy path.
        """
        lift = self.initial_monodromy(i, Npts=Npts, lift_paths=True,
                                      *args, **kwds)
        clrs = ['b','g','r','k','c','m','y']*int(self.deg/7+1)

        fig = plt.figure()
        ax  = fig.add_subplot(111)

        for i in xrange(self.deg):
            yi = lift[:,i]
            yi_re = numpy.real(yi)
            yi_im = numpy.imag(yi)
            ax.plot(yi_re, yi_im, color=clrs[i], linewidth=3, alpha=0.6)

        plt.show()

        


if __name__=='__main__':
   from sympy.abc import x,y

   f = y**3 - 2*x**3*y - x**9
   M = Monodromy(f,x,y)
