"""
Monodromy
"""

import functools

import numpy as np
import scipy as sp
import sympy as sy
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Wedge
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

import pdb

class cachedmethod(object):
   """
   Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned 
   (not reevaluated).
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args, **kwds):
      try:
         return self.cache[(args,kwds)]
      except KeyError:
         value = self.func(*args,**kwds)
         self.cache[(args,kwds)] = value
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args,**kwds)
   def __repr__(self):
      return self.func.__doc__
   def __get__(self, obj, objtype):
      return functools.partial(self.__call__, obj)



class Permutation:
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
        Converts ``l``, a list of integers viewed as one-line permutation
        notation, into a permutation.
        """
        if isinstance(l,list): self._list = l
        else:                  self._list = list(l)
        self._hash = None

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




class Monodromy:
    def __init__(self, f, x, y):
        """
        The monodromy group corresponding to the complex plane algebraic curve
        `f = f(x,y)`.
        """
        self.f = f
        self.x = x
        self.y = y

        self.kappa = None

    def __repr__(self):
        return "Monodromy group of the Riemann surface defined by the algebraic curve %s." %(self.f)

    
        
    @cachedmethod
    def discriminant_points(self):
        """
        Computes a list of the  discriminant points of a plane algebraic 
        curve `f = f(x,y)`
        """
        resultant = sy.resultant(f,f.diff(y),y)
        return sy.roots(resultant,x).keys()


    @cachedmethod
    def monodromy_radius(self, kappa=1/2.9):
        """
        Helper function for ``monodromy``.

        Returns the radius of the circles contained in the initial 
        monodromy paths.

        INPUTS:

        - ``kappa``: a relaxation factor. Decreasing `\kappa` decreases the radius
        of the circles. Setting `\kappa = 1` would cause the circles to touch, 
        which is not recommended for computational purposes.
        """
        self.kappa = kappa

        min_rho = 2**(-np.nbytes[self.dtype]*8)      # minimal bits of precision
        rho     = sy.oo
        b       = [np.complex(pt) for pt in self.discriminant_points()]
        for b1 in b:
            for b2 in b:
                if b1 != b2:
                    dist = np.abs(b1 - b2)
                    if dist < rho: rho = dist

        if rho < min_rho:
            warnings.warn("Cannot compute monodromy: discriminant points are too close and may cause numerical errors.",
                          RuntimeWarning)

        radius = np.double(kappa*rho/2.0)
        return radius
        

    def initial_monodromy(self, kappa=1/2.9):
        """
        Compute the initial monodromy paths and "pathinds".

        The paths are represented by a list of tuples, each tuple representing a 
        line from one (labeled) discriminant point to another. That is `[i,j]` 
        indicates a straight line path from the `i`th discriminant point to the 
        `j`th discriminant point.

        The variable ``pathinds`` indicates the starting and 
        ending orientations of the paths on the path circles. That is, if 
        `[i,j]` is a path and, say, `[1,-1]` is its corresponding "``pathind``",
        this indicates that the path starts on the rightmost point on the circle
        about the `i`th discriminant point and ends on the leftmost point on the 
        circle about the `j`th discriminant point.

        INPUTS:

        - ``kappa``: a relaxation factor. Decreasing `\kappa` decreases the 
        radius of the circles. Setting `\kappa = 1` would cause the circles 
        to touch, which is not recommended for computational purposes.

        OUTPUTS:

        - ``H``: minimal spanning tree

        - ``disc_pts``: discriminant points sorted according to [FKS]

        - ``paths``: minimal spanning tree paths as described above

        - ``pathinds``: path indicators, as described above
        """
        # convert discriminant points to complex floats for performance
        radius = self.monodromy_radius(kappa=kappa)
        disc_pts = np.array(disc_pts, dtype=np.complex).tolist()

        # Compute starting point and reorder discriminant points
        key        = lambda z: 100*np.real(z) - np.imag(z)
        starting_vertex_pos = min(disc_pts, key=key)
        base_point = np.complex(starting_vertex_pos - radius)

        def cmp(z):
            return np.angle(z-base_point)

        disc_pts.sort(key=cmp)
        starting_vertex = disc_pts.index(starting_vertex_pos)

        # Compute minimal spanning tree. the weights are the distances 
        # between the nodes. the spanning tree algorihtm is from a Sage
        # implementation of Prim with a designated starting node since
        n = len(disc_pts)
        G = nx.complete_graph(n)
        weight_function = lambda e: abs(disc_pts[e[1]] - disc_pts[e[0]])
        spanning_tree = prim_fringe(G, weight_function=weight_function, 
                                    starting_vertex=starting_vertex)
        H = nx.Graph()
        H.add_edges_from(spanning_tree)

        # add node positions
        pos = {}
        for i in H.nodes_iter():
            b = np.complex(disc_pts[i])
            pos[i] = (np.real(b), np.imag(b))
            H.node[i]['pos'] = pos[i]

        #
        # Compute paths and path indicators
        #
        paths = {}
        paths = paths.fromkeys(H.edges())    
        for i,j in paths.keys():
            d = np.real(disc_pts[j] - disc_pts[i])
            if   d >= radius:  ind = (1,-1)
            elif d <= -radius: ind = (-1,1)
            else:              ind = (-1,-1)
            paths[(i,j)] = ind

        return H, disc_pts, paths, starting_vertex


    def plot_initial_monodromy(self, kappa=1/2.9):
        """
        Returns a plot of the initial monodromies of the discriminant points of an
        algebraic curve. Used for debugging purposes.

        INPUTS:

        - ``f``: a plane algebraic curve

        - ``x``: the independent variable

        - ``y``: the dependent variable

        - ``kappa``: (default ``1/2.9``) a "relaxation coefficient" for 
        determining the radius of the circles in the monodromy paths. 
        ``kappa = 1`` produces touching circles, which in most cases is 
        not computationally efficient.

        - ``verbose``: (default ``False``) Set to ``True`` for debugging
        statements

        OUTPUTS:

        - plot: a plot of the initial monodromy.


        EXAMPLES:
        """
        # compute discriminant points and radii
        disc_pts = self.discriminant_points()
        disc_pts = np.array(disc_pts, dtype=np.complex)
        radius   = self.monodromy_radius(kappa=kappa)
        ht = max(np.append(np.real(disc_pts), np.imag(disc_pts))) + radius


        # compute paths and path indices
        H, disc_pts, paths, starting_vertex = self.initial_monodromy(kappa)
        n = len(disc_pts)

        # construct figure and axes
        fig = plt.figure()
        ax  = fig.add_subplot(111,aspect=1)
        ax.axis([-ht,ht,-ht,ht])
        patches = []

        # plot circles
        for k in range(n):
            b  = disc_pts[k]
            pt = (np.real(b), np.imag(b))

            ax.text(pt[0], pt[1], str(k), ha='center', va='center', color='black')
            circle = Wedge(pt,radius,theta1=0,theta2=360,width=0.1*radius)
            patches.append(circle)

        p = PatchCollection(patches, alpha=0.6)
        ax.add_collection(p)

        # plot paths between circles
        for path,ind in paths.iteritems():    
            b0 = disc_pts[path[0]]
            b1 = disc_pts[path[1]]
            x  = (np.real(b0) + ind[0]*radius, np.real(b1) + ind[1]*radius)
            y  = (np.imag(b0), np.imag(b1))
            line = Line2D(x,y, lw=2, alpha=0.6)
            ax.add_line(line)

        ht = np.max(np.abs(disc_pts)) + radius
        plt.axes([-ht,ht,-ht,ht])
        plt.show()

        return None


    def interpolate_line(self, start, end, Npts=16):
        """
        Returns a list of uniformly interpolated points between the complex numbers
        ``start`` and ``end``. The end point is not included.

        INPUTS:

        - ``start``: the starting point

        - ``end``: the ending point

        - ``Npts``: (default ``32``) the number of interpolating points to 
        compute.

        OUTPUTS:

        - list: a list of interpolating points from ``start`` to ``end`` of
        length ``Npts``

        """
        step = 1.0/Npts
        pts  = range(Npts)
        for n in range(Npts):
            t = n * step
            pts[n] = (1-t)*start + t*end

        return pts



    def interpolate_circle(self, center, radius, start, orient, Npts=32):
        """
        Returns a list of uniformly interpolated points between the complex numbers
        ``start`` and ``end``. The end point is not included.

        INPUTS:

        - ``start``: the starting point

        - ``end``: the ending point

        - ``Npts``: (default ``32``) the number of interpolating points to 
        compute.

        OUTPUTS:

        - list: a list of interpolating points from ``start`` to ``end`` of
        length ``Npts``

        """
        if orient not in [-1,1]:
            raise ValueError("Semicircle orientation must be -1 or 1.")

        step = np.pi/(Npts)
        pts  = range(Npts)
        for n in range(Npts):
            theta  = orient * n * step + start
            pts[n] = np.complex(radius*exp(1.0j*theta) + center)

        return pts




    def interpolate_path(self, b, Npts=32):
        H, disc_pts, paths, starting_vertex = self.inital_monodromy()
        verts = H.shortest_path(starting_vertex,b)
        path  = [(verts[i],verts[i+1]) for i in range(len(verts)-1)]
        path_points = []

        if path == []:
            start = pi
            pts = circ_interp(disc_pts[starting_vertex], radius, start=start, orient=1, Npts=Npts)
            path_points.extend(pts)
            pts = circ_interp(disc_pts[starting_vertex], radius, start=start+pi, orient=1, Npts=Npts)
            path_points.extend(pts)
        else:    
            # 1)
            side = -1    # use to keep track of where we were when computing where to go on the circles
            for seg in path:
                # obtain the path segment information
                ind = paths[seg]          # which sides of the circle to travel to / from

                # if the circle side we're starting on is not equal to the
                # path index, create a circle (i.e. switch sides
                if side != ind[0]:
                    if side == -1: start = pi
                    else:          start = 0
                    pts = circ_interp(disc_pts[seg[0]], radius, start=start, orient=1, Npts=Npts)
                    path_points.extend(pts)

                # construct line:
                pts = line_interp(disc_pts[seg[0]] + radius*ind[0], disc_pts[seg[1]] + radius*ind[1], Npts=Npts)
                path_points.extend(pts)

                # update which side of the circle we're on
                side = ind[1]

            return_path = [z for z in path_points]
            return_path.reverse()


            # 2) Encircle the target branch point
            if ind[1] ==  1: start = 0
            else:            start = pi
            pts = circ_interp(disc_pts[seg[1]], radius, start=start, orient=1, Npts=Npts)
            path_points.extend(pts)
            pts = circ_interp(disc_pts[seg[1]], radius, start=start+pi, orient=1, Npts=Npts)
            path_points.extend(pts)

            # add the last point on the circle
            path_points.append(disc_pts[seg[1]] + ind[1]*radius)


            # 3) Add the return path and the base point
            path_points.extend(return_path)
            path_points.append(disc_pts[starting_vertex]-radius)

        return path_points



        


if __name__=='__main__':
    from sympy.abc import x,y

    f = y**3 - 2*x**3*y - x**9


"""
    print "Example curve..."
    sy.pprint(f)

    print "\nComputing discriminant points..."
    disc_pts = discriminant_points(f,x,y)
    for b in disc_pts:
        print "\t",np.complex(b)

    print "\nComputing radius..."
    radius = monodromy_radius(disc_pts)
    print "\t", radius

#    pdb.set_trace()

    print "\nComputing initial monodromy..."
    H, disc_pts, paths, starting_vertex = initial_monodromy(disc_pts)
    print "\n\tStarting vertex...\n\t\tv = ", starting_vertex
    print "\n\tSorted discriminant points..."
    for b in disc_pts: print "\t\t", b
    print "\n\tPaths and path indicators..."
    print "\t\t", paths
    print "\n\tGraph..."
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect=1)
    d = disc_pts
    pos = dict([(i,(np.real(d[i]),np.imag(d[i]))) for i in H.nodes()])
    print pos
    #nx.draw(H,pos=pos,ax=ax)
    #plt.show()

    print "\n\tGraph..."
    plot_initial_monodromy(f,x,y)
    
    
    print "...done."
"""
