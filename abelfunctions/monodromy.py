"""
Monodromy
"""

import numpy as np
import scipy as sp
import sympy as sy
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Wedge
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

import pdb


def discriminant_points(f,x,y):
    """
    Computes the discriminant points of a plane algebraic curve `f = f(x,y)`
    
    INPUTS:
        
    - ``f``: a polynomial in two variables
      
    - ``x``: the independent variable of ``f``
       
    - ``y``: the dependent variable of ``f``
        
    - ``ring``: (default: ``QQbar``) the ring over which to compute the
      discriminant points. It's recommended for application purposes that
      this ring is either `\bar{\QQ}` or `\CC`.
          
    OUTPUTS:
        
    - (list) a list of the discriminant points of the plane algebraic curve
        
    EXAMPLES:
    """
    resultant = sy.resultant(f,f.diff(y),y)
    return sy.polyroots.roots(resultant,x).keys()


def monodromy_radius(disc_pts, kappa=1/2.9, verbose=False):
    """
    Helper function for ``monodromy``.
    
    Returns the radius of the circles contained in the initial 
    monodromy paths.
    
    INPUTS:
        
    - ``disc_pts``: list of discriminant points
        
    - ``kappa``: a relaxation factor. Decreasing `\kappa` decreases the radius
      of the circles. Setting `\kappa = 1` would cause the circles to touch, 
      which is not recommended for computational purposes.
          
    OUTPUTS:
        
    - radius of circles in initial monodromy paths
        
    EXAMPLES:
    """
    min_rho = 10**(-14)  # minimal radius (for numerical purposes)
    rho     = sy.oo
    b       = [np.complex(pt) for pt in disc_pts]
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

def prim_fringe(G, weight_function=lambda e: 1, starting_vertex=None):
    """
    Variant of Prim's algorithm with starting vertex based on Sage's 
    implementation.
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


def initial_monodromy(disc_pts, kappa=1/2.9, verbose=False):
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
        
        - ``disc_pts``: the discriminant points of the algebraic curve
        
        - ``kappa``: a relaxation factor. Decreasing `\kappa` decreases the 
          radius of the circles. Setting `\kappa = 1` would cause the circles 
          to touch, which is not recommended for computational purposes.
          
        - ``verbose``: output debug statements
        
    OUTPUTS:
        
        - ``H``: minimal spanning tree
            
        - ``disc_pts``: discriminant points sorted according to [FKS]
            
        - ``paths``: minimal spanning tree paths as described above
            
        - ``pathinds``: path indicators, as described above
        
    EXAMPLES:
        
        sage: 1+1
        2
    """
    # convert discriminant points to complex floats for performance
    radius = monodromy_radius(disc_pts, kappa=kappa, verbose=verbose)
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


def plot_initial_monodromy(f,x,y, kappa=1/2.9):
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
    disc_pts = discriminant_points(f,x,y)
    disc_pts = np.array(disc_pts, dtype=np.complex)
    radius   = monodromy_radius(disc_pts, kappa=kappa)
    ht = max(np.append(np.real(disc_pts), np.imag(disc_pts))) + radius
    
    
    # compute paths and path indices
    H, disc_pts, paths, starting_vertex = initial_monodromy(disc_pts, kappa)
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


def interpolate_line(start, end, Npts=16):
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



def interpolate_circle(center, radius, start, orient, Npts=32):
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



        


if __name__=='__main__':
    from sympy.abc import x,y

    f = y**3 - 2*x**3*y - x**9

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
