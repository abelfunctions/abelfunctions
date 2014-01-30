"""
Riemann Surface Path Factory
============================

This module implements the :class:RiemannSurfacePathFactory class, a
class for generating :class:RiemannSurfacePath objects from various
kinds of input data.

Authors
-------

* Chris Swierczewski (January 2013)
"""


def polyroots(f,x,y,xi,types='numpy'):
    """
    Helper function for computing multiprecise roots of polynomials
    using `sympy.mpmath`.

    Precision is set by modifying `sympy.mpmath.mp.dps`.

    Input:

    - `f,x,y`: a complex plane algebraic curve in x,y with
      `sympy.mpmath.mpc` coefficients

    - `xi`: a complex x-point

    Output:

    - the multiprecise roots of ``f(xi,y) = 0``.
    """
    dps = sympy.mpmath.mp.dps
    p = f.as_poly(y)

    if types == 'mpmath':
        coeffs = [c.evalf(subs={x:xi},n=dps) for c in p.all_coeffs()]
        coeffs = [sympy.mpmath.mpc(*(z.as_real_imag())) for z in coeffs]
        return sympy.mpmath.polyroots(coeffs)
    else:
        coeffs = [c.evalf(subs={x:xi},n=15) for c in p.all_coeffs()]
        coeffs = [numpy.complex(z) for z in coeffs]
        poly = numpy.poly1d(coeffs)
        return (poly.r).tolist()


def path_segments_from_cycle(cycle, G, base_point=None):
    """
    Given a cycle, which is a list of the form

        (...,s_i,(b_i,n_i),....)

    where s_i is a sheet index, b_i is a branch point, and n_i is the
    number of times and direction one goes around the branch point,
    return a list of path segments parameterizing the input cycle.

    The path segment is constructed by performing repeated calls to
    path_around_granch_point() and path_around_infinity().

    Input:

    * cycle: a cycle in the form as output by homology()

    * G: the monodromy graph, as output by monodromy()

    * base_point: (optional) a custom base point
    """
    path_segments = []

    # for each (branch point, rotation number) pair appearing in the
    # cycle, determine the path segments in the complex x-plane going
    # around that branch point "rotation number" number of times
    branch_points = [data['value'] for key,data in G.nodes(data=True)]
    for (bpt,rot) in cycle[1::2]:
        if (bpt == sympy.oo) or (bpt == numpy.Inf):
            bpt_path_seg = path_around_infinity(G,rot)
        else:
            idx = branch_points.index(bpt)
            bpt_path_seg = path_around_branch_point(G,idx,rot)
        path_segments.extend(bpt_path_seg)

    # if a custom base point is provided then add a line segment going
    # from the custom base point to the default one chosen by monodromy
    if base_point:
        seg = (base_point, G.node[0]['basepoint'])
        path_segments = [seg] + path_segments + [tuple(reversed(seg))]

    return path_segments


def path_around_branch_point(G, bpt, rot, types='numpy'):
    """
    Returns a list of tuples encoding information about the x-part of the
    Riemann Surfaace path starting from the base point going around the branch
    point, "bpt", "rot" number of times, and then returning to the starting
    point.

    Line segments are encoded as tuples (z0,z1) where z0 is the starting
    x-value and z1 is the ending x-value.

    Semicircles are encoded as tuples (R,w,arg,dir) where R is the radius, w is
    the center, arg is the starting argument (e.g. arg=0 means start on the
    right side of the circle), and dir indicates which direction to travel
    around the circle in a semicircular arc.

    Input:

    - G: the "monodromy graph", as computed by monodromy_graph()

    - bpt: the index of the target branch point

    - rot: the rotation number and direction of the path going around
    branch point "bpt".

    Output:

    A list of tuples encoding the path information.
    """
    abs = numpy.abs
    pi = numpy.pi

    # retreive the vertices between the base point vertex and the target
    # vertex.
    root = G.node[0]['root']
    path_vertices = nx.shortest_path(G, source=root, target=bpt)

    # retreive the conjugate nodes. a conjugate node is one that we traverse
    # clockwise (negative direction) instead of counter-clockwise (positive
    # direction)
    conjugates = G.node[bpt]['conjugates']

    # (1) Compute path data for semi-circle / line pairs leading to the circle
    # encircling the target branch point. (Taking conjugates into account.)
    # Conjugation will indicate whether to pass a vertex on the path towards
    # the target either above or below the vertex.
    path_data = []
    prev_node = root
    for idx in range(len(path_vertices)-1):
        curr_node   = path_vertices[idx]
        curr_radius = G.node[curr_node]['radius']
        curr_value  = G.node[curr_node]['value']

        next_node   = path_vertices[idx+1]
        next_radius = G.node[next_node]['radius']
        next_value  = G.node[next_node]['value']

        # Determine if semi-circle is needed. This is done by checking the path
        # index of the previous edge with the path index of the next edge. If
        # needed, add semicircle going in the appropriate direction where the
        # direction is determined by the conjugation list. A special case is
        # taken if we're at the root vertex.
        curr_edge_index = G[curr_node][next_node]['index']
        if prev_node == root:
            prev_edge_index = (0,-1)  # in the root node case
        else:
            prev_edge_index = G[prev_node][curr_node]['index']
        if prev_edge_index[1] != curr_edge_index[0]:
            arg = pi if prev_edge_index[1] == -1 else 0
            dir = -1 if curr_node in conjugates else 1 # XXX
            path_data.append((curr_radius, curr_value, arg, dir))

        # Add the line to the next discriminant point.
        start = curr_value + curr_edge_index[0]*curr_radius
        end   = next_value + curr_edge_index[1]*next_radius
        path_data.append((start,end))

        # Update previous point
        prev_node = curr_node

    # (2) Construct interpolating points around the target branch point. The
    # rotation number "rot" tells us how many times to go around the branch
    # point and in which direction. There's a special case for when we just
    # encircle the root node.
    if len(path_vertices) == 1:
        next_value  = G.node[root]['value']
        next_radius = G.node[root]['radius']
        curr_edge_index = (-1,-1)
    arg = pi if curr_edge_index[1] == -1 else 0
    dir = 1 if rot > 0 else -1
    circle_data = [
        (next_radius, next_value, arg, dir),
        (next_radius, next_value, arg+pi, dir)
        ]
    circle_data = circle_data * int(abs(rot))

    # (3) Construct the reverse path data. This is just the path defined in
    # part (1) above but traversed in teh reverse direction. This is just an
    # appropriate transformation on the path segment data.
    reversed_path_data = []
    for datum in reversed(path_data):
        if len(datum) == 2:
            z0,z1 = datum
            reversed_path_data.append( (z1,z0) )
        else:
            R,w,arg,dir = datum
            reversed_path_data.append( (R,w,arg+pi,-dir) )

    return path_data + circle_data + reversed_path_data


def path_around_infinity(G, rot, types='numpy'):
    """
    Returns a list of labmda functions paramterizing the path starting
    from the base point and going around infinity.

    Input:

    - G: the "monodromy graph", as computed by monodromy_graph()

    - bpt: the index of the target branch point

    - rot: the rotation number and direction of the path going around
    branch point "bpt".

    Output:

    A list of path segments ``(x(t), dxdt(t))`` for ``t \in [0,1]``
    defined by lambda functions.
    """
    abs = numpy.abs
    arg = numpy.angle
    pi = numpy.pi
    CC = numpy.complex

    # determine the center of the circle encircling the entire graph
    values = [data['value'] for node,data in G.nodes(data=True)]
    center = 0 #sum(values)/len(values)

    # the radius of the circle is the distance from the center to the furthest
    # away branch point plus the monodromy path radius at that branch point
    radius = 0
    for node,data in G.nodes(data=True):
        node_value = data['value']
        node_radius = data['radius']

        current_radius = abs(node_value) + node_radius
        radius = current_radius if current_radius > radius else radius
        if current_radius > radius:
            radius = current_radius
        max(abs(value-center) for value in values)

    # the base point is chosen to be furthest to the left of all branch points.
    # travel along the line made by the base point and the center until we
    # reach the perimeter of the circle.
    base_point = CC(G.node[0]['basepoint'])
    z0 = CC(base_point)
    arg0 = arg(z0)               # starting angle on the circle
    z1 = CC((z0/abs(z0))*radius) # starting point on the circle
    dir = -1 if rot > 0 else 1   #XXX

    # construct path
    if abs(base_point - z1) > 1e-14:
        path_data = [(base_point,z1)]
    else:
        path_data = []

    circle_data = [(radius,center,arg0,dir), (radius,center,arg0+pi,dir)]
    circle_data = circle_data * int(abs(rot))

    # (3) Construct the reverse path data. This is just the path defined in
    # part (1) above but traversed in teh reverse direction. This is just an
    # appropriate transformation on the path segment data.
    if path_data == []:
        reversed_path_data = []
    else:
        reversed_path_data = [(z1,base_point)]

    return path_data + circle_data + reversed_path_data

