"""
Riemann Surface Path Factory
============================

This module implements the :class:`RiemannSurfacePathFactory` class, a
class for generating :class:`RiemannSurfacePath` objects from various
kinds of input data.

Authors
-------

* Chris Swierczewski (January 2013)

"""

def path_segments_from_cycle(cycle, G, base_point=None):
    """Converts a homology cycle encoding to a path segment data list.

    Given a cycle, which is a list of the form .. math::

        \left( \ldots , s_i, (b_i,n_i) , \ldots \right)

    where :math:`s_i` is a sheet index, :math:`b_i` is a branch point,
    and :math`n_i` is the number of times and direction one goes around
    the branch point, return a list of path segments parameterizing the
    input cycle.

    The path segment is constructed by performing repeated calls to
    :function:`path_around_branch_point` and
    :function:`path_around_infinity`.

    Auguments
    ---------
    cycle : list
        A cycle in the form as output by :function:`homology`.
    G : networkx.Graph
        The monodromy graph, as output by :function:`monodromy`.
    base_point : complex, optional
        A custom base point. :function:`monodromy` will choose a default
        base point unless otherwise given.

    Returns
    -------
    list
        A list of tuples with each tuple representing a single segment
        of the path. Lines are represented as :math:`(z_0,z_1)` tuples
        and semicircles are given as :math:`(R,w,\text{arg},\text{dir}`
        tuples.
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
    """Returns a list of tuples paramterizing the x-part of the path
    starting from the base point and going around infinity.

    Arguments
    ---------
    G : networkx.Graph
        The "monodromy graph", as computed by monodromy_graph().
    bpt : complex
        Index of the target branch point
    rot : int
        Rotation number and direction of the path going around branch
        point `bpt`.

    Returns
    -------
    list
        A list of tuples with each tuple representing a single segment
        of the path. Lines are represented as :math:`(z_0,z_1)` tuples
        and semicircles are given as :math:`(R,w,\text{arg},\text{dir}`
        tuples.
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
    """Returns a list of tuples paramterizing the x-part of the path
    starting from the base point and going around infinity.

    Arguments
    ---------
    G : networkx.Graph
        The "monodromy graph", as computed by monodromy_graph().
    rot : int
        Rotation number and direction of the path going around infinity.

    Returns
    -------
    list
        A list of tuples with each tuple representing a single segment
        of the path. Lines are represented as :math:`(z_0,z_1)` tuples
        and semicircles are given as :math:`(R,w,\text{arg},\text{dir}`
        tuples.

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



class RiemannSurfacePathFactory(object):
    """
    """
    def __init__(self, RS):
        self.RS = RS
        self.Mon = RS.Monodromy
        self.Hom = RS.Homology

        # cache the a-, b-, and c-cycles
        self._a_cycles = None
        self._b_cycles = None
        self._c_cycles = None

    def __str__(self):
        return 'Riemann Surface Path Factory for %s'%(self.RS)

    def a_cycles(self):
        """Returns the a-cycles on the Riemann surface.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the a-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        if self._a_cycles:
            return self._a_cycles

        cyc = []



        self._a_cycles = [c for c in cyc]
        return self._a_cycles

    def b_cycles(self):
        """Returns the b-cycles on the Riemann surface.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the b-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        if self._b_cycles:
            return self._b_cycles

        cyc = []



        self._b_cycles = [c for c in cyc]
        return self._b_cycles


    def c_cycles(self):
        """Returns the c-cycles of the Riemann surface.

        The a- and b- cycles of the Riemann surface are formed from
        linear combinations of the c-cycles. These linear combinations
        are obtained from the :method:`linear_combinations` method.

        .. note::

            It may be computationally more efficient to integrate over
            the (necessary) c-cycles and take linear combinations of the
            results than to integrate over the a- and b-cycles
            separately. Sometimes the column rank of the linear
            combination matrix (that is, the number of c-cycles used to
            construct a- and b-cycles) is lower than the size of the
            homology group and sometimes the c-cycles are simpler and
            shorter than the homology cycles.

        Returns
        -------
        list, RiemannSurfacePath
            A list of the c-cycles of the Riemann surface as
            :class:`RiemannSurfacePath` objects.
        """
        if self._c_cycles:
            return self._c_cycles

        cyc = []

        self._c_cycles = [c for c in cyc]
        return self._c_cycles


    def linear_combinations(self):
        pass
