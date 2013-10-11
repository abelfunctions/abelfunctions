"""
Homology
"""
import numpy
import scipy
import sympy
import networkx as nx

from operator import itemgetter
from abelfunctions.monodromy import Permutation, monodromy
from abelfunctions.singularities import genus
from abelfunctions.utilities import cached_function

import pdb


def find_cycle(pi, j):
    """
    Returns the cycle (as a list) of the permutation pi
    containing j.

    Note: The ordering of a cycle is important for the homology functions
    since cycles are used to index dictionaries. For example, although
    "(0 7 4)" and "(7 4 0)" are the same cycle, this function outputs
    the cycles sith the smallest element of the cycle first.
    """
    if isinstance(pi, list):
	pi = Permutation(pi)

    cycles = pi._cycles
    for cycle in cycles:
	if j in cycle:
	    return tuple(reorder_cycle(cycle, min(cycle)))


def smallest(l):
    """
    The cycles of the homology are written with their smallest sheet
    number first. This function finds the smallest sheet number in the
    cycle l = (sheet, branch point, sheet, branch point, ...)
    """
    a = l[:]

    # If the first element of the cycle is a branch point then just shift
    # the cycle by one.
    if not isinstance(a[0], int):
	a = a[1:] + [a[0]]

    # Return the smallest sheet number appearing in the cycle
    return min([a[2*i] for i in xrange(len(a)/2)])


def reorder_cycle(c, j=None):
    """
    Returns a cycle (as a list) with the element "j" occuring first. If
    "j" isn't provided then assume sorting by the smallest element
    """
    n = len(c)
    try:
	if j != None:
	    i = c.index(j)
	else:
	    sheet = smallest(c)
	    i = c.index(sheet)
    except ValueError:
	raise ValueError("%d does not appear in the cycle %s"%(j,c))

    return [c[k%n] for k in xrange(i,i+n)]



def frobenius_transform(A,g):
    """
    This procedure brings any intersection matrix a to its canonical
    form b by a transformation alpha * a * transpose(alpha)=b. If
    2g=rank(a) and d is the size of the square matrix a, then b has
    d-2g null rows and d-2g null columns. These are moved to the lower
    right corner. On its diagonal, b has 2 gxg null blocks. Above the
    diagonal is a gxg identity block. Below the diagonal is a gxg
    negative identity block. The output of the procedure is the
    transformation matrix alpha.
    """
    if not isinstance(A,numpy.matrix):
	K = numpy.matrix(A, dtype=numpy.int)
    else:
	K = A
    dim = K.shape[0]

    # the rand of an antisymmetric matrix is always even and is equal
    # to 2g in this case
    T = numpy.eye(dim, dtype=numpy.int)

    # create the block below the diagonal. make zeros everywhere else
    # in the first g columns
    for i in xrange(g):
	# make sure column i has a suitable pivot by swapping rows
	# and columns
	counter = dim-1

	while numpy.all( K[(g+i):,i] == numpy.zeros(dim-(g+i)) ):
	    T[[i,counter],:] = T[[counter,i],:]
	    K[:,[i,counter]] = K[:,[counter,i]]
	    K[[i,counter],:] = K[[counter,i],:]
	    counter -= 1

	if K[i+g,i] == 0:
	    # if the pivot element is zero then change rows to make it
	    # non-zero
	    k = i+g+1
	    while K[i+g,i] == 0:
		if K[k,i] != 0:
		    pivot = -1/K[k,i];

		    T[k,:]      *= pivot                         # scale row
		    T[[k,i+g],:] = T[[i+g,k],:]                  # swap rows

		    K[k,:]      *= pivot                         # scale row
		    K[[k,i+g],:] = K[[i+g,k],:]                  # swap rows
		    K[:,k]      *= pivot                         # scale column
		    K[:,[k,i+g]] = K[:,[i+g,k]]                  # swap columns

		k += 1
	else:
	    # otherwise, if the pivot element is non-zero then scale
	    # it so it's equal to -1
	    pivot = -1/K[i+g,i]
	    T[i+g,:] *= pivot
	    K[i+g,:] *= pivot
	    K[:,i+g] *= pivot

	for j in range(i,i+g) + range(i+g+1,dim):
	    # use the pivot to create zeros in the rows above it and below it
	    pivot = -K[j,i]/K[i+g,i]
	    T[j,:] += pivot * T[i+g,:]
	    K[j,:] += pivot * K[i+g,:]
	    K[:,j] += pivot * K[:,i+g]

    for i in xrange(g):
	# the block aboce the diagonal is already there. use it to
	# create zeros everywhere else in teh second block of g
	# columns. automatically all other coluns are then zero,
	# because the rank of the intersection matrix K is only 2g
	for j in range(i+g+1,dim): #XXX check dims
	    pivot = -K[j,i+g]
	    T[j,:] = T[j] + pivot * T[i,:]
	    K[j,:] = K[j,:] + pivot * K[i,:]
	    K[:,j] = K[:,j] + pivot * K[:,i]


    # sanity check: did the Frobenius transform produce the correct
    # result?  T * K * T.T = J where J has the gxg identity I in the
    # top right block and -I in the lower left block (the Jacobian
    # matrix)
    J = numpy.dot(numpy.dot(T, numpy.matrix(A)), T.T)
    for i in xrange(g):
	for j in xrange(g):
	    if j==i+g and i<g:   val = 1
	    elif i==j+g and j<g: val = -1
	    else:                val = 0

	    if J[i,j] != val:
		raise ValueError("Could not compute Frobenuis transform of " + \
                                 "intersection matrix.")
    return T



def tretkoff_graph(hurwitz_system):
    """
    There are two types of nodes:

    - sheets: (integer) these occur on the even levels

    - branch places: (complex, permutation) the first elements is the
    projection of the place in the complex x-plane. the second element
    is a cycle appearing in the monodromy element. (places above a branch
    point are in 1-1 correspondence with the cycles of the permuation) these
    occur on the odd levels
    """
    base_point, base_sheets, branch_points, monodromy, G = hurwitz_system

    # initialize graph with base point: the zero sheet
    C = nx.Graph()
    node = (0,)
    C.add_node(node)
    C.node[node]['value'] = 0
    C.node[node]['label'] = "$0$"
    C.node[node]['final'] = False
    C.node[node]['level'] = 0
    C.node[node]['nrots'] = 0

    # keep track of sheets and branch places that we've already
    # visited. initialize with the zero sheet and all branch places
    # with a stationary cycle (a cycle with one element)
    covering_number = len(base_sheets)
    t = len(branch_points)
    visited_sheets = [0]
    visited_branch_places = [
	(branch_points[i],find_cycle(monodromy[i],j))
	for j in xrange(covering_number)
	for i in xrange(t)
	if len(find_cycle(monodromy[i],j)) == 1
	]

    level = 0
    endpoints = [node]
    final_edges = []
    while len(endpoints) > 0:
	# obtain the endpoints on the previous level that are not
	# final and sort by their "succession" order".
	endpoints = sorted([n for n,d in C.nodes_iter(data=True)
                            if d['level'] == level and not d['final']])

	for node in endpoints:
	    # determine the successors for this node. we use a
	    # different method depending on what level we're on:
	    #
	    # if on an even level (on a sheet): the successors
	    # are branch places. these are the places other than the one
	    # that is the predecessor to this node.
	    #
	    # if on an odd level (on a branch place): the successors are
	    # sheets. these sheets are simply the sheets found in the branch
	    # place whose order is determined by the predecessor sheet.
	    ###################################################################
	    if level % 2 == 0:
		current_sheet = C.node[node]['value']

		# determine which branch points to add. in the initial
		# case, add all branch points. for all subsequent
		# sheets add all branch points other than the one that
		# brought us to this sheet
		if current_sheet == 0:
		    branch_point_indices = range(t)
		else:
                    pred = C.neighbors(node)[0]
		    bpt,pi = C.node[pred]['value']
		    ind = branch_points.index(bpt)
		    branch_point_indices = range(ind+1,t) + range(ind)

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
                        C.node[succ]['value'] = value
                        C.node[succ]['label'] = "$b_{%d},%s$"%(idx,pi)
                        C.node[succ]['final'] = final
                        C.node[succ]['level'] = level+1
                        ctr += 1

	    ###################################################################
	    else:
		current_place = C.node[node]['value']
		bpt,pi = current_place

		# C is always a tree. obtain the previous node (which
		# is the source sheet) since we order cycles with the
		# source sheet appearing first.
		#
		# we also try to minimize the number of rotations performed
		# by allowing reverse rotations.
		n = len(pi)
                pred = C.neighbors(node)[0]
		previous_sheet = C.node[pred]['value']
		pi = reorder_cycle(pi,previous_sheet)
                ctr = 0
		for idx in range(1,n):
                    succ = tuple(list(node) + [ctr])
		    value = pi[idx]
		    edge = (succ,node)

                    if value in visited_sheets:
                        final = True
                    else:
                        final = False
                        visited_sheets.append(value)

                    C.add_edge(node,succ)
                    C.node[succ]['value'] = value
                    C.node[succ]['label'] = "$%d$"%(value)
                    C.node[succ]['final'] = final
                    C.node[succ]['level'] = level+1
                    C.node[succ]['nrots'] = idx if idx <= n/2 else idx-n
                    ctr += 1

	# we are done adding succesors to all endpoints at this
	# level. level up!
	level += 1

    return C



def final_edges(C):
    """
    Returns a list of final edges from the homology graph.

    The final edges are those that define the c-cycles on the Riemann
    surface. Note that the edges returned are such that the nodes of the
    edge are _both_ final nodes.

    The final edges are ordered such that the sheet number appears first
    in the edge.

    Input:

    - homology graph

    Output:

    - list of (ordered) tuples representing the final edges

    """
    final_nodes = [n for n in C.nodes() if C.node[n]['final']]
    edges = []
    while len(final_nodes) > 0:
        node = final_nodes.pop()
        pred = C.neighbors(node)[0]
        pred_val = C.node[pred]['value']
        other = [n for n in final_nodes if C.node[n]['value'] == pred_val and
                 C.node[C.neighbors(n)[0]]['value'] == C.node[node]['value']]
        other = other[0]

        final_nodes.remove(other)
        if isinstance(C.node[node]['value'],tuple):
            edges.append((other,node))
        else:
            edges.append((node,other))

    return edges


def intersection_matrix(final_edges, g):
    """
    Compute the intersection matrix of the c-cycles from the
    Tretkoff graph and final edge data output by `tretkoff_graph()`.

    Input:

    - C: (networkx.Graph) Tretkoff graph

    - final_edges: each edge corresponds to a c-cycle on the Riemann surface

    - g: the expected genus of the riemann surface as given by
      singularities.genus()
    """
    def intersection_number(ei,ej):
	"""
	Returns the intersection number of two edges of the Tretkoff graph.

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
        else:
            return 0

	raise ValueError('Unable to determine intersection index of ' + \
			 'edge %s with edge %s'%(ei,ej))

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
	raise ValueError("Found inconsistent genus in homolgy " + \
			 "intersection matrix.")
    return K


def compute_c_cycles(tretkoff_graph, final_edges):
    """
    Returns the c-cycles of the Riemann surface.

    Input:

    - C: the Tretkoff graph

    - final_edges: a list of the final edges of the Tretkoff graph

    Output:

    A list of the form

	[s_0, (b_{i_0}, n_{i_0}), s_1, (b_{i_1}, n_{i_1}), ...]

    where "s_k" is a sheet number, "b_{i_k}" is the {i_k}'th branch
    point, and "n_{i_k}" is the number of times and direction to go
    about branch point "b_{i_k}".
    """
    root = tuple([0])
    C = tretkoff_graph
    c_cycles = []

    # recall that the edges have a direction: edge[0] is the starting
    # node and edge[1] is the ending node. This determines the
    # direction of the c-cycle.
    for edge in final_edges:
	# obtain the vertices on the Tretkoff graph starting from the
	# base place, going through the edge, and then back to the
	# base_place
        edge = map(lambda n: C.neighbors(n)[0], edge)
	path_to_edge = nx.shortest_path(C,root,edge[0])
	path_from_edge = nx.shortest_path(C,edge[1],root)
        path = path_to_edge + path_from_edge
        path_values = map(lambda n: C.node[n]['value'], path)

        # convert branch places (branch point, permutation) to
        # point-rotations pairs (branch point, number and direction of
        # rotations)
        for n in range(1,len(path),2):
            branch_place = path_values[n]

            # update the path entry (remember, Python uses references to
            # lists) if we are traveling the return path then reverse the
            # rotations.
            if n < len(path_to_edge):
                next_sheet = path[n+1]
                nrots = C.node[next_sheet]['nrots']
            else:
                prev_sheet = path[n-1]
                nrots = - C.node[prev_sheet]['nrots']
            path_values[n] = (branch_place[0], nrots)

        c_cycles.append(path_values)

    return c_cycles



def reverse_cycle(cycle):
    """
    Returns the reversed cycle. Note that rotation numbers around
    branch points are correctly computed.
    """
    rev_cycle = list(reversed(cycle))
    for n in range(1,len(cycle),2):
	rev_cycle[n] = (rev_cycle[n][0], -rev_cycle[n][1])
    return rev_cycle



def compress_cycle(cycle, tretkoff_graph, monodromy_graph):
    """
    Given a cycle, the Tretkoff graph, and the monodromy graph, return a
    shortened equivalent cycle.
    """
    # Compression #1: add rotation numbers of successive cycle
    # elements if the branch points are equal
    N = len(cycle)
    n = 1
    while n < (N-2):
	curr_sheet = cycle[n-1]
	curr_place = cycle[n]
	next_sheet = cycle[n+1]
	next_place = cycle[n+2]

	# if two successive branch points are the same then delete one
	# of them and sum the number of rotations.
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
        sheet = cycle[n]
        branch = cycle[n+1]

        if branch[1] == 0:
            cycle.pop(n)
            cycle.pop(n)
            N -= 2
        else:
            n += 2

    return cycle



def compute_ab_cycles(c_cycles, linear_combinations, g,
		      tretkoff_graph, monodromy_graph):
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
	a = compress_cycle(a, tretkoff_graph, monodromy_graph)
	b = compress_cycle(b, tretkoff_graph, monodromy_graph)

	a_cycles.append(a)
	b_cycles.append(b)

    return a_cycles, b_cycles


@cached_function
def homology(f,x,y,base_point=None,base_sheets=None,verbose=True):
    """
    Given a plane representation of a Riemann surface, that is, a
    complex plane algebraic curve, return a canonical basis for the
    homology of the Riemann surface.

    Inputs:

    - f,x,y: an algebraic curve in x and y.

    - base_point: (optional) the base point of the monodromy group lying
      in the complex x-plane

    - base_sheets: (optional) perscribed ordering of the y-sheets lying
    above the base_point

    - verbose: (optional) return a dictionary containing the following
      information

      * 'basepoint' : the base point used in the monodromy group

      * 'basesheets' : the base sheets (order matters) lying above the
        base point

      * 'c-cycles' : a list of intermediate cycles. linear combinations
        of these make up the a- and b-cycles

      * 'linearcombination' : a (2g x m) integer matrix, where g is the
        genus of the curve f = f(x,y). the first g rows show which
        linear combination of the c-cycles produce the a-cycles. the
        second g rows give the b-cycles.

      * 'a-cycles' : a list of the a-cycles

      * 'b-cycles' : a list of the b-cycles

    Outputs:

    """
    g = int(genus(f,x,y))
    hurwitz_system = monodromy(f,x,y,base_point=base_point,
                               base_sheets=base_sheets)
    base_point, base_sheets, branch_points, mon, G = hurwitz_system

    # compute primary data elements:
    #
    # * tretkoff_graph gives us the key combinatorial data
    #
    # * intersection_matrix takes this data and tells us how
    #   the c-cycles intersect
    #
    # * the frobenius_transform of the intersection matrix
    #   gives us which linear combinations of the c_cycles we
    #   need to obtain the a- and b-cycles
    C = tretkoff_graph(hurwitz_system)
    edges = final_edges(C)
    K = intersection_matrix(edges,g)
    T = frobenius_transform(K,g)

    c_cycles = compute_c_cycles(C, edges)
    a_cycles, b_cycles = compute_ab_cycles(c_cycles, T, g, C, G)

    if verbose:
        return {
            'basepoint': base_point,
            'basesheets': base_sheets,
            'c-cycles': c_cycles,
            'linearcombinations': T[:2*g,:],
            'a-cycles': a_cycles,
            'b-cycles': b_cycles,
            }
    else:
        return a_cycles, b_cycles


def show_homology(C):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except:
        raise

    final_nodes = [n for n,d in C.nodes(data=True) if d['final']]
    final_edges = [e for e in C.edges()
                   if e[0] in final_nodes or e[1] in final_nodes]
    edges = [e for e in C.edges() if e not in final_edges]

    # compute positions
    pos = {(0,):(0,0)}
    labels = {(0,):"$0$"}
    level = 0
    level_points = [(0,)]
    while len(level_points) > 0:
        level += 1
        level_points = sorted([n for n,d in C.nodes(data=True)
                               if d['level'] == level])

        num_level_points = len(level_points)
        n = 0
        for point in level_points:
            pos[point] = (level,n-num_level_points/2.0)
            labels[point] = C.node[point]['label'] + \
                '\n $' + str(point) + '$'
            n += 1

    # draw it
    nx.draw_networkx_nodes(C, pos, node_color='w')
    nx.draw_networkx_nodes(C, pos, nodelist=final_nodes, node_color='w')
    nx.draw_networkx_edges(C, pos, edgelist=edges)
    nx.draw_networkx_edges(C, pos, edgelist=final_edges,
                           edge_color='b', style='dashed')
    nx.draw_networkx_labels(C, pos, labels=labels, font_size=13)

    plt.xticks([])
    plt.yticks([])
    plt.show()



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
    f11= y**2 - (x**2+1)*(x**2-1)*(4*x**2+1)  # simple genus two hyperelliptic
    f12 = x**4 + y**4 - 1
    f13 = y**2 - (x-2)*(x-1)*(x+1)*(x+2)  # simple genus one hyperelliptic

    f = f2

    print("\nComputing monodromy...")
    # f2
    I = 1.0j
    base_point = -1.44838920232100
    base_sheets = [-3.20203812255,
                    1.60101906127-1.26997391750*I,
                    1.60101906127+1.26997391750*I]

    hs = monodromy(f,x,y,base_point=base_point,base_sheets=base_sheets)
    print("\nComputing genus...")
    g = int(genus(f,x,y))
    print("...genus = %d"%g)

    print("\nBranch points...")
    for bpt in hs[2]: print bpt

    print("\nMonodromy Group...")
    for mon in hs[-2]: print mon

    print("\nComputing Tretkoff Graph...")
    C = tretkoff_graph(hs)

    print("\nComputing final edges...")
    e = map(lambda (ei,ej): (C.node[ei]['value'],C.node[ej]['value']),
            final_edges(C))
    for ei in e: print ei

    show_homology(C)

    print("\nComputing intersection matrix and lincombs...")
    edges = final_edges(C)
    K = intersection_matrix(edges, g)
    T = frobenius_transform(K,g)
    J = numpy.dot(numpy.dot(T,K),T.T)
    print("K =\n%s\n\nT =\n%s\n\nJ =\n%s"%(K,T,J))

    print("\nComputing c-cycles...")
    c_cycles = compute_c_cycles(C, edges)
    for c in c_cycles: print c

    print("\nComputing a- and b-cycles")
    a,b = compute_ab_cycles(c_cycles, T, g, C, None)
    print("a-cycles:")
    for ai in a: print ai
    print("b-cycles:")
    for bi in b: print bi
