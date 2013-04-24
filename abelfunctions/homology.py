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
    form by a transformation alpha * a * transpose(alpha)=b. If
    2g=rank(a) and d is the size of the square matrix a, then b has
    d-2g null rows and d-2g null columns. These are moved to the lower
    right corner. On its diagonal, b has 2 gxg null blocks. Above the
    diagonal is a gxg identity block. Below the diagonal is a gxg
    -identity block. The output of the procedure is the transformation
    matrix alpha.
    """
    if not isinstance(A,numpy.matrix):
        B = numpy.matrix(A, dtype=numpy.int)
    else:
        B = A
    dim = B.shape[0]

    # the rand of an antisymmetric matrix is always even and is equal
    # to 2g in this case
    alpha = numpy.eye(dim, dtype=numpy.int)
    
    # create the block below the diagonal. make zeros everywhere else
    # in the first g columns
    for i in xrange(g):
        # make sure column i has a suitable pivot by swapping rows
        # and columns
        counter = dim-1

        while numpy.all( B[(g+i):,i] == numpy.zeros(dim-(g+i)) ):
            alpha[[i,counter],:] = alpha[[counter,i],:]
            B[:,[i,counter]]     = B[:,[counter,i]]
            B[[i,counter],:]     = B[[counter,i],:]
            counter -= 1
        
        if B[i+g,i] == 0:
            # if the pivot element is zero then change rows to make it
            # non-zero
            k = i+g+1
            while B[i+g,i] == 0:
                if B[k,i] != 0:
                    pivot = -1/B[k,i];

                    alpha[k,:]      *= pivot                     # scale row
                    alpha[[k,i+g],:] = alpha[[i+g,k],:]          # swap rows

                    B[k,:]      *= pivot                         # scale row
                    B[[k,i+g],:] = B[[i+g,k],:]                  # swap rows
                    B[:,k]      *= pivot                         # scale column
                    B[:,[k,i+g]] = B[:,[i+g,k]]                  # swap columns
                    
                k += 1
        else:
            # otherwise, if the pivot element is non-zero then scale
            # it so it's equal to -1
            pivot = -1/B[i+g,i]
            alpha[i+g,:] *= pivot
            B[i+g,:]     *= pivot
            B[:,i+g]     *= pivot

        for j in range(i,i+g) + range(i+g+1,dim):
            # use the pivot to create zeros in the rows above it and below it
            pivot = -B[j,i]/B[i+g,i]
            alpha[j,:] += pivot * alpha[i+g,:]
            B[j,:]     += pivot * B[i+g,:]
            B[:,j]     += pivot * B[:,i+g]

    for i in xrange(g):
        # the block aboce the diagonal is already there. use it to
        # create zeros everywhere else in teh second block of g
        # columns. automatically all other coluns are then zero,
        # because the rank of B is only 2g
        for j in range(i+g+1,dim): #XXX check dims
            pivot = -B[j,i+g]
            alpha[j,:] = alpha[j] + pivot * alpha[i,:]
            B[j,:]     = B[j,:] + pivot * B[i,:]
            B[:,j]     = B[:,j] + pivot * B[:,i]

    return alpha


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
    C.add_node(0)
    C.node[0]['final'] = False
    C.node[0]['label'] = '$%d$'%(0)
    C.node[0]['level'] = 0
    C.node[0]['order'] = [0]

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
    endpoints = [0]
    final_edges = []
    while len(endpoints) > 0:
        # obtain the endpoints on the previous level that are not
        # final and sort by their "succession" order".
        endpoints = sorted([n for n,d in C.nodes_iter(data=True) 
                     if d['level'] == level],
                     key=lambda n: C.node[n]['order'][-1])
        
        order_counter = 0
        
        # print "level =", level
        # print "endpoints ="
        # print endpoints

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
                current_sheet = node

                # determine which branch points to add. in the initial
                # case, add all branch points. for all subsequent
                # sheets add all branch points other than the one that
                # brought us to this sheet
                if current_sheet == 0:
                    branch_point_indices = range(t)
                else:
                    bpt,pi = C.neighbors(current_sheet)[0]
                    ind = branch_points.index(bpt)
                    branch_point_indices = range(ind+1,t) + range(ind)

                # for each branch place connecting the curent sheet to other
                # sheets, add a final edge if we've already visited the place
                # or connect it to the graph, otherwise.
                for idx in branch_point_indices:
                    bpt = branch_points[idx]
                    pi = find_cycle(monodromy[idx],current_sheet)
                    succ = (bpt,pi)
                    edge = (node,succ) # final edges point from sheets to bpts
                    
                    # determine whether or not this is a successor or a
                    # "final" vertex
                    if succ in visited_branch_places:
                        if edge not in final_edges and len(pi) > 1:
                            final_edges.append(edge)
                    elif len(pi) > 0:
                        visited_branch_places.append(succ)
                        C.add_edge(node,succ)
                        C.node[succ]['label'] = '$b_%d, %s$'%(idx,pi)
                        C.node[succ]['level'] = level+1
                        C.node[succ]['nrots'] = None
                        C.node[succ]['order'] = C.node[node]['order'] + \
                                                [order_counter]
                                                
                    # the counter is over all succesors of all current
                    # sheets at the current level (as opposed to just
                    # successors of this sheet)
                    order_counter += 1

            ###################################################################
            else:
                current_place = node
                bpt,pi = current_place

                # C is always a tree. obtain the previous node (which
                # is the source sheet) since we order cycles with the
                # source sheet appearing first.
                #
                # we also try to minimize the number of rotations performed
                # by allowing reverse rotations.
                n = len(pi)
                previous_sheet = C.neighbors(current_place)[0]
                pi = reorder_cycle(pi,previous_sheet)
                                
                for idx in range(1,n):
                    next_sheet = pi[idx]
                    succ = next_sheet
                    edge = (succ,node) # final edges point from sheets to bpts

                    if next_sheet in visited_sheets:
                        if edge not in final_edges:
                            final_edges.append(edge)
                    else:
                        visited_sheets.append(next_sheet)
                        C.add_edge(succ,node)
                        C.node[succ]['label'] = '$%d$'%(next_sheet)
                        C.node[succ]['level'] = level+1
                        C.node[succ]['nrots'] = idx if idx < n/2 else n-idx
                        C.node[succ]['order'] = C.node[node]['order'] + \
                                                [order_counter]
                                                
                    # the counter is over all succesors of all current
                    # branch places at the current level (as opposed
                    # to just successors of this branch place)
                    order_counter += 1

        # we are done adding succesors to all endpoints at this
        # level. level up!
        level += 1

    # the tretkoff graph is constructed. return the final edge. we
    # also return the graph since it contains ordering and
    # rotational data
    return C, final_edges


def intersection_matrix(C, final_edges):
    """
    Compute the intersection matrix of the c-cycles from the
    Tretkoff graph and final edge data output by `tretkoff_graph()`.

    Input:

    - C: (networkx.Graph) Tretkoff graph

    - final_edges: each edge corresponds to a c-cycle on the Riemann surface
    """
    def intersection_number(ei,ej):
        """
        Returns the intersection number of two edges of the Tretkoff graph.

        Note: Python is smart and uses lexicographical ordering on lists
        which is exactly what we need.
        """
        ei_start,ei_end = map(lambda n: C.node[n]['order'], ei)
        ej_start,ej_end = map(lambda n: C.node[n]['order'], ej)

        # if the starting node of ei lies before the starting node of ej
        # then simply return the negation of (ej o ei)
        if ei_start > ej_start:
            return (-1)*intersection_number(ej,ei)
        # otherwise, we need to check the relative ordering of the
        # ending nodes of the edges with the starting nodes.
        else:
            if ej_start < ei_end < ej_end:
                return 1
            elif (ej_end < ei_end < ej_start) or (ej_start < ei_start <ej_end):
                return -1
            else:
                return 0
            
        raise ValueError('Unable to determine intersection index of ' + \
                         'edge %s with edge %s'%(ei,ej))


    # the intersection matrix is anti-symmetric, so we only determine
    # the intersection numbers of the upper triangle
    num_final_edges = len(final_edges)
    K = numpy.zeros((num_final_edges, num_final_edge), dtype=numpy.int)
    for i in range(num_final_edges):
        ei = final_edges[i]
        for j in range(i+1,num_final_edges):
            ej = final_edges[j]
            K[i,j] = intersection_number(ei,ej)

    # obtain the intersection numbers below the diagonal
    K = K - K.T
    return K


def create_cycles_from_final_edges(C, final_edges):
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
    c_cycles = []

    # recall that the edges have a direction: edge[0] is the starting
    # node and edge[1] is the ending node. This determines the
    # direction of the c-cycle.
    for edge in final_edges:
        # obtain the vertices on the Tretkoff graph starting from the
        # base place, going through the edge, and then back to the
        # base_place
        path_to_edge = nx.shortest_path(C,0,edge[0])
        path_from_edge = nx.shortest_path(C,edge[1],0)
        path = path_to_edge + edge + path_from_edge

        # the path information is currently of the form:
        #
        # [0, .., s_j, (b_{i_j}, pi_{i_j}), ...]
        #
        # (each odd element is a branch place - permutation pair.
        # Use the
        c_cycles.append(path)

    return c_cycles



def canonical_basis(f,x,y):
    """
    Given a plane representation of a Riemann surface, that is, a
    complex plane algebraic curve, return a canonical basis for the
    homology of the Riemann surface.
    """
    g = int(genus(f,x,y))
    hurwitz_system = monodromy(f,x,y)
    base_point, base_sheets, branch_points, mon, G = hurwitz_system

    # compute key data elements
    # - t_table: path data from Tretkoff graph
    # - t_basis: a collection of cycles generated from the Tretkoff table.
    #            the homology cycles are formed by a linear comb of these
    # - t_list:  a ordering of the pi/qi symbols used to dermine the
    #            intersection matric of the t_basis cycles
    t_table = tretkoff_table(hurwitz_system)
    t_basis = homology_basis(t_table)
    t_list  = tretkoff_list(t_table)
    
    c = len(t_list) / 2
    t_matrix = intersection_matrix(t_list, 
                                   [t_table['p'][i] for i in xrange(c)] + \
                                   [t_table['q'][i] for i in xrange(c)])

    # sanity check: make sure intersection matrix produces the same genus
    rank = numpy.linalg.matrix_rank(t_matrix)
    if rank/2 != g:
        raise ValueError("Found inconsistent genus in homolgy " + \
                         "intersection matrix.")
    
    alpha = frobenius_transform(t_matrix,g)

    # sanity check: did the Frobenius transform produce the correct result?
    # alpha * t_matrix * alpha.T = J where J has the gxg identity I in the
    # top right block and -I in the lower left block
    #
    # XXX move this code to frobenius_transform???
    t_matrix_check = numpy.dot(numpy.dot(alpha, t_matrix), alpha.T)
    for i in xrange(c):
        for j in xrange(c):
            if j==i+g and i<g:   val = 1
            elif i==j+g and j<g: val = -1
            else:                val = 0

            if t_matrix_check[i,j] != val:
                raise Error("Could not compute Frobenuis transform of " + \
                            "intersection matrix.")

    # place results in a dictionary
    c = {}
    c['basepoint'] = base_point
    c['sheets'] = base_sheets
    c['genus'] = g
    c['cycles'] = map(reform_cycle, t_basis)
    c['linearcombination'] = alpha[:2*g,:]

    return c



def homology(*args, **kwds):
    return canonical_basis(*args, **kwds)


def plot_homology(C,final_edges):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except:
        raise

    edges = C.edges()
    labels = dict([(n,d['label']) for n,d in C.nodes(data=True)])

    # compute positions
    pos = {0:(0,0)}
    level = 1
    prev_points = [0]
    level_points = [0]
    N_prev = 1
    while len(level_points) > 0:
        level_points = sorted([n for n,d in C.nodes(data=True)
                               if d['level'] == level],
                               key = lambda n: C.node[n]['order'])
        
        N = len(level_points)
        for k in range(N):
            node = level_points[k]
            pred = [p for p in C.neighbors(node)
                    if C.node[p]['level'] < level][0]

            # complex position distributed evenly about unit circle
            theta = numpy.double(k)/N
            z = numpy.exp(1.0j*numpy.pi*theta)

            # cluster by predecessor location

            # scale by level
            z *= level
            
            pos[node] = (z.real, z.imag)

        level += 1
        N_prev = N
        prev_points = level_points[:]
            

    # draw it
    nx.draw_networkx_nodes(C, pos)
    nx.draw_networkx_edges(C, pos, edgelist=edges, width=2)
    nx.draw_networkx_edges(C, pos, edgelist=final_edges,
                           edge_color='b', style='dashed')
    nx.draw_networkx_labels(C, pos, labels=labels, font_size=16)
    
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


    f = f12
    hs = monodromy(f,x,y)
    C, final_edges = tretkoff_graph(hs)
    labels = dict((n,C.node[n]['label']) for n in C.nodes())

    plot_homology(C,final_edges)


