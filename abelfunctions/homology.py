"""
Homology
"""

import numpy
import scipy
import sympy
import networkx as nx

from monodromy import Permutation, Monodromy

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

    cycle = [j]
    k = pi(j)
    while k != j:
        cycle.append(k) # update the cycle
        k = pi(k)       # iterate

    return reorder_cycle(tuple(cycle), min(cycle))


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
    "j" isn't provided then assume
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

    return tuple([c[k%n] for k in xrange(i,i+n)])
        


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

    XXX not well tested....XXX
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
        while numpy.all(B[(g+i-1):,i] == numpy.zeros(dim-(i+g-1))):

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

        for j in range(i,i+g) + range(i+g+1,dim):  # XXX double-check indices
            # use the pivot to create zeros in the rows above it and below it
            pivot = -B[j,i]/B[i+g,i]
            alpha[j,:] = alpha[j,:] + pivot * alpha[i+g,:]
            B[j,:] = B[j,:] + pivot * B[i+g,:]
            B[:,j] = B[:,j] + pivot * B[:,i+g]

    for i in range(g):
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
        
    


# def alt_tretkoff_table(hutwitz_system):
#     base_point, base_sheets, branch_points, monodromy = hurwitz_system

#     # the number of sheets of the Riemann surface
#     covering_number = len(base_sheets)

#     # the number of branch points
#     t = len(branch_points)

#     # the branch points together with their permutations, using the
#     # notation of Tretkoff and Tretkoff
#     c = [[ (branch_points[i], find_cycle(monodromy[i],j)) 
#            for j in xrange(covering_number) ]
#          for i in xrange(t) ]
    
#     # initial construction of first circle: from sheet one go to all
#     # branchpoints on sheet one which are not fixed points
#     G = nx.DiGraph()
#     G.add_node(0)
#     G.add_edges_from( [(0,c[i][0]) for i in xrange(t)] )
    
#     # vitied_labels keeps track of which sheets we've already visited
#     used_sheet_labels = [0]
    
#     # used_bpt_labels keepst rack of which preimages of branch points
#     # (cycles) we've used already. this includes one cycles, which are
#     # terminal points
#     used_bpt_labels = [c[i][0] for i in xrange(t)]
#     for i in xrange(t):
#         for j in xrange(j):
#             if len(c[i][j][1]) == 1: 
#                 used_bpt_labels.append(c[i][j])


#     # loop until all sheets have been reached using a breadth first search
#     level == 0
#     while len(used_sheet_labels) < t:
#         # grab all of the nodes that we haven't computed successors to
#         # that haven't already been visited in the past
#         if level % 2:
#             current_level_nodes = [n for n in G.nodes if G.successors(n) == []
#                                    and n not in used_bpt_labels]
#         else:
#             current_level_nodes = [n for n in G.nodes if G.successors(n) == []
#                                    and n not in used_sheet_labels]

        
        


    


def tretkoff_table(hurwitz_system):
    """
    Encodes data from a given Hurwitz system into a graph that
    represents the corresponding Riemann Surface. 

    A spanning tree of this graph produces a finite set of cycles
    which contains a basis for the homology of the Riemann
    surface. The particular way in which this is done allows the
    determination of the intersection numbers of the cycles.

    Input:
    - a Hurwitz system
    
    Output:
    - a Tretkoff table encoding the c-cycle data.
    """
    base_point, base_sheets, branch_points, monodromy = hurwitz_system
    
    # XXX USE BRANCH POINT INDICES INSTEAD FOR EASE OF READNIG XXX
    branch_points = range(len(branch_points))

    # the number of sheets of the Riemann surface
    covering_number = len(base_sheets)

    # the number of branch points
    t = len(branch_points)

    # the branch points together with their permutations, using the
    # notation of Tretkoff and Tretkoff
    c = [[ (branch_points[i], find_cycle(monodromy[i],j)) 
           for j in xrange(covering_number) ]
         for i in xrange(t) ]
    
    # initial construction of first circle: from sheet one go to all
    # branchpoints on sheet one which are not fixed points
    finished = False
    startseq = []
    for i in xrange(t):
        if len(c[i][0][1]) != 1:
            startseq.append(c[i][0])
        else:
            startseq.append(['stop',c[i][0]])
    
    # initialize Tretkoff table
    tretkoff = {'C':{0:[[0,[],startseq,[]]]}, 'q':{}, 'p':{}}
    
    # vitied_labels keeps track of which sheets we've already visited
    used_sheet_labels = [0]
    
    # used_bpt_labels keepst rack of which preimages of branch points
    # (cycles) we've used already. this includes one cycles, which are
    # terminal points
    used_bpt_labels = [c[i][0] for i in xrange(t)]
    for i in xrange(t):
        for j in xrange(covering_number):
            if len(c[i][j][1]) == 1: 
                used_bpt_labels.append(c[i][j])


    # Main Loop: 'level' keeps track of which level from the root C0
    # we are on in the graph. 'final_edges' holds the final edges in
    # the graph, as in the notation of [TT]. 'q_edges'....XXXX.
    # 'q_counter' keeps track of the final points
    level = 1
    final_edges = []
    q_edges = []
    q_counter = 0
    while not finished:
        finished = True
        # previous = number of branches from the previous level to the
        # current level
        previous = len(tretkoff['C'][level-1])
        
        # odd levels correspond to sheets
        if (level % 2):
            # number of vertices at this level equals the number of
            # branches originating at the previous level and pointing
            # to this level
            tretkoff['C'][level] = [[] for _ in xrange(previous)]
            for i in xrange(previous):
                # it's possible that a vertex at the previous level
                # doesn't point to this level. This occurs in the case
                # when the vertex is a final vertex. In that case,
                # nothing corresponds to that vertex at this level.
                if tretkoff['C'][level-1][i][2] != []:
                    finished = False
                    # sourcelist = vertex at previous level
                    sourcelist = tretkoff['C'][level-1][i]
                    entry = [[] for _ in xrange(len(sourcelist[2]))]

                    # construct a new vertex at thsi level for every
                    # branch leaving the vertex at the previous level
                    for j in xrange(len(sourcelist[2])):
                        # a1 = new vertex
                        # a2 = originating vertex
                        # a3 = branches to future vertices
                        # a4 = label in the graph
                        a1 = sourcelist[2][j]
                        a2 = sourcelist[0]
                        a3 = []
                        a4 = sourcelist[3] + [j]

                        if a1[0] != 'stop':
                            # (order is important)
                            newlist = reorder_cycle(sourcelist[2][j][1],
                                                    sourcelist[0])

                            # for each label in the cycle from the
                            # originating vertex, (which was moved to
                            # the front of the cycle) check if we need
                            # to add an edge to the graph.
                            for k in range(1,len(newlist)):
                                nlk = newlist[k]
                                # sheet has not been visited. add it
                                # to the min spanning tree
                                if nlk not in used_sheet_labels:
                                    a3.append(nlk)
                                    used_sheet_labels.append(nlk)
                                # sheet has been visited. this means
                                # that we would be adding an edge that
                                # we would otherwise take out of the
                                # min spanning tree. determine what
                                # type of edge it is
                                else:
                                    a3.append(['stop',nlk])
                                    # 'edge' denotes a branch pointing
                                    # at an endpoint of the graph
                                    edge = [nlk, a1]
                                    if edge not in final_edges:
                                        # it's the first time this
                                        # edge occurs: add it to the
                                        # list of final edges and mark
                                        # it by a q-endpoint
                                        final_edges.append(edge)
                                        q_edges.append(edge)
                                        tretkoff['q'][q_counter] = [
                                            ['stop',nlk],
                                            a1,
                                            [],
                                            a4 + [k-1],
                                        ]
                                        q_counter += 1
                                    else:
                                        # this branch has occured
                                        # before. Find out where and
                                        # give it the corresponding
                                        # p-endpoint. If this is
                                        # possible it only occured
                                        # because of a final edge
                                        # which does not lead to a
                                        # cycle
                                        if edge in q_edges:
                                            p_counter = q_edges.index(edge)
                                            tretkoff['p'][p_counter] = [
                                                a1,
                                                a2,
                                                a3,
                                                a4 + [k-1],
                                            ]
                                        #endif edge in q_edges
                                    #endif edge not in final_edges
                                #endif newlist[k] not in used_sheets labels
                            #endfor k in range(1,len(newlist))
                        #endif a1[0] != 'stop'

                        # create the new vertex
                        entry[j] = [a1,a2,a3,a4]

                    #endfor j in range(len(sourcelist)

                    # add / replace the new vertex to the graph
                    tretkoff['C'][level][i] = entry

                #endif tretkoff['C'][level-1][i][2] != []:
            #endfor i in xrange(previous)
        #endif (level % 2)
        else:
            # even levels correspond to pre-images of branch points:
            # number of vertices at this level equals the number of
            # branches originating at the previous level and point to
            # this level
            tretkoff['C'][level] = [[] for _ in xrange(previous)]
            for i in xrange(previous):
                # it's possible that a vertex at the previous level
                # doesn't point to this level, i.e. it is a final
                # vertex. in that case, nothing corresponds to that
                # vetex at this level.
                if tretkoff['C'][level-1][i][2] != []:
                    finished = False
                    # sourcelist = vertex at previous level
                    sourcelist = tretkoff['C'][level-1][i]
                    # construct a new vertex at this level for every
                    # branch leaving the vertex at the previous level
                    entry = [[] for _ in xrange(len(sourcelist[2]))]
                    for j in xrange(len(sourcelist[2])):
                        # b1 = new vertex
                        # b2 = originating vertex
                        # b3 = branches for future 
                        # b4 = level in the graph. which path to
                        #      follow starting form the root to 
                        #      get here
                        b1 = sourcelist[2][j]
                        b2 = sourcelist[0]
                        b3 = []
                        b4 = sourcelist[3] + [j]
                        
                        # Note: the order in whihc the sheets are
                        # visited is important. it is obviously given
                        # by the monodromy permutation related to each
                        # branch oint. as a consequence, the following
                        # is split into two parts which need to be
                        # done in order: first the sheets that are
                        # next in the permutation, then the sheets in
                        # the permutation preceeding the current one
#                        if b1[0] != 'stop':
                        if not isinstance(b1,list):
                            startingindex = branch_points.index(b2[0])
                            for k in range(startingindex+1,t):
                                ckb1 = c[k][b1]
                                if ckb1 not in used_bpt_labels:
                                    # the preimage of the branchpoint
                                    # has not been used. add it.
                                    if len(ckb1[1]) != 1:
                                        b3.append(ckb1)
                                        used_bpt_labels.append(ckb1)
                                else:
                                    # the preimage of the branchpoint
                                    # has been used.
                                    b3.append(['stop',ckb1])
                                    # 'edge' denotes a branch pointing
                                    # at an endpoint on the graph
                                    edge = [b1,ckb1]
                                    if edge not in final_edges:
                                        # it's the first time this
                                        # edge occurs: add it tothe
                                        # list of final edges and mark
                                        # it by a q-endpoint
                                        final_edges.append(edge)
                                        if len(ckb1[1]) != 1:
                                            tretkoff['q'][q_counter] = [
                                                ['stop',ckb1],
                                                b1,
                                                [],
                                                b4 + [k-startingindex],
                                            ]
                                            q_edges.append(edge)
                                            q_counter += 1
                                    else:
                                        # this branch has occured
                                        # before. find out where and
                                        # give it the corresponding
                                        # p-endpoint. if this is
                                        # possible, it only occured
                                        # because of a final edge
                                        # which does not lead to a
                                        # cycle
                                        if edge in q_edges:
                                            p_counter = q_edges.index(edge)
                                            tretkoff['p'][p_counter] = [
                                                b1,
                                                b2,
                                                b3,
                                                b4 + [k-startingindex],
                                            ]
                                        #fi
                                    #fi
                                #fi
                            #od
                            for k in xrange(startingindex):
                                ckb1 = c[k][b1]
                                if ckb1 not in used_bpt_labels:
                                    # the preimage of the branchpoint
                                    # has not been used
                                    if len(ckb1[1]) != 1:
                                        b3.append(ckb1)
                                        used_bpt_labels.append(ckb1)
                                else:
                                    # the primeage of the branchpoint
                                    # has been used
                                    b3.append(['stop', ckb1])
                                    # 'edge' denotes a branch pointing
                                    # at an endpoint of the graph
                                    edge = [b1, ckb1]
                                    if edge not in final_edges:
                                        # it's the first time this
                                        # edge occurs: add it to the
                                        # list of final edges and mark
                                        # it by a q-endpoint
                                        final_edges.append(edge)
                                        if len(ckb1[1]) != 1:
                                            tretkoff['q'][q_counter] = [
                                                ['stop',ckb1],
                                                b1,
                                                [],
                                                b4 + [k+t-startingindex],
                                            ]
                                    else:
                                        # this branch has occured
                                        # before. find out where and
                                        # give it the corresponding
                                        # p-endpoint. If is possible,
                                        # it only occured becuase of a
                                        # final edge which does not
                                        # lead to a cycle
                                        if edge in q_edges:
                                            p_counter = q_edges.index(edge)
                                            tretkoff['p'][p_counter] = [
                                                b1,
                                                b2,
                                                b3,
                                                b4 + [k+t-startingindex],
                                            ]
                                        #end if edge in q_edge
                                    #end if edge not in final_edges
                                #end if ckb1 not in used_bpt_labels
                            #od
                        #fi
                        # create a new vertex
                        entry[j] = [a1,a2,a3,a4]
                    #od
                    # add the new vertex to the graph
                    tretkoff['C'][level][i] = entry
                #fi 
            #od 
        #fi 
        # don't bunch the vertices together according
        # to their origin. All vertices are treated equal.
        # i.e. flatten the list
        tretkoff['C'][level] = [vertex for vertex_list in tretkoff['C'][level]
                                for vertex in vertex_list]
        level += 1
    #od
    # How many levels with new information are there? This excludes the last
    # level which contains only final points.
    tretkoff['depth'] = level - 2
    
    # How many cycles are generated by the spanning tree?
    tretkoff['numberofcycles'] = q_counter - 1
    return tretkoff



def tretkoff_list(tretkoff_table):
    """
    Determines a sequence of p_i and q_i symbols which are used later
    to determine the intersection indices of the cycles of the
    homology.
    """
    def cmp(l1,l2):
        """l1 < l2 if the words formed from ..."""
        j1 = len(l1)
        j2 = len(l2)
        if j1 <= j2:
            i = 0
            while i <= j1:
                if l1[i] != l2[i]:
                    return (l1[i] < l2[i])  # check this since cmp \in {-1,0,1}
                i += 1
        else:
            return not cmp(l2,l1)
    
    n = tretkoff_table['numberofcycles']
    result = []

    lijst = [ tretkoff_table['p'][i][3] for i in xrange(n) ]
    lijst.extend( [ tretkoff_table['q'][i][3] for i in xrange(n) ] )
    
    # check this since cmp \in {-1,0,1}
    lijst.sort(cmp=cmp)
    for e in lijst:
        j = lijst.index(e)
        # result:=result,`if`(j>n,q[j-n],p[j])
        result.append(tretkoff_table['q'][j-n] if j >= n   #XXX
                      else tretkoff_table['p'][j])

    return result



def make_cycle(a,b):
    """
    This procedure removes the common parts of two lists before
    putting them together to create a cycle.
    """
    H = -1
    while a[H+1] == b[H+1]: H += 1

    A = [a[i] for i in xrange(H,len(a))]
    B = [b[i] for i in range(H+1,len(b)-1)]
    B.reverse()    
    A.extend(B)
    cycle = reorder_cycle(A)

    return cycle


def intersection_matrix(lijlist, elements):
    """
    Computes the intersection matrix, K, of the c-cycles given the pi/qi
    points. The ordering of the pi's and qi's determines the entry in the
    intersection matrix: -1, 0, or 1.
    """
    length = len(lijlist)
    dim    = length / 2
    K      = numpy.zeros((dim,dim), dtype=numpy.int)
    a      = [lij for lij in lijlist]

    for i in xrange(dim-1):
        a = reorder_cycle(a,elements[i])
        qi = a.index(elements[i+dim])

        for j in range(i+1,dim):
            pj = a.index(elements[j])
            qj = a.index(elements[j+dim])
            
            if   (pj<qi) and (qi<qj): K[i,j] = 1
            elif (qj<qi) and (qi<pj): K[i,j] = -1
            else:                     K[i,j] = 0

    return K - K.T



def homology_basis(tretkoff_table):
    """
    This procedure does not really determine a basis for the
    homology. It determines a finite set containing a basis. Some
    elements in the set may be dependent in the homology, however.

    The cycle is found by following the path that leads to qi from the
    root. Then we follow the path from the root to pi. These paths are
    pasted together and their overlap around the root is removed.
    """
    c = []
    for i in xrange(tretkoff_table['numberofcycles']):
        pi = tretkoff_table['p'][i]
        qi = tretkoff_table['q'][i]
        ppath = pi[3][:-1]
        qpath = qi[3]

        for Z in range(2):
            part = [0]  # XXX
            k = 0
            vertex = tretkoff_table['C'][0][0]
            for j in ppath if Z == 0 else qpath:
                part.append(vertex[2][j])
                loc1 = tretkoff_table['C'][k].index(vertex)
                loc2 = 0
                
                if loc1 > 0:                  # XXX
                    for m in range(loc1):     # XXX
                        loc2 += len(tretkoff_table['C'][k][m][2])

                loc2 += j
                k += 1
                vertex = tretkoff_table['C'][k][loc2]

            if Z == 0:
                ppart = part
            else:
                qpart = part

        # By construction, the last element of qpart should be a ['stop',sheet]
        # tuple. Replace this tuple with the sheet number it contains since
        # we don't need to keep track of 'stop's anymore.
        qpart[-1] = qpart[-1][1]
        c.append( make_cycle(ppart, qpart) )

    return c




def canonical_basis(f,x,y):
    """
    Given a plane representation of a Riemann surface, that is, a
    complex plane algebraic curve, return a canonical basis for the
    homology of the Riemann surface.
    """
#    g = genus(f,x,y)   # XXX write this function...
    mon = Monodromy(f,x,y)
    hurwitz_system = mon.hurwitz_system()

    # compute key data elements
    # - t_table: path data from Tretkoff graph
    # - t_basis: a collection of cycles generated from the Tretkoff table.
    #            the homology cycles are formed by a linear comb of theseXS
    # - t_list:  a ordering of the pi/qi symbols used to dermine the
    #            intersection matric of the t_basis cycles
    t_table = tretkoff_table(hurwitz_system)
    t_basis = homology_basis(t_table)
    t_list  = tretkoff_list(t_table)
    
    c = len(t_list) / 2
    t_matrix = intersection_matrix(t_list, 
                                   [t_table['p'][i] for i in xrange(c)] + \
                                   [t_table['q'][i] for i in xrange(c)])

    pdb.set_trace()

    # sanity check: make sure intersection matrix produces the same genus
    rank = numpy.linalg.matrix_rank(t_matrix)
    g = rank/2 # XXX See comment below
#    if rank/2 != g:  # XXX genus
#        raise ValueError("Found inconsistent genus in homolgy " + \
#                         "intersection matrix.")
    
    alpha = frobenius_transform(t_matrix, g)

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
    pdb.set_trace()
    c = {}
    c['basepoint'] = mon.base_point()
    c['sheets']    = mon.base_sheets()
    c['genus']     = g
    c['cycles']    = t_basis
    c['linearcombination'] = alpha[:2*g,:]
    c['canonicalcycles'] = ab_cycles(t_basis, alpha[:2*g,:])

    return c




def homology(*args, **kwds):
    return canonical_basis(*args, **kwds)




def simplify_cycle(t_table, lin_comb):
    """
    Simplifies a cycle given as a linear combination of other
    cycles. Returns a list of cycles, the sum of which constitutes the
    new cycle.
    """
    r = len(lin_comb)
    i = 0
    while i != -1:
        if lin_comb[i] != 0:
            index = i
            temp_cycle = t_table[i]
            n = len(temp_cycle)

            if lin_comb[i] < 0:
                # reverse the cycle (with the first element fixed)    
                temp_cycle = [temp_cycle[0]] + \
                    [temp_cycle[n-j] for j in xrange(n-1)]

            # take as many copies of the cycle as needed
            cycle = temp_cycle*abs(lin_comb[i])  # XXX index check
            i = 0
        else:
            i += 1

    # compose this cycle with the other cycles that appear in the linear
    # combination.
    sheets = [cycle[2*i-1] for i in xrange(len(cycle)/2)]
    
    for i in xrange(index1+1,r):   # XXX index check
        if lin_comb[i] != 0:
            temp_cycle = t_table[i]
            n = len(temp_cycle)

            if lin_comb[i] < 0:
                # reverse the cycle (with the first element fixed)    
                temp_cycle = [temp_cycle[0]] + \
                    [temp_cycle[n-j] for j in xrange(n-1)]

            # find a common sheet
            temp_sheets = [temp_cycle[2*j-1] 
                           for j in xrange(len(temp_cycle)/2)]
            # take as many copies of the temp. cycle as needed
            temp_cycle = temp_cycle*abs(lin_comb[i])

            k = 0
            while (k != -1) and (k < len(cycle)): # XXX index check
                if sheets[k] in temp_sheets:
                    sheet = sheet[k]
                    cycle[k] = reorder_cycle(cycle[k], sheet)
                    temp_cycle = reorder_cycle(temp_cycle, sheet)
                    cycle[k] = cycle[k] + temp_cycle  # XXX double check
                    k = 0
                else:
                    k += 1
                    sheets.extend(temp_sheets)  # XXX extend or append
            if k == len(cycle):
                cycle.extend(temp_cycle)

    return cycle



def compress_cycle(cycle):
    """
    Returns a "compressed form" of a list representing a cycle.

    If the (i+2)nd element in the list is equal to the ith element
    then elements (i+1) and (i+2) can be removed from the
    list. Geometrically, this represents...XXX.
    
    Finally, the cycle is rewritten to start fomt eh sheets with the
    smallest sheet number.
    """
    n = len(cycle)
    c_cycle = []

    # XXX there's definitely a way to make this more compact and
    # pythonic
    for i in xrange(n):
        c = cycle[i]
        j = 0
        while j < len(c):
            if j == (len(c)-1):
                if c[j] == c[1]:
                    c = [c[k]] + c[3:]
                    j = 1
                else:
                    j += 1

            elif j == (len(c)-2):
                if c[j] == c[0]:
                    c = [c[k]] + c[:(len(c)-2)]
                    j = 1
                else:
                    j += 1
            else:
                if c[j] == c[j+2]:
                    c = c[:j] + c[(j+2):]  # XXX index check
                    j = 1
                else:
                    j += 1

        c_cycle.extend(reform_cycle(c,min(c)))   # XXX or append?

    return c_cycle
                



def reform_cycle(cycle):
    """
    Rewrite a cycle in a specific form.

    The odd entries in the output list are sheet numbers. The even
    entries are lists with two elements: the first is the location of
    the branch point in the complex plane, the second indicates how
    many times one needs to go around the branch point (in the
    positive direction) to get to the next sheet.
    """
    # XXX F**K THE INDEX TRANSLATION FROM MAPLE SUCKS
    n = len(cycle)
    lijst = cycle[:]  # make a copy, not a pointer to the same list
    for i in xrange(n/2):
        a = lijst[2*i]
        b = lijst[2*i+1]   # XXX check above to make sure no "-1" indices
        if (2*i+1) == n:   # XXX check index
            c = lijst[0]
        else:
            c = lijst[2*i+2]  
            
        pos1 = b[1].index(a)
        pos2 = b[1].index(c)
        mini = min( [abs(pos2-pos1), pos2-pos1+len(b[1]), 
                     abs(pos2-pos1-len(b[1]))] )

        if abs(pos2-pos1) == mini:        around = pos2-pos1
        elif pos2-pos1+len(b[2]) == mini: around = pos2-pos1+len(b[1])
        else:                             around = pos2-pos1-len(b[1])
            
        b[1] = around
        lijst[2*i+1] = b

    return lijst
        


def ab_cycles(t_basis, alpha):
    """
    Returns a list of the basis cycles.
    """
    g,_ = alpha.shape
    c = {}
    for i in xrange(2*g):
        key = b[i-g] if i>=g else a[i]
        c[key] = compress_cycle(simplify_cicle(t_table, alpha[i,:]))

    return c
    



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
    f11= y**2 - x*(x-1)*(x-2)*(x-3)  # simple genus two hyperelliptic
    
    f     = f11

    basis = canonical_basis(f,x,y)

    print basis
