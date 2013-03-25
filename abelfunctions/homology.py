"""
Homology
"""

import numpy
import scipy
import sympy
import networkx as nx

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

#     cycle = [j]
#     k = pi(j)
#     while k != j:
#         cycle.append(k) # update the cycle
#         k = pi(k)       # iterate
        
    cycles = pi._cycles
    for cycle in cycles:
        if j in cycle:
            return reorder_cycle(cycle, min(cycle))


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
    base_point, base_sheets, branch_points, monodromy, G = hurwitz_system
    
    # XXX USE BRANCH POINT INDICES INSTEAD FOR EASE OF READNIG XXX
    #branch_points = range(len(branch_points))

    # the number of sheets of the Riemann surface
    covering_number = len(base_sheets)

    # the number of branch points
    t = len(branch_points)

    # the branch points together with their permutations, using the
    # notation of Tretkoff and Tretkoff
    c = [[ [branch_points[i], find_cycle(monodromy[i],j)] 
           for j in xrange(covering_number) ]
         for i in xrange(t) ]
    
    # initial construction of first circle: from sheet zero go to all
    # branchpoints on sheet one which are not fixed points
    finished = False
    startseq = []
    for i in xrange(t):
        if len(c[i][0][1]) != 1:
            startseq.append(c[i][0])
        else:
            startseq.append(['stop',c[i][0]])
    
    # initialize Tretkoff table:
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
    #
    # tretkoff['C'][level] consists of a list of data:
    #
    # * level even:
    #    sheet number, 
    
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
                    sourcelist = [n for n in tretkoff['C'][level-1][i]] + []
#                    entry = [[] for _ in xrange(len(sourcelist[2]))]
                    entry = []

                    # construct a new vertex at thsi level for every
                    # branch leaving the vertex at the previous level
                    for j in xrange(len(sourcelist[2])):
                        # a1 = new vertex
                        # a2 = originating vertex
                        # a3 = branches to future vertices
                        # a4 = label in the graph
                        a1 = sourcelist[2][j] + []
                        a2 = sourcelist[0]
                        a3 = []
                        a4 = sourcelist[3] + [j]

                        if a1[0] != 'stop':
                            # (order is important)
                            newlist = reorder_cycle(a1[1],sourcelist[0])

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
# XXX ?
                                    a3.append(['stop',nlk])
                                    # 'edge' denotes a branch pointing
                                    # at an endpoint of the graph
                                    edge = [nlk,a1]
                                    if edge not in final_edges:
                                        # it's the first time this
                                        # edge occurs: add it to the
                                        # list of final edges and mark
                                        # it by a q-endpoint
                                        final_edges.append(edge)
                                        q_edges.append(edge)
                                        l = [['stop',nlk]]
                                        l.extend(a1)
                                        l.append([])
                                        l.append(a4+[k-1])
                                        tretkoff['q'][q_counter] = l
                                        q_counter += 1
#                                         tretkoff['q'][q_counter] = [
#                                             ['stop',nlk],
#                                             a1,
#                                             [],
#                                             a4 + [k-1],
#                                         ]
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
                                            l = a1
                                            l.extend(a2)
                                            l.append(a3)
                                            l.append(a4+[k-1])
                                            tretkoff['p'][p_counter] = l
#                                             tretkoff['p'][p_counter] = [
#                                                 a1,
#                                                 a2,
#                                                 a3,
#                                                 a4 + [k-1],
#                                             ]

                                        #endif edge in q_edges
                                    #endif edge not in final_edges
                                #endif newlist[k] not in used_sheets labels
                            #endfor k in range(1,len(newlist))
                        #endif a1[0] != 'stop'

                        # create the new vertex
#                        entry[j] = [item for item in [a1,a2,a3,a4]]
                        entry.append([a1,a2,a3,a4])

                    #endfor j in range(len(sourcelist)

                    # add / replace the new vertex to the graph
                    assert len(entry) == len(sourcelist[2])
                    tretkoff['C'][level][i] = [item for item in entry]

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
                    sourcelist = tretkoff['C'][level-1][i] + []
                    # construct a new vertex at this level for every
                    # branch leaving the vertex at the previous level
                    entry = [[] for _ in xrange(len(sourcelist[2]))]
#                    entry = []
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
                                            l = [['stop',ckb1]]
                                            l.append(b1)
                                            l.append([])
                                            l.append(b4+[k-startingindex])

#                                             tretkoff['q'][q_counter] = [
#                                                 ['stop',ckb1],
#                                                 b1,
#                                                 [],
#                                                 b4 + [k-startingindex],
#                                             ]
                                            tretkoff['q'][q_counter] = l
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
                                            l = [b1]
                                            l.extend(b2)
                                            l.append(b3)
                                            l.append(b4 + [k-startingindex])
#                                             tretkoff['p'][p_counter] = [
#                                                 b1,
#                                                 b2,
#                                                 b3,
#                                                 b4 + [k-startingindex],
#                                             ]
                                            tretkoff['p'][p_counter] = l
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
                                            l = [['stop',ckb1]]
                                            l.append(b1)
                                            l.append([])
                                            l.append(b4+[k+t-startingindex])

#                                             tretkoff['q'][q_counter] = [
#                                                 ['stop',ckb1],
#                                                 b1,
#                                                 [],
#                                                 b4 + [k+t-startingindex],
#                                             ]
                                            tretkoff['q'][q_counter] = l

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
                                            l = [b1]
                                            l.extend(b2)
                                            l.append(b3)
                                            l.append(b4 + [k+t-startingindex])

#                                             tretkoff['p'][p_counter] = [
#                                                 b1,
#                                                 b2,
#                                                 b3,
#                                                 b4 + [k+t-startingindex],
#                                             ]
                                            tretkoff['p'][p_counter] = l

                                        #end if edge in q_edge
                                    #end if edge not in final_edges
                                #end if ckb1 not in used_bpt_labels
                            #od
                        #fi
                        # create a new vertex
                        entry[j] = [item for item in [a1,a2,a3,a4]]
#                        entry.append([a1,a2,a3,a4])
#                        assert len(entry) == len(sourcelist[2])
                    #od
                    # add the new vertex to the graph
                    tretkoff['C'][level][i] = [item for item in entry]
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
    tretkoff['numberofcycles'] = q_counter
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



def reform_cycle(cycle):
    """
    Rewrite a cycle in a specific form.

    The odd entries in the output list are sheet numbers. The even
    entries are lists with two elements: the first is the location of
    the branch point in the complex plane, the second indicates how
    many times one needs to go around the branch point (in the
    positive direction) to get to the next sheet.

    Input: 

    A cycle of the form

        [s_0, (b_{i_0}, pi_{i_0}), s_1, (b_{i_1}, pi_{i_1}), ...]

    where "s_k" is a sheet number, "b_{i_k}" is the {i_k}'th branch
    point, and "pi_{i_k}" is the corresponding sheet permutation
    associated with the branch point.

    It is assumed that each of these sheet / branch point pairs
    appear uniquely in this cycle since the input is recieved from
    the function "compress_cycle()".

    Output:
    
    A list of the form

        [s_0, (b_{i_0}, n_{i_0}), s_1, (b_{i_1}, n_{i_1}), ...]

    where "s_k" is a sheet number, "b_{i_k}" is the {i_k}'th branch
    point, and "n_{i_k}" is the number of times and direction to go
    about branch point "b_{i_k}".
    """
    n = len(cycle)
    lijst = cycle[:]  # make a copy, not a pointer to the same list
    for i in xrange(n/2):
        # Grab the current sheet (a) + branch point pair (b).
        a = lijst[2*i]
        b = lijst[2*i+1]

        # If we're at the end of the cycle then wrap around to get the
        # "next" sheet. Otherwise, the next sheet is the element
        # following.
        if (2*i+1) == (n-1):
            c = lijst[0]
        else:
            c = lijst[2*i+2]
        
        # Branch points are of the form (branch point number/index,
        # sheet permutation). "a" and "c" are the source and target
        # sheets, respectively. Find where these sheets are located in
        # the permutation and find the distance between the two
        # sheets. This distance is equal to the number of times and
        # direction one must go around the given branch point in order
        # to get from sheet a to sheet c. Of all ways to go around the
        # branch point to get from sheet a to c the one with the
        # fewest number of rotations is selected.
        pos1 = b[1].index(a)
        pos2 = b[1].index(c)
        mini = min( [abs(pos2-pos1), pos2-pos1+len(b[1]), 
                     abs(pos2-pos1-len(b[1]))] )

        if abs(pos2-pos1) == mini:        around = pos2-pos1
        elif pos2-pos1+len(b[2]) == mini: around = pos2-pos1+len(b[1])
        else:                             around = pos2-pos1-len(b[1])
            
        # Replace the permutation with the number of times 
        b = (b[0], around)
        lijst[2*i+1] = b

    return lijst




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


    f = f2

    hom = homology(f,x,y)
    for key,value in hom.iteritems():
        print key
        print value
        print
