r"""Riemann Theta Integer Points :mod:`abelfunctions.riemanntheta.integer_points`
=============================================================================

Functions for computing the set of integer points used in evaluating the
oscillatory part of the Riemann theta function.

Functions
---------

.. autosummary::

    integer_points

References
----------

.. [CRTF] B. Deconinck, M.  Heil, A. Bobenko, M. van Hoeij and
   M. Schmies, Computing Riemann Theta Functions, Mathematics of
   Computation, 73, (2004), 1417-1442.

.. [DLMF] B. Deconinck, Digital Library of Mathematics Functions -
   Riemann Theta Functions, http://dlmf.nist.gov/21

Contents
--------

"""

cimport cython
import numpy
cimport numpy

from cpython.array cimport array, clone, extend
from libc.math cimport ceil, floor, sqrt, M_PI, lround


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _reallocate_radii(array rad, double[:] radii, array unit_double):
    r"""Double the current memory for radii storage.

    Helper function for :func:`_find_int_points`. Doubles the current
    storage of the array and updates the associated view on the memory.

    Parameters
    ----------
    rad : cpython array
        Current memory allocated to store radius values.
    radii : double[:]
        View on the memory allocated to store radius values.
    unit_double: cpython array
        Memory associated with a single double-sized value in a cpython
        array.

    Returns
    -------
    None

    """
    cdef int size = sizeof(rad)/sizeof(double)
    cdef array temp = clone(unit_double, size, zero=False)
    rad = extend(rad, temp)
    radii = rad

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _reallocate_points(array pts, int[:] points, array unit_int):
    r"""Double the current memory for integer points storage.

    Helper function for :func:`_find_int_points`. Doubles the current
    storage of the array and updates the associated view on the memory.

    Parameters
    ----------
    pts : cpython array
        Current memory allocated to store integer point values.
    points : int[:]
        View on the memory allocated to store integer point values.
    unit_int: cpython array
        Memory associated with a single int-sized value in a cpython
        array.

    Returns
    -------
    None

    """
    cdef int size = sizeof(pts)/sizeof(int)
    cdef array temp = clone(unit_int, size, zero=False)
    pts = extend(pts,temp)
    points = pts

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _reallocate_centers(array cen, double[:] centers, array unit_double):
    r"""Double the current memory for center point storage.

    Helper function for :func:`_find_int_points`. Doubles the current
    storage of the array and updates the associated view on the memory.

    Parameters
    ----------
    cen : cpython array
        Current memory allocated to store center point values.
    centers : double[:]
        View on the memory allocated to store center points= values.
    unit_double: cpython array
        Memory associated with a single double-sized value in a cpython
        array.

    Returns
    -------
    None

    """
    cdef int size = sizeof(cen)/sizeof(double)
    cdef array temp = clone(unit_double, size, zero=False)
    cen = extend(cen,temp)
    centers = cen

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _reallocate(int max_count, array rad, double[:] radii, array pts,
                     int[:] points, array cen, double[:] centers,
                     array unit_int, array unit_double):
    r"""Double the current memory for radii, integer point, and center point
    storage.

    Helper function for :func:`_find_int_points`. Doubles the current
    storage of the arrays and updates the associated views on the
    memory.

    Parameters
    ----------
    max_count: int
        The current number of points for which memory is available.
    rad : cpython array
        Current memory allocated to store radius values.
    radii : double[:]
        View on the memory allocated to store radius values.
    pts : cpython array
        Current memory allocated to store integer point values.
    points : int[:]
        View on the memory allocated to store integer point values.
    cen : cpython array
        Current memory allocated to store center point values.
    centers : double[:]
        View on the memory allocated to store center point values.
    unit_int: cpython array
        Memory associated with a single int-sized value in a cpython array.
    unit_double: cpython array
        Memory associated with a single double-sized value in a cpython array.

    Returns
    -------
    int
        The new maximum number of points for which memory is
        available. Twice the previous storage.
    """
    _reallocate_radii(rad, radii, unit_double)
    _reallocate_points(pts, points, unit_int)
    _reallocate_centers(cen, centers, unit_double)
    return 2*max_count

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int[:] _find_int_points(int genus, double radius,
                      double[:,:] T):
    r"""Calculates the coordinates of the points with integer-valued
    coordinates contained in the ellipsoid defined by the equation

    .. math::

        `\|T\textbf{n}\|\leq \frac{R}{\sqrt{\pi}}`.

    Parameters
    ----------
    genus : int
        The genus associated with the Riemann theta function which
        corresponds to the dimension of the ellipsoid.
    radius : double
        The 'R' in the defining equation of the ellipsoid.
    T : double[:,:]
        The Cholesky decomposition of the imaginary part of the Riemann
        matrix `\Omega`.

    Returns
    -------
    int[:]
        A flattened view on the integer points contained in the ellipsoid.
    """
    cdef:
        int[:] points
        double[:] centers, that, radii, var
        double[:,:] Tnew, Tinv
        double absolute_bound, c_g, r, correction
        int left_bound, right_bound, i, j, new_point_count, l, g, n_g
        int newpoint, point_count, max_count

    point_count = 0   # current number of integer points
    max_count = 1000  # number of points allocated for in memory
    correction = 0.5  # necessary for uniform approximation

    # prototypical int and double array sizes
    cdef array unit_int = array('i', [])
    cdef array unit_double = array('d', [])

    # initial allocation of radius, point, and center point memory
    cdef array rad = clone(unit_double, max_count, zero=False)
    radii = rad
    cdef array pts = clone(unit_int, max_count*genus, zero=False)
    points = pts
    cdef array cen = clone(unit_double, max_count*genus, zero=False)
    centers = cen

    # absolute_bound is the largest unrounded magnitude of an integer
    # point coordinate. left_bound and right_bound are the gth element
    # of the center -/+ the absolute_bound, respectively. Since the
    # center is the g-dimensional vector of zeros on the first
    # iteration, this is just -/+ the absolute bound on the first
    # iteration.
    absolute_bound = radius/(sqrt(M_PI)*T[genus-1,genus-1]) + correction
    left_bound = int(ceil(-absolute_bound))
    right_bound = int(floor(absolute_bound))
    #ensure enough memory is allocated for the new coordinates.
    new_point_count = right_bound - left_bound
    while new_point_count > max_count:
        max_count = _reallocate(max_count, rad, radii, pts, points,
                               cen, centers, unit_int, unit_double)
    # collect the last coordinate of the int_points
    for newpoint in range(left_bound,right_bound + 1):
        radii[point_count] = radius
        points[point_count*genus + genus-1] = newpoint
        point_count += 1

    # the main loop adjusts the center and radius based on where we are
    # along the currently collected coordinates of the integer points by
    # looking at a g-dimensional projection of the ellipsoid.
    for g in range(genus-1,-1,-1):
        if g == 0: # we're done!
            break
        # pull off and store the last column of T for later computation.
        that = T[:g,g]
        # consider a smaller selection of the matrix for the next
        # coordinate.
        Tnew = T[:g,:g]
        Tinv = scipy.linalg.inv(Tnew)

        # the secondary loop performs the actual projection based on
        # each of the previously collected coordinates, reducing the
        # dimension of our problem by one.
        for i in range(point_count):
            #grab the last calculated center and point coordinates for
            #recentering.
            c_g = centers[i*genus + g]
            n_g = points[i*genus + g]

            # Recenter the ellipsoid in all dimensions < g
            var = numpy.dot(Tinv,that) * (n_g-c_g)
            for l in range(g):
                centers[i*genus + l] = centers[i*genus + l] - var[l]

            # Compute a new radius^2 for the g-1st coordinate from data
            # gathered in computing the gth coordinate
            r = radii[i]**2 - M_PI*(T[g,g] * (n_g-c_g)**2)

            # if radius failed to compute, n-c = 0, which implies we're
            # at the edge of the ellipsoid, so append the current center
            # coordinate
            if r <= 0:
                points[i*genus + (g-1)] = lround(centers[i*genus + (g-1)])

            # otherwise, we iterate again, collecting all possible
            # coordinates of the points that can possibly satisfy our
            # initial equation.
            else:
                # compute the actual new radius for the next iteration.
                radii[i] = sqrt(r)

                # get the bounds for our new coordinate, similarly to
                # above.
                absolute_bound = (radii[i]/(sqrt(M_PI)*Tnew[g-1,g-1]) +
                                  correction)
                left_bound = int(ceil(centers[i*genus + (g-1)] -
                                      absolute_bound))
                right_bound = int(floor(centers[i*genus + (g-1)] +
                                        absolute_bound))

                # memory already allocated for first new coordinate
                points[i*genus + (g-1)] = left_bound

                # ensure we have enough memory for the remaining point
                # coordinates
                new_point_count = right_bound - left_bound
                while (new_point_count + point_count > max_count):
                    max_count = _reallocate(max_count,rad,radii,pts,points,
                                            cen,centers,unit_int,unit_double)

                # now construct a new point for each new coordinate
                # combination and store the corresponding radius and
                # center information associated with the coordinates
                # collected so far
                for newpoint in range(left_bound+1, right_bound+1):
                    radii[point_count] = radii[i]
                    for j in range(genus):
                        points[genus*point_count + j] = points[genus*i + j]
                        centers[genus*point_count + j] = centers[genus*i + j]
                    points[genus*point_count + (g-1)] = newpoint
                    point_count += 1

    points = numpy.array(points[0:(point_count*genus)], dtype=numpy.int)
    return points


def _find_int_points_1(int g, c, R, T):
    cdef int x
    cdef int a,b
    points = []
    stack = []
    stack.append(((), g, c, R))
    FINISHED = False
    while (not FINISHED):
        start, g, c, R = stack.pop()
        a = <int>np.ceil((c[g] - R/T[g,g]).real)
        b = <int>np.ceil((c[g] + R/T[g,g]).real)
        #Check if reached the edge of the ellipsoid
        if not a < b:
            if (len(stack) == 0):
                FINISHED = True
            continue
        #Last dimension reached, append points
        if g == 0:
            for x in range(a, b+1):
                s = (x,) + start
                points.extend(s)
        else:
            newT = T[:g,:g]
            newTinv = la.inv(newT)
            for x in range(a, b+1):
                chat = c[:g]
                that = T[:g,g]
                newc = chat - newTinv * that * (x - c[g])
                newR = np.sqrt(R**2 - (T[g,g] * (x - c[g]))**2)
                newStart = (x,) + start
                stack.append((newStart, g - 1, newc, newR[0]))
        if (len(stack) == 0):
            FINISHED = True
    return points

