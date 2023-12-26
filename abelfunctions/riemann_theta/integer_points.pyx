r"""Riemann Theta Integer Points :mod:`abelfunctions.riemanntheta.integer_points`
=============================================================================

Functions for computing the set of integer points used in evaluating the
oscillatory part of the Riemann theta function.

Functions
---------

.. autosummary::

    integer_points_python

References
----------

.. [CRTF] B. Deconinck, M.  Heil, A. Bobenko, M. van Hoeij and M. Schmies,
   Computing Riemann Theta Functions, Mathematics of Computation, 73, (2004),
   1417-1442.

.. [DLMF] B. Deconinck, Digital Library of Mathematics Functions - Riemann Theta
   Functions, http://dlmf.nist.gov/21

Contents
--------

"""

cimport cython
import numpy
cimport numpy

from array import array
from cpython.array cimport clone, extend
from cpython.array cimport array as c_array
from libc.math cimport ceil, floor, sqrt, M_PI, lround

from sage.all import cached_function


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _reallocate_radii(c_array rad, double[:] radii, c_array unit_double):
    r"""Double the current memory for radii storage.

    Helper function for :func:`_find_int_points`. Doubles the current storage of
    the array and updates the associated view on the memory.

    Parameters
    ----------
    rad : cpython array
        Current memory allocated to store radius values.
    radii : double[:]
        View on the memory allocated to store radius values.
    unit_double: cpython array
        Memory associated with a single double-sized value in a cpython array.

    Returns
    -------
    None

    """
    cdef int size = sizeof(rad)/sizeof(double)
    cdef c_array temp = clone(unit_double, size, zero=False)
    rad = extend(rad, temp)
    radii = rad

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _reallocate_points(c_array pts, int[:] points, c_array unit_int):
    r"""Double the current memory for integer points storage.

    Helper function for :func:`_find_int_points`. Doubles the current storage of
    the array and updates the associated view on the memory.

    Parameters
    ----------
    pts : cpython array
        Current memory allocated to store integer point values.
    points : int[:]
        View on the memory allocated to store integer point values.
    unit_int: cpython array
        Memory associated with a single int-sized value in a cpython array.

    Returns
    -------
    None

    """
    cdef int size = sizeof(pts)/sizeof(int)
    cdef c_array temp = clone(unit_int, size, zero=False)
    pts = extend(pts,temp)
    points = pts

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _reallocate_centers(c_array cen, double[:] centers, c_array unit_double):
    r"""Double the current memory for center point storage.

    Helper function for :func:`_find_int_points`. Doubles the current storage of
    the array and updates the associated view on the memory.

    Parameters
    ----------
    cen : cpython array
        Current memory allocated to store center point values.
    centers : double[:]
        View on the memory allocated to store center points= values.
    unit_double: cpython array
        Memory associated with a single double-sized value in a cpython array.

    Returns
    -------
    None

    """
    cdef int size = sizeof(cen)/sizeof(double)
    cdef c_array temp = clone(unit_double, size, zero=False)
    cen = extend(cen,temp)
    centers = cen

@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _reallocate(int max_count, c_array rad, double[:] radii, c_array pts,
                     int[:] points, c_array cen, double[:] centers,
                     c_array unit_int, c_array unit_double):
    r"""Double the current memory for radii, integer point, and center point
    storage.

    Helper function for :func:`_find_int_points`. Doubles the current storage of
    the arrays and updates the associated views on the memory.

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
        The new maximum number of points for which memory is available. Twice
        the previous storage.
    """
    _reallocate_radii(rad, radii, unit_double)
    _reallocate_points(pts, points, unit_int)
    _reallocate_centers(cen, centers, unit_double)
    return 2*max_count

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef int[:] integer_points_cython(int genus, double radius, double[:,:] T):
    r"""Calculates the coordinates of the points with integer-valued coordinates
    contained in the ellipsoid defined by the equation

    .. math::

        `\|T\textbf{n}\|\leq \frac{R}{\sqrt{\pi}}`.

    Parameters
    ----------
    genus : int
        The genus associated with the Riemann theta function which corresponds
        to the dimension of the ellipsoid.
    radius : double
        The 'R' in the defining equation of the ellipsoid.
    T : double[:,:]
        The Cholesky decomposition of the imaginary part of the Riemann matrix
        `\Omega`.

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
    cdef c_array unit_int = array('i', [])
    cdef c_array unit_double = array('d', [])

    # initial allocation of radius, point, and center point memory
    cdef c_array rad = clone(unit_double, max_count, zero=False)
    radii = rad
    cdef c_array pts = clone(unit_int, max_count*genus, zero=False)
    points = pts
    cdef c_array cen = clone(unit_double, max_count*genus, zero=False)
    centers = cen

    # absolute_bound is the largest unrounded magnitude of an integer point
    # coordinate. left_bound and right_bound are the gth element of the center
    # -/+ the absolute_bound, respectively. Since the center is the
    # g-dimensional vector of zeros on the first iteration, this is just -/+ the
    # absolute bound on the first iteration.
    absolute_bound = radius/(sqrt(M_PI)*T[genus-1,genus-1]) + correction
    left_bound = int(ceil(-absolute_bound))
    right_bound = int(floor(absolute_bound))

    # ensure enough memory is allocated for the new coordinates.
    new_point_count = right_bound - left_bound
    while new_point_count > max_count:
        max_count = _reallocate(max_count, rad, radii, pts, points,
                               cen, centers, unit_int, unit_double)
    # collect the last coordinate of the int_points
    for newpoint in range(left_bound,right_bound + 1):
        radii[point_count] = radius
        points[point_count*genus + genus-1] = newpoint
        point_count += 1

    # the main loop adjusts the center and radius based on where we are along
    # the currently collected coordinates of the integer points by looking at a
    # g-dimensional projection of the ellipsoid.
    for g in range(genus-1,-1,-1):
        if g == 0: # we're done!
            break
        # pull off and store the last column of T for later computation.
        that = T[:g,g]
        # consider a smaller selection of the matrix for the next coordinate.
        Tnew = T[:g,:g]
        Tinv = numpy.linalg.inv(Tnew)

        # the secondary loop performs the actual projection based on each of the
        # previously collected coordinates, reducing the dimension of our
        # problem by one.
        for i in range(point_count):
            # grab the last calculated center and point coordinates for
            # recentering.
            c_g = centers[i*genus + g]
            n_g = points[i*genus + g]

            # Recenter the ellipsoid in all dimensions < g
            var = numpy.dot(Tinv,that) * (n_g-c_g)
            for l in range(g):
                centers[i*genus + l] = centers[i*genus + l] - var[l]

            # Compute a new radius^2 for the g-1st coordinate from data gathered
            # in computing the gth coordinate
            r = radii[i]**2 - M_PI*(T[g,g] * (n_g-c_g)**2)

            # if radius failed to compute, n-c = 0, which implies we're at the
            # edge of the ellipsoid, so append the current center coordinate
            if r <= 0:
                points[i*genus + (g-1)] = lround(centers[i*genus + (g-1)])

            # otherwise, we iterate again, collecting all possible coordinates
            # of the points that can possibly satisfy our initial equation.
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

                # now construct a new point for each new coordinate combination
                # and store the corresponding radius and center information
                # associated with the coordinates collected so far
                for newpoint in range(left_bound+1, right_bound+1):
                    radii[point_count] = radii[i]
                    for j in range(genus):
                        points[genus*point_count + j] = points[genus*i + j]
                        centers[genus*point_count + j] = centers[genus*i + j]
                    points[genus*point_count + (g-1)] = newpoint
                    point_count += 1

    points = numpy.array(points[0:(point_count*genus)], dtype=numpy.int32)
    return points


def _find_int_points_python(g, R, T, c, start):
    r"""Helper function for :func:`integer_points_python`."""
    # determine the endpoints of this dimension of the ellipsoid and check if we
    # reached a boundary
    a_ = c[g] - R/(numpy.sqrt(numpy.pi)*T[g,g])
    b_ = c[g] + R/(numpy.sqrt(numpy.pi)*T[g,g])
    a = int(numpy.ceil(a_))
    b = int(numpy.floor(b_))
    if not a <= b:
        return numpy.array([], dtype=numpy.double)

    # construct the integer points when the final dimension is reached
    if g == 0:
        points = numpy.array([], dtype=numpy.double)
        for i in range(a, b+1):
            # this algorithm works backwards on the coordinates: the last
            # coordinate found is n1 if our coordinates are {n1,n2,...,ng}
            points = numpy.append(numpy.append([i],start), points)
        return points

    # compute new shifts, radii, start, and recurse
    newg = g-1
    newT = T[:(newg+1),:(newg+1)]
    newTinv = numpy.linalg.inv(newT)
    pts = []
    for n in range(a, b+1):
        chat = c[:newg+1]
        that = T[:newg+1,g]
        newc = (chat.T - numpy.dot(newTinv, that)*(n - c[g])).T
        newR = R**2 - numpy.pi*T[g,g]**2*(n-c[g])**2
        newR = numpy.sqrt(newR)
        newstart = numpy.append([n],start)
        newpts = _find_int_points_python(newg,newR,newT,newc,newstart)
        pts = numpy.append(pts,newpts)
    return pts


def integer_points_python(g, R, T):
    r"""Returns the set of integer points used in computing the Riemann theta
    function finite sum.

    Parameters
    ----------
    g : int
        Genus.
    R : double
        Primary radius of the ellipsoid. See :func:`radius.radius` for more
        information.
    T : double[:,:]
        The Cholesky decomposotion of the imaginary part of the Riemann matrix.

    Returns
    -------
    double[:,:]

        An array of integer vectors (given as doubles) in row-dominant
        form. That is, each row of the output array is an integer vector over which
        the finite sum is computed.
    """
    c = numpy.zeros((g,1))
    points = _find_int_points_python(g-1, R, T, c, [])
    points = numpy.array(points, dtype=numpy.double)
    N = len(points)//g
    points.resize((N,g))
    return points
