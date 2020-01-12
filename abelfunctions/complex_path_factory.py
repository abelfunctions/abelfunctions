r"""X-Path Factory :mod:`abelfunctions.complex_path_factory`
=================================================

Module for computing the monodromy group of the set of discriminant points
of a complex plane algebraic curve.

"""
import numpy

from numpy import double, complex, floor, angle
from sage.all import infinity, QQbar, scatter_plot

from abelfunctions.complex_path import (
    ComplexLine,
    ComplexArc,
    ComplexPath,
)


class ComplexPathFactory(object):
    r"""Factory for computing complex paths on the x-projection of a Riemann surface
    determined by an algebraic curve :math:`C : f(x,y) = 0`.

    Since paths on a Riemann surface are computed via analytic continuation
    care needs to be taken when the x-part of the path gets close to a
    discriminant point of the algebraic curve from which the Riemann surface is
    derived. This is because some of the y-sheets of the curve, when considered
    as a covering of the complex x-plane, coalesce at the discriminant points.
    Therefore, "bounding circles" need to be computed at each discriminant
    point.

    Attributes
    ----------
    riemann_surface : RiemannSurface
        The Riemann surface on which to construct the x-paths.
    base_point : complex
        If a base point isn't provided, one will be chosen.
    kappa : double (default: 3/5)
        A scaling factor between 0.5 and 1.0 used to modify the radius
        of the bounding circles.
    discriminant_points
        The discriminant points of the curve.
    discriminant_points_complex
        Floating point approximations of the discriminant points. Used for
        computational efficiency since converting from QQbar to CDF is slow

    Methods
    -------
    .. autosummary::

      closest_discriminant_point
      radius
      intersecting_discriminant_points
      intersects_discriminant_points
      intersection_points
      path_to_monodromy_point
      path
      monodromy_path
      monodromy_path_infinity
      show_paths

    """
    @property
    def base_point(self):
        return self._base_point

    @property
    def discriminant_points(self):
        return self._discriminant_points

    @property
    def discriminant_points_complex(self):
        return self._discriminant_points_complex

    @property
    def radii(self):
        return self._radii

    def __init__(self, f, base_point=None, kappa=3./5.):
        """Initialize a complex path factory.

        Complex path factories require a base point from which most complex
        paths begin on a Riemann surface. In particular, this base point is
        used as the base point in constructing the monodromy group of the
        Riemann surface.

        Parameters
        ----------
        f : polynomial
            The plane algebraic curve defining the Riemann surface.
        base_point : complex
            The base point of the factory and of the monodromy group of the
            Riemann surface. If not provided one will be chosen based on the
            discriminant point placement.
        kappa : double
            A scaling factor used to determine the radii of the "bounding
            circles" around each discriminant point. `kappa = 1.0` means the
            bounding circles are made as large as possible, resulting in
            possibly touching circles between two or more discriminant points.

        """
        self.f = f

        # compute the discriminant points and determine a base point if none
        # was provided
        b,d,dc = self._compute_discriminant_points(base_point)
        self._base_point = b
        self._discriminant_points = d
        self._discriminant_points_complex = dc

        # compute the bounding circle radii from the discriminant points
        r = self._compute_radii(kappa)
        self._radii = r

    def _compute_discriminant_points(self, base_point):
        r"""Computes and stores the discriminant points of the underlying curve.

        A discriminant point :math:`x=b` is an x-point where at least one
        y-root lying above has multiplicity greater than one. A
        :class:`PuiseuxTSeries` is required to represent a place on the Riemann
        surface whose x-projection is a discriminant point. These kinds of
        places are of type :class:`DiscriminantPlace`.

        .. note::

            The ordering of the discriminant points is important for the
            purposes of computing the monodromy group, which is done in the
            :class:`RiemannSurfacePathFactory` attribute, `PathFactory`.

        Parameters
        ----------
        None

        Returns
        -------
        list : complex
            Return a list of ordered discriminant points from the base point.

        """
        # compute the symbolic and numerical discriminant points
        f = self.f
        x,y = f.parent().gens()
        res = f.resultant(f.derivative(y), y).univariate_polynomial()
        rts = res.roots(ring=QQbar, multiplicities=False)
        discriminant_points = numpy.array(rts)
        discriminant_points_complex = numpy.array(rts, dtype=complex)

        # determine a base_point, if not specified
        if not base_point:
            a = min(complex(bi).real for bi in discriminant_points)
            a = a - 1
            aint = complex(floor(a))
            base_point = aint

        # sort the discriminant points first by argument with the base point
        # and then by distance from the base point. the points need to be exact
        centered_points = discriminant_points_complex - base_point
        distances = abs(centered_points)
        arguments = angle(centered_points)
        sort_index = numpy.lexsort((distances, arguments))

        # sort and return
        discriminant_points = discriminant_points[sort_index]
        discriminant_points_complex = discriminant_points_complex[sort_index]
        return base_point, discriminant_points, discriminant_points_complex

    def closest_discriminant_point(self, x, exact=True):
        r"""Returns the closest discriminant point to a point x.

        An often-used helper function by several components of
        :class:`RiemannSurface`.

        Parameters
        ----------
        x : complex
            A complex x-point.
        exact : boolean
            If `True`, returns a `sympy.Expr` representing the discriminant
            point exactly. Otherwise, returns a numerical approximation.

        Returns
        -------
        complex or sympy.Expr
            The discriminant point, either exact or numerical.
        """
        # use floating points approximations for performance
        b = self.discriminant_points_complex
        x = complex(x)
        idx = numpy.argmin(abs(b - x))
        if exact:
            return self.discriminant_points[idx]
        return self.discriminant_points_complex[idx]

    def _compute_radii(self, kappa):
        """Returns the radii of the bounding circles.

        Parameters
        ----------
        kappa : double
            A scaling factor between 0.5 and 1.0. `kappa = 1.0` means that the
            bounding circles are taken to be as large as possible without
            overlapping.

        Returns
        -------
        radii : array
            An ordered list of radii. The radius at index `k` is associated
            with the discriminant point at index `k` in
            `self.discriminant_points`.
        """
        # special case when there is only one finite discriminant point: take
        # the distance from the base point to the discriminant point (scaled by
        # kappa, of course)
        if len(self.discriminant_points_complex) == 1:
            b = self.discriminant_points_complex[0]
            radius = numpy.abs(self.base_point - b)
            radius *= kappa/2.0
            radii = numpy.array([radius], dtype=double)
            return radii

        # when there is more than one discriminant point we scale disctances
        # accordingly. coerce to numerical.
        radii = []
        b = self.discriminant_points_complex
        for bi in b:
            dists = [abs(bi - bj) for bj in self.discriminant_points_complex
                     if bi != bj]
            rho = min(dists)
            radius = rho*kappa/2.0
            radii.append(radius)
        radii = numpy.array(radii, dtype=double)

        # final check: assert that the base point is sufficiently far away from
        # the discriminant points
        dists = [abs(bi - self.base_point) for bi in b]
        dists = numpy.array(dists, dtype=double) - radii
        if any(dists < 0):
            raise ValueError('Base point lies in the bounding circles of the '
                             'discriminant points. Use different base point or '
                             'circle scaling factor kappa.')
        return radii

    def radius(self, bi):
        """Returns the radius of the bounding circle around `bi`.

        Parameters
        ----------
        bi : complex
            A discriminant point of the algebraic curve.

        Returns
        -------
        radius : double
            The radius of the bounding circle.
        """
        # find the index where bi appears in the list of discriminant points.
        # it's done numerically in case a numerical approximation bi is given
        bi = complex(bi)
        index = 0
        for z in self.discriminant_points_complex:
            if abs(z-bi) < 1e-14:
                break
            index += 1

        # raise an error if not found
        if index == len(self.discriminant_points_complex):
            raise ValueError('%s is not a discriminant point of %s' % (bi, self.f))

        radius = self.radii[index]
        return radius

    def intersecting_discriminant_points(self, z0, z1, exact=False):
        r"""Return the discriminant points which are too close to the line from
        `z0` to `z1` along with the corresponding orientations.

        Parameters
        ----------
        z0 : complex
            Line start.
        z1 : complex
            Line end.

        Returns
        -------
        """
        if exact:
            points = [bi for bi in self.discriminant_points
                      if self.intersects_discriminant_point(z0, z1, bi)]
        else:
            points = [bi for bi in self.discriminant_points_complex
                      if self.intersects_discriminant_point(z0, z1, bi)]
        return points

    def intersects_discriminant_point(self, z0, z1, bi):
        """Returns `True` if the line from `z0` to `z1` intersects the bounding circle
        around the discriminant point `bi`.

        Parameters
        ----------
        z0 : complex
            Line starting point.
        z1 : complex
            Line ending point.
        bi : complex
            A discriminant point.

        Returns
        -------
        is_intersecting : bool
            `True` if the line from `z0` to `z1` gets too close to `bi`.
        """
        # first check the perpendicular distance from bi to the line
        # passing through z0 and z1
        z0 = complex(z0)
        z1 = complex(z1)
        bi = complex(bi)
        direction = numpy.sign(angle(z1-z0) - angle(bi-z0))
        normv = abs(z1-z0)
        v = 1.0j*direction*(z1 - z0)
        r = z0 - bi

        # degenerate case: the line through z0 and z1 crosses bi. in this case
        # just check if the branch point lies in between
        if direction == 0:
            if (abs(bi - z0) <= normv) and (abs(bi - z1) <= normv):
                return True
            return False

        # return False if the distance from the _line_ passing through
        # z0 and z1 to bi is greater than the radius fo teh bounding
        # circle.
        distance = (v.real*r.real + v.imag*r.imag)
        distance = distance / normv
        if distance > self.radius(bi):
            return False

        # also need to check if bi "lies between" the _line segment_
        # between z0 and z1. use the distance vector w = d*v/|v|. the
        # distance from vtilde to z0 and z1 should be less that the
        # distance between z0 and z1
        w = distance*v/normv + bi
        if (abs(w - z0) <= normv) and (abs(w - z1) <= normv):
            return True
        return False

    def intersection_points(self, z0, z1, b, R):
        """Returns the complex points `w0,w1` where the line from `z0` to `z1`
        intersects the bounding circle around `bi`.

        Parameters
        ----------
        z0 : complex
            Line starting point.
        z1 : complex
            Line ending point.
        bi : complex
            A discriminant point.
        Ri : double
            The radius of the circle around bi.

        Returns
        -------
        w0, w1 : complex
            Points on the bounding circle of `bi` where the line z0-z1
            intersects.

        """
        # special case when z1 = b: 
        if abs(z1 - b) < 1e-14:
            R = self.radius(b)
            b = complex(b)
            l = lambda s: z0 + (b - z0)*s
            s = 1.0 - R/abs(z0 - b)
            z = l(s)
            return z,z

        # construct the polynomial giving the distance from the line l(t),
        # parameterized by t in [0,1], to bi.
        z0 = complex(z0)
        z1 = complex(z1)
        b = complex(b)
        R = double(R)
        v = z1 - z0
        w = z0 - b
        p2 = v.real**2 + v.imag**2
        p1 = 2*(v.real*w.real + v.imag*w.imag)
        p0 = w.real**2 + w.imag**2 - R**2   # solving |l(t) - bi| = Ri

        # find the roots of this polynomial and sort by increasing t
        p = numpy.poly1d([p2, p1, p0])
        t = numpy.roots(p)
        t.sort()

        # compute ordered intersection points
        w0 = v*t[0] + z0   # first intersection point
        w1 = v*t[1] + z0   # second intersection point
        return w0,w1

    def path_to_discriminant_point(self, bi):
        r"""Returns the complex path to the bounding circle around `bi` which avoids
        other discriminant points.

        This is a specific implementation of the routine used in
        :meth:`path_to_point`. Although similar, this routine takes branch
        point ordering into account when determining whether to go above or
        below intersecting discriminant points. (See
        :meth:`intersecting_discriminant_points`)

        Parameters
        ----------
        bi : complex
            A discriminant / branch point of the curve.

        Returns
        -------
        gamma : ComplexPath
            The corresponding monodromy path.

        See Also
        --------
        intersecting_discriminant_points
        path_to_point
        """
        # make sure we have the discriminant point exactly
        point = self.closest_discriminant_point(bi, exact=True)
        if abs(complex(point) - complex(bi)) > 1e-4:
            raise ValueError('%s is not a discriminant point of %s'%(bi,self.f))
        bi = point
        Ri = self.radius(bi)

        # compute the list points we need to stay sufficiently away from and
        # sort them in increasing distance from the base point
        z0 = self.base_point
        _,z1 = self.intersection_points(z0, complex(bi), bi, Ri)
        points_to_avoid = self.intersecting_discriminant_points(z0, z1, exact=False)
        points_to_avoid.sort(key=lambda bj: abs(bj-z0))

        # determine the relative orientations of the avoiding discriminant
        # points with the point bi. recall that the ordering of discriminant
        # points establishes the orientation. (points earlier in the list lie
        # below those later in the list.)
        #
        # positive/negative orientation with a given bj means we need to go
        # above/below bj, respectively.
        orientations = []
        i = numpy.argwhere(self.discriminant_points_complex == complex(bi)).item(0)
        for bj in points_to_avoid:
            j = numpy.argwhere(self.discriminant_points_complex == bj).item(0)
            if i < j:
                orientations.append(-1)
            else:
                orientations.append(1)

        # we now have sorted orientations and points to avoid. for each such
        # point:
        #
        # 1. determine the points of intersection with the bounding circle
        # 2. determine the appropriate arc along the bounding circle
        # 3. construct the path segment using a line (if necessary) and the arc
        segments = []
        for j in range(len(points_to_avoid)):
            bj = points_to_avoid[j]
            oj = orientations[j]
            Rj = self.radius(bj)
            w0,w1 = self.intersection_points(z0,z1,bj,Rj)
            arc = self.avoiding_arc(w0,w1,bj,Rj,orientation=oj)

            if abs(z0-w0) > 1e-14:
                segments.append(ComplexLine(z0,w0))
            segments.append(arc)

            # repeat by setting the new "start point" to be w1, the last point
            # reached on the arc.
            z0 = w1

        # build the avoiding path from the segments
        segments.append(ComplexLine(z0,z1))
        if len(segments) == 1:
            path = segments[0]
        else:
            path = ComplexPath(segments)
        return path

    def path(self, z0, z1):
        r"""Returns the complex path to the bounding circle around `bi` which avoids
        other discriminant points.

        This is a specific implementation of the routine used in :meth:`path`.
        Although similar, this routine takes branch point ordering into account
        when determining whether to go above or below intersecting discriminant
        points. (See :meth:`intersecting_discriminant_points`)

        Parameters
        ----------
        bi : complex
            A discriminant / branch point of the curve.

        Returns
        -------
        gamma : ComplexPath
            The corresponding monodromy path.

        See Also
        --------
        intersecting_discriminant_points
        path
        """
        # compute the list points we need to stay sufficiently away from and
        # sort them in increasing distance from the base point
        points_to_avoid = self.intersecting_discriminant_points(z0, z1, exact=False)
        points_to_avoid.sort(key=lambda bj: abs(bj-z0))

        # for each points we want to avoid
        #
        # 1. determine the points of intersection with the bounding circle
        # 2. determine the appropriate arc along the bounding circle
        # 3. construct the path segment using a line (if necessary) and the arc
        segments = []
        for j in range(len(points_to_avoid)):
            bj = points_to_avoid[j]
            Rj = self.radius(bj)
            w0,w1 = self.intersection_points(z0,z1,bj,Rj)
            arc = self.avoiding_arc(w0,w1,bj,Rj)

            if abs(z0-w0) > 1e-14:
                segments.append(ComplexLine(z0,w0))
            segments.append(arc)

            # repeat by setting the new "start point" to be w1, the last point
            # reached on the arc.
            z0 = w1

        # append the final line and build the avoiding path from the segments
        segments.append(ComplexLine(z0,z1))
        if len(segments) == 1:
            path = segments[0]
        else:
            path = ComplexPath(segments)
        return path

    def monodromy_path(self, bi, nrots=1):
        """Returns the complex path starting from the base point, going around the
        discriminant point `bi` `nrots` times, and returning to the base
        x-point.

        The sign of `nrots` indicates the sign of the direction.

        Parameters
        ----------
        bi : complex
            A discriminant point.
        nrots : integer (default `1`)
            A number of rotations around this discriminant point.

        Returns
        -------
        path : ComplexPath
            A complex path representing the monodromy path with `nrots`
            rotations about the discriminant point `bi`.

        """
        if bi in [infinity, numpy.Infinity, 'oo']:
            return self.monodromy_path_infinity(nrots=nrots)

        path_to_bi = self.path_to_discriminant_point(bi)

        # determine the rotational path around the discriminant point
        z = path_to_bi(1.0)
        bi = complex(bi)
        Ri = self.radius(bi)
        theta = angle(z - bi)
        dtheta = numpy.pi if nrots > 0 else -numpy.pi
        circle = ComplexArc(Ri, bi, theta, dtheta) + \
                 ComplexArc(Ri, bi, theta + dtheta, dtheta)
        path_around_bi = circle
        for _ in range(abs(nrots)-1):
            path_around_bi += circle

        # the monodromy path is the sum of the path to the point, the
        # rotational part, and the return path to the base point
        path = path_to_bi + path_around_bi + path_to_bi.reverse()
        return path

    def monodromy_path_infinity(self, nrots=1):
        """Returns the complex path starting at the base point, going around
        infinity `nrots` times, and returning to the base point.

        This path is sure to not only encircle all of the discriminant
        points but also stay sufficiently outside the bounding circles
        of the points.

        Parameters
        ----------
        nrots : integer, (default `1`)
            The number of rotations around infinity.

        Returns
        -------
        RiemannSurfacePath
            The complex path encircling infinity.

        """
        path = []

        # determine the radius R of the circle, centered at the origin,
        # encircling all of the discriminant points and the bounding circles
        b = self.discriminant_points
        R = numpy.abs(self.base_point)
        for bi in b:
            radius = self.radius(bi)
            Ri = numpy.abs(bi) + 2*radius  # to be safely away
            R = Ri if Ri > R else R

        # the path begins with a line starting at the base point and ending at
        # the point -R (where the circle will begin)
        path = ComplexLine(self.base_point, -R)

        # the positive direction around infinity is equal to the
        # negative direction around the origin
        dtheta = -numpy.pi if nrots > 0 else numpy.pi
        for _ in range(abs(nrots)):
            path += ComplexArc(R, 0, numpy.pi, dtheta)
            path += ComplexArc(R, 0, 0, dtheta)

        # return to the base point
        path += ComplexLine(-R, self.base_point)

        # determine if the circle actually touches the base point. this occurs
        # when the base point is further away from the origin than the bounding
        # circles of discriminant points. in this case, the path only consists
        # of the arcs defining the circle
        if abs(self.base_point + R) < 1e-15:
            path = ComplexPath(path.segments[1:-1])
        return path


    def show_paths(self, *args, **kwds):
        """Plots all of the monodromy paths of the curve.

        Returns
        -------
        None
        """
        # fill the bounding circles around each discriminant point
        a = complex(self.base_point)
        b = numpy.array(self.discriminant_points, dtype=complex)

        # plot the base point and the discriminant points
        pts = [(a.real, a.imag)]
        plt = scatter_plot(pts, facecolor='red', **kwds)
        pts = list(zip(b.real, b.imag))
        plt += scatter_plot(pts, facecolor='black', **kwds)

        # plot the monodromy paths
        for bi in b:
            path = self.monodromy_path(bi)
            plt += path.plot(**kwds)
        return plt

    def avoiding_arc(self, w0, w1, b, R, orientation=None):
        """Returns the arc `(radius, center, starting_theta, dtheta)`, from the points
        `w0` and `w1` on the bounding circle around `bi`.

        The arc is constructed in such a way so that the monodromy properties
        of the path are conserved.

        Parameters
        ----------
        w0 : complex
            The starting point of the arc on the bounding circle of `bi`.
        w1 : complex
            The ending point of the arc on the bounding circle of `bi`.
        b : complex
            The discriminant point to avoid.
        R : double
            The radius of the bounding circle.

        Returns
        -------
        arc : ComplexArc
            An arc from `w0` to `w1` around `bi`.

        """
        w0 = complex(w0)
        w1 = complex(w1)
        b = complex(b)
        R = double(R)

        # ASSUMPTION: Re(w0) < Re(w1)
        if w0.real >= w1.real:
            raise ValueError('Cannot construct avoiding arc: all paths must '
                             'travel from left to right unless "reversed".')

        # ASSERTION: w0 and w1 lie on the circle of radius Ri centered at bi
        R0 = abs(w0 - b)
        R1 = abs(w1 - b)
        if abs(R0 - R) > 1e-13 or abs(R1 - R) > 1e-13:
            raise ValueError('Cannot construct avoiding arc: '
                             '%s and %s must lie on the bounding circle of '
                             'radius %s centered at %s'%(w0,w1,R,b))

        # degenerate case: w0, bi, w1 are co-linear
        #
        # if no orientation is provided then go above. otherwise, adhere to the
        # orientation: orientation = +1/-1 means the path goes above/below
        phi_w0_w1 = numpy.angle(w1-w0)
        phi_w0_b = numpy.angle(b-w0)
        if abs(phi_w0_w1 - phi_w0_b) < 1e-13:
            theta0 = numpy.angle(w0-b)
            dtheta = -numpy.pi  # default above
            if not orientation is None:
                dtheta *= orientation
            return ComplexArc(R, b, theta0, dtheta)

        # otherwise: w0, bi, w1 are not co-linear
        #
        # first determine if the line form w0 to w1 is above or below the
        # branch point bi. this will determine if dtheta is negative or
        # positive, respectively
        if phi_w0_b <= phi_w0_w1:
            dtheta_sign = -1
        else:
            dtheta_sign = 1

        # now determine the angle between w0 and w1 on the circle. since w0,
        # bi, and w1 are not colinear this angle must be normalized to be in
        # the interval (-pi,pi)
        theta0 = numpy.angle(w0 - b)
        theta1 = numpy.angle(w1 - b)
        dtheta = theta1 - theta0
        if dtheta > numpy.pi:
            dtheta = 2*numpy.pi - dtheta
        elif dtheta < -numpy.pi:
            dtheta = 2*numpy.pi + dtheta

        # sanity check: |dtheta| should be less than pi
        if abs(dtheta) >= numpy.pi:
            raise ValueError('Cannot construct avoiding arc: '
                             '|dtheta| must be less than pi.')

        dtheta = dtheta_sign * abs(dtheta)

        # finally, take orentation into account. orientation is a stronger
        # condition than the above computations.
        #
        # in the case when the signs of the orientation and the dtheta are
        # opposite then do nothing since: orentation = +1/-1 implies go
        # above/below implies dtheta negative/positive.
        #
        # when the signs are same then make adjustments:
        if not orientation is None:
            if orientation == 1 and dtheta > 0:
                dtheta = dtheta - 2*numpy.pi
            elif orientation == -1 and dtheta < 0:
                dtheta = 2*numpy.pi + dtheta

        # add the path from z0 to w1 going around bi
        arc = ComplexArc(R, b, theta0, dtheta)
        return arc

