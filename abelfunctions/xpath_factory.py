"""
Monodromy

Module for computing the monodromy group of the set of discriminant points
of a complex plane algebraic curve.
"""

import numpy
import scipy
import sympy
import networkx as nx
import matplotlib.pyplot as plt

import pdb

class XPathFactory(object):
    """X-paths are lists of tuples representing straight line segments `(z0,
    z1)` or arcs `(radius, center, starting_argument, delta_argument)`
    in the complex plane that together form a piecewise continuous
    path. An `XPathFactory` is a structure for constructing x-paths.

    Since paths on a Riemann surface are computed via analytic
    continuation care needs to be taken when the x-part of the path gets
    close to a discriminant point of the algebraic curve from which the
    Riemann surface is derived. This is because some of the y-sheets of
    the curve, when considered as a covering of the complex x-plane,
    coalesce at the discriminant points. Therefore, "bounding circles"
    need to be computed at each discriminant point.

    Attributes
    ----------
    RS : RiemannSurface
    base_point : complex, optional
        If a base point isn't provided, one will be chosen.
    kappa : double (default: 3/5)
        A scaling factor between 0.5 and 1.0 used to modify the radius
        of the bounding circles.

    Methods
    -------
    discriminant_points
        Returns the discriminant points of the curve.
    xpath_to_discriminant_point
        Returns an x-path from the base point to the bounding circle
        around a given discriminant point. (Implemented in subclasses.)
    xpath_circle_discriminant_point
        Returns an x-path parameterizing the bounding circle around a
        given discriminant point with start and end equal to the ending
        point of `xpath_to_discriminant_point()`. (Implemented in
        subclasses.)
    xpath_monodromy_path
        Returns the x-path representing the monodromy path of the given
        discriminant point. Used in computing the monodromy group of the
        curve.
    xpath_to_point
        Returns an x-path to an arbitrary point in the complex
        plane. (Implemented in subclasses.)
    xpath_reverse
        Reverses an x-path.
    show_paths
        Plots all of the monodromy paths of the curve.

    """
    def __init__(self, RS, base_point=None, kappa=3./5.):
        """Initialize a path factory.

        Subclasses should overload how to compute a base point if none
        is provided.

        Arguments
        ---------
        RS : RiemannSurface
        base_point : complex
        kappa : double

        """
        self.RS = RS
        self._base_point = base_point
        self._discriminant_points = None
        self._discriminant_points = self.discriminant_points()
        self._radii = self._compute_radii(kappa=kappa)

    def base_point(self):
        """Returns the base point of the monodromy group. Same point as the
        x-coordinate of the base place on the Riemann surface.

        """
        return self._base_point

    def _compute_radii(self, kappa=3./5.):
        """Returns the radii of the boudnding circles.

        Arguments
        ---------
        kappa : double
            A scaling factor between 0.5 and 1.0. `kappa = 1.0` means
            that the bounding circles are taken to be as large as
            possible without overlapping.

        """
        radii = []
        for bi in self._discriminant_points:
            dists = [numpy.abs(bi - bj) for bj in self._discriminant_points
                     if bi != bj]
            rho = numpy.min(dists)
            radius = rho*kappa/2.0
            radii.append(radius)

        return numpy.array(radii, dtype=numpy.double)

    def radius(self, bi):
        """Returns the raidus of the bounding circle around `bi`.

        Arguments
        ---------
        bi : complex
            A discriminant point of the algebraic curve.

        """
        return self._radii[self._discriminant_points == bi][0]

    def discriminant_points(self):
        """Returns the ordered discriminant points of the curve.

        The discriminant points are ordered by increasing argument with
        the base point.

        """
        if not self._discriminant_points is None:
            return self._discriminant_points

        f = self.RS.f
        x = self.RS.x
        y = self.RS.y
        p = sympy.Poly(f, [x,y])
        res = sympy.Poly(sympy.resultant(p, p.diff(y), y), x)

        # Compute the numerical roots. Since Sympy has issues balancing
        # between precision and speed we compute the roots of each
        # factor of the resultant.
        dps = sympy.mpmath.mp.dps
        rts = []
        for factor,degree in res.factor_list_include():
            rts.extend(factor.nroots(n=dps+3))
        rts = map(lambda z: z.as_real_imag(), rts)
        disc_pts = map(lambda z: sympy.mpmath.mpc(*z), rts)

        # Pop any roots that appear to be equal up to the set
        # multiprecision. Geometrically, this may cause two roots to be
        # interpreted as one thus reducing the genus of the curve.
        N = len(disc_pts)
        i = 0
        while i < N:
            k = 0
            while k < N:
                eq = sympy.mpmath.almosteq(disc_pts[i], disc_pts[k])
                if (k != i) and eq:
                    disc_pts.remove(disc_pts[k])
                    N -= 1
                else:
                    k += 1
            i += 1

        # sort the discriminant points first by argument with the base
        # point and then by distance from the base point.
        def cmp(z, a=None):
            return (sympy.mpmath.arg(z - a), sympy.mpmath.absmax(z - a))

        # determine the base point
        if self._base_point is None:
            a = min([numpy.complex(bi).real - 1 for bi in disc_pts])
            self._base_point = a

        disc_pts = sorted(disc_pts, key=lambda z: cmp(z, a=self._base_point))
        self._discriminant_points = numpy.array(disc_pts, dtype=numpy.complex)
        return self._discriminant_points

    def xpath_to_discriminant_point(self, bi):
        """Returns the xpath leading to the discriminant point `bi`.

        Arguments
        ---------
        bi : complex
            A discriminant point of the curve.

        """
        raise NotImplementedError('Implement in subclass.')

    def xpath_around_discriminant_point(self, bi, nrots=1):
        """Returns the xpath consisting of `nrots` rotations around the
        bounding circle of discriminant point `bi`.

        The sign of `nrots` indicates the sign of the direction.

        Arguments
        ---------
        bi : complex
            A discriminant point of the curve.
        nrots : integer (default `1`)
            A number of rotations around this discriminant point.

        """
        raise NotImplementedError('Implement in subclass.')

    def xpath_monodromy_path(self, bi, nrots=1):
        """Returns the xpath starting from the base point, going around the
        discriminant point `bi` `nrots` times, and returning to the base
        x-point.

        The sign of `nrots` indicates the sign of the direction.

        Arguments
        ---------
        bi : complex
            A discriminant point.
        nrots : integer (default `1`)
            A number of rotations around this discriminant point.

        Returns
        -------
        list
            An x-path representing the monodromy path with `nrots`
            rotations about the discriminant point `bi`.

        """
        # special case when going around infinity.
        if bi == sympy.oo:
            return self.xpath_around_infinity(nrots=nrots)

        xpath_to_bi = self.xpath_to_discriminant_point(bi)
        xpath_around_bi = self.xpath_around_discriminant_point(bi, nrots=nrots)
        xpath_from_bi = self.xpath_reverse(xpath_to_bi)
        xpath = xpath_to_bi + xpath_around_bi + xpath_from_bi
        return xpath

    def xpath_around_infinity(self, nrots=1):
        """Returns the xpath starting at the base point, going around infinity
        `nrots` times, and returning to the base point.

        This path is sure to not only encircle all of the discriminant
        points but also stay sufficiently outside the bounding circles
        of the points.

        Arguments
        ---------
        nrots : integer, (default `1`)
            The number of rotations around infinity.

        Returns
        -------
        RiemannSurfacePath
            The xpath encircling infinity.

        """
        xpath = []

        # determine the radius R of the circle, centered at the origin,
        # encircling all of the discriminant points and their
        # "protection" circles
        b = self.discriminant_points()
        R = numpy.abs(self.base_point())
        for bi in b:
            radius = self.radius(bi)
            Ri = numpy.abs(bi) + 2*radius  # to be safely away
            R = Ri if Ri > R else R

        # the path begins with a line starting the base point and ending
        # at -R.
        xpath.append((self.base_point(), -R))

        # the positive direction around infinity is equal to the
        # negative direction around the origin
        dtheta = -numpy.pi if nrots > 0 else numpy.pi
        for _ in range(abs(nrots)):
            xpath.append((R, 0, numpy.pi, dtheta))
            xpath.append((R, 0, 0, dtheta))

        # return to the base point
        xpath.append((-R, self.base_point()))
        return xpath

    def xpath_to_point(self, x):
        """Returns an xpath to an arbitrary point `x`.

        Arguments
        ---------
        x : complex
            A point on the complex x-sphere.

        """
        raise NotImplementedError('Implement in subclass.')

    def xpath_reverse(self, xpath):
        """Reverses an x-path.

        Useful for building the return path from a discriminant point to
        the base point.

        Arguments
        ---------
        xpath : list
            A list of tuples defining lines and semicircles in the
            complex x-plane.

        Returns
        -------
        list
            A list of tuples defining lines and semicircles in the
            complex x-plane representing the reverse of `xpath`.

        """
        reverse_xpath = []
        for data in xpath[::-1]:
            if len(data) == 2:
                z0, z1 = data
                data = (z1, z0)
            else:
                R, w, theta, dtheta = data
                theta += dtheta
                dtheta = -dtheta
                data = (R, w, theta, dtheta)

            reverse_xpath.append(data)
        return reverse_xpath

    def show_paths(self, ax=None, *args, **kwds):
        """Plots all of the monodromy paths of the curve.

        Arguments
        ---------
        ax : matplotlib.Axes
            The figure axes on which to plot the paths.

        Returns
        -------
        None
        """
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection

        # helper functions for plotting line, arcs, and xpaths
        def plot_line(xline, ax, *args, **kwds):
            t = numpy.linspace(0,1,16)
            z0, z1 = xline
            l = z0*(1-t) + z1*t
            ax.plot(l.real, l.imag, 'k', *args, **kwds)

        def plot_arc(xarc, ax, *args, **kwds):
            t = numpy.linspace(0,1,16)
            R, w, theta, dtheta = xarc
            a = R*numpy.exp(1.0j*(theta + dtheta*t)) + w
            ax.plot(a.real, a.imag, 'k', *args, **kwds)

        def plot_xpath(xpath, ax, *args, **kwds):
            for segment in xpath:
                if len(segment) == 2:
                    plot_line(segment, ax, *args, **kwds)
                elif len(segment) == 4:
                    plot_arc(segment, ax, *args, **kwds)

        # get the current axes / creat one if none is provided
        if ax is None:
            ax = plt.gca()

        # fill the bounding circles around each discriminant point
        a = self.base_point()
        b = self.discriminant_points()
        R = self._radii
        patches = []
        for bi, Ri in zip(b,R):
            circle = Circle((bi.real, bi.imag), Ri)
            patches.append(circle)

        p = PatchCollection(patches, alpha=0.4)
        ax.add_collection(p)

        # plot the base point and the discriminant points
        ax.plot(a.real, a.imag, 'ro', *args, **kwds)
        ax.plot(b.real, b.imag, 'bo', *args, **kwds)

        # plot the monodromy paths
        for bi in b:
            xpath = self.xpath_monodromy_path(bi)
            plot_xpath(xpath, ax, *args, **kwds)




class XPathFactoryAbel(XPathFactory):
    """An `XPathFactory` designed for constructing paths used in the Abel
    map. Based on the technique of Deconinck, van Hoeij, and Patterson [1]_.

    Attributes
    ----------
    See :py:class:`XPathFactory`.

    Methods
    -------
    See :py:class:`XPathFactory`. Additional methods and overloads are
    listed below.

    xpath_to_discriminant_point
        Returns an x-path from the base point to the bounding circle
        around a given discriminant point. (Implemented in subclasses.)
    xpath_circle_discriminant_point
        Returns an x-path parameterizing the bounding circle around a
        given discriminant point with start and end equal to the ending
        point of `xpath_to_discriminant_point()`. (Implemented in
        subclasses.)
    xpath_to_point
        Returns an x-path to an arbitrary point in the complex
        plane. (Implemented in subclasses.)
    xpath_build_avoiding_path
        Returns an xpath to an arbitrary point in the complex plane that
        avoids the bounding circles of the discriminant points.

    References
    ----------

    .. [1] B. Deconinck and M. Patterson, "Computing the Abel map",
    Physica D: Nonlinear Phenomena, v. 237, no. 24, pp. 3214--3232,
    2008.

    """
    def xpath_to_discriminant_point(self, bi):
        """Returns the xpath leading to the discriminant point `bi`.

        Arguments
        ---------
        bi : complex
            A discriminant point of the curve.

        """
        Ri = self.radius(bi)
        z = bi - Ri
        return self.xpath_build_avoiding_path(self.base_point(), z)

    def xpath_around_discriminant_point(self, bi, nrots=1):
        """Returns the xpath consisting of `nrots` rotations around the
        bounding circle of discriminant point `bi`.

        The sign of `nrots` indicates the sign of the direction.

        Arguments
        ---------
        bi : complex
            A discriminant point of the curve.
        nrots : integer (default `1`)
            A number of rotations around this discriminant point.

        """
        # note that the Deconinck, van Hoeij paths always meet the
        # bounding circle on the left
        Ri = self.radius(bi)
        dtheta = numpy.pi if nrots > 0 else -numpy.pi
        circle = [
            (Ri, bi, -numpy.pi, dtheta),
            (Ri, bi, 0, dtheta)
            ]
        return circle * abs(nrots)

    def xpath_to_point(self, x):
        """Returns an xpath to an arbitrary point `x`.

        .. todo::

            Implement this.

        Arguments
        ---------
        x : complex
            A point on the complex x-sphere.

        """
        raise NotImplementedError('Implement in subclass.')

    def xpath_build_avoiding_path(self, z0, z1):
        """Returns an xpath to `z` from `a` (default: base point) avoiding
        discriminant points as necessary.

        """
        # compute the set of discriminant points whose bounding circle
        # intersects the line from a to z
        xpath = []
        b = numpy.array([bi for bi in self.discriminant_points()
                         if self._intersects_disc_pt(z0, z1, bi)],
                        dtype=numpy.complex)

        # sort by increasing distance from z0
        b = b.tolist()
        b.sort(key=lambda bi: numpy.abs(bi - z0))
        for bi in b:
            # compute the intersection points of the segment from z0 to
            # z1 with the circle around bi.
            w0, w1 = self._intersection_points(z0, z1, bi)

            # compute the arc going from w0 to w1 avoiding the bounding
            # circle around bi.
            arc = self._avoiding_arc(w0, w1, bi)

            # add to the path and update the loop
            xpath.extend([(z0,w0), arc])
            z0 = w1

        # add the final line segment and return
        xpath.append((z0,z1))
        return xpath

    def _intersection_points(self, z0, z1, bi):
        """Returns the complex points `w0,w1` where the line from `z0` to `z1`
        intersects the bounding circle around `bi`.

        """
        # construct the polynomial giving the distance from the line
        # l(t), parameterized by t in [0,1], to bi.
        Ri = self.radius(bi)
        v = z1 - z0
        w = z0 - bi
        p2 = v.real**2 + v.imag**2
        p1 = 2*(v.real*w.real + v.imag*w.imag)
        p0 = w.real**2 + w.imag**2 - Ri**2  # solving |l(t) - bi| = Ri

        # find the roots of this polynomial and sort by increasing t
        p = numpy.poly1d([p2, p1, p0])
        t = numpy.roots(p)
        t.sort()

        w0 = v*t[0] + z0   # first intersection point
        w1 = v*t[1] + z0   # second intersection point

        return w0, w1

    def _avoiding_arc(self, w0, w1, bi):
        """Returns the arc `(radius, center, starting_theta, dtheta)`, from the
        points `w0` and `w1` on the bounding circle around `bi`.

        The arc is constructed in such a way so that the monodromy
        properties of the path are conserved.

        """
        # the angles of w0 and w1 around the circle tells us the length
        # of the arc connecting the points
        theta0 = numpy.angle(w0 - bi)
        theta1 = numpy.angle(w1 - bi)

        # the angles that w1 and bi make with w0 tell us whether the
        # path will go above bi or below bi
        phi1 = numpy.angle(w1 - w0)
        phii = numpy.angle(bi - w0)

        # go above (counterclockwise) if w1 lies over bi
        direction = numpy.sign(phii - phi1)
        if (theta0 < 0) and (theta1 > 0):
            theta0 += 2*numpy.pi
        if (theta0 > 0) and (theta1 < 0):
            theta1 += 2*numpy.pi
        dtheta = min(numpy.abs(theta0 - theta1),
                     numpy.abs(theta1 - theta0))
        dtheta = direction*dtheta

        # degenerate case when the path is colinear with bi: always go
        # above bi. this implies that discriminant points further away
        # are above those that are closer.
        if dtheta == 0:
            dtheta = -numpy.pi

        # add the path from z0 to w1 going around bi
        Ri = self.radius(bi)
        arc = (Ri, bi, theta0, dtheta)
        return arc


    def _intersects_disc_pt(self, z0, z1, bi):
        """Returns `True` if the line from `z0` to `z1` intersects the bounding
        circle around the discriminant point `bi`.

        """
        # first check the perpendicular distance from bi to the line
        # passing through z0 and z1
        direction = numpy.sign(numpy.angle(z1-z0) - numpy.angle(bi-z0))
        normv = numpy.abs(z1-z0)
        v = 1.0j*direction*(z1 - z0)
        r = z0 - bi

        # degenerate case: the line through z0 and z1 crosses bi. in
        # this case just check if the branch point lies in between
        if direction == 0:
            if (numpy.abs(bi - z0) <= normv) and (numpy.abs(bi - z1) <= normv):
                return True
            else:
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
        if (numpy.abs(w - z0) <= normv) and (numpy.abs(w - z1) <= normv):
            return True
        else:
            return False
