import pytest
import numpy

from abelfunctions.riemann_theta.radius import (
    radius1,
    radius2,
    radiusN,
)

@pytest.mark.parametrize('deriv', [
    [1,0,0], 
    [0,1,0], 
    [0,0,1], 
    [0.1 + 0.2j, 0.3+0.4j, 0.5+0.6j]
])
@pytest.mark.parametrize('eps', [1e-8, 1e-14])
def test_vs_radius1(deriv, eps):
    # genus 3 example
    T = numpy.diag([1,2,3]) + numpy.diag([4,5], k=1) + numpy.diag([6], k=2)
    rho = 0.9  # any rho is fine
    with pytest.warns(DeprecationWarning):
        R1 = radius1(eps, rho, 3, T, deriv)
    R2 = radiusN(eps, rho, 3, T, [deriv])
    assert R1 == pytest.approx(R2)


@pytest.mark.parametrize('deriv', [
    [[1,0,0], [1,0,0]], 
    [[0,1,0], [1,0,0]], 
    [[0,0,1], [1,0,0]], 
    [[1,0,0], [0,1,0]], 
    [[0,1,0], [0,1,0]], 
    [[0,0,1], [0,1,0]], 
    [[1,0,0], [0,0,1]],
    [[0,1,0], [0,0,1]],
    [[0,0,1], [0,0,1]],
    [[0.1 + 0.2j, 0.3+0.4j, 0.5+0.6j], [0.7 + 0.8j, 0.9+1.0j, 1.1+1.2j]]
])
@pytest.mark.parametrize('eps', [1e-8, 1e-14])
def test_vs_radius2(deriv, eps):
    # genus 3 example
    T = numpy.diag([1,2,3]) + numpy.diag([4,5], k=1) + numpy.diag([6], k=2)
    rho = 0.9  # any rho is fine
    with pytest.warns(DeprecationWarning):
        R1 = radius2(eps, rho, 3, T, deriv)
    R2 = radiusN(eps, rho, 3, T, deriv)
    assert R1 == pytest.approx(R2)

