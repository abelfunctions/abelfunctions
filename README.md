# Abelfunctions

[![Gitter](https://badges.gitter.im/abelfunctions/abelfunctions.svg)](https://gitter.im/abelfunctions/abelfunctions?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Build Status](https://travis-ci.org/abelfunctions/abelfunctions.svg?branch=master)](https://travis-ci.org/abelfunctions/abelfunctions)

A [Sage](http://www.sagemath.org) library for computing with Abelian functions, Riemann surfaces, and algebraic curves. Abelfunctions is the Ph.D. thesis work of [Chris Swierczewski](http://www.cswiercz.info). (GitHub: [cswiercz](https://github.com/cswiercz)).  Abelfunctions requires Sage 8.0 or later.

```python
sage: from abelfunctions import *
sage: R.<x,y> = QQ[]
sage: X = RiemannSurface(y**3 + 2*x**3*y - x**7)
sage: X.riemann_matrix()
array([[-1.30901699+0.95105652j, -0.80901699+0.58778525j],
       [-0.80901699+0.58778525j, -1.00000000+1.1755705j ]])
sage: P = X(0)[0]; P
(t, 1/2*t^4 + O(t^7))
sage: AbelMap(P)
array([-0.29124012+0.64492948j, -0.96444625+1.1755705j ])
sage: gamma = X.path(P)
sage: gamma.plot_x(); gamma.plot_y(color='green');
```
![x-projection of path](https://raw.githubusercontent.com/abelfunctions/abelfunctions/master/doc/img/xpath.png)
![y-projection of path](https://raw.githubusercontent.com/abelfunctions/abelfunctions/master/doc/img/ypath.png)

## Documentation and Help

For installation instructions, tutorials on how to use this software, and a complete reference to the code, please see the [Documentation](https://github.com/abelfunctions/abelfunctions/blob/master/doc/README.md). You can also post questions in the [Abelfunctions chat room](https://gitter.im/abelfunctions/abelfunctions)

Please report any bugs you find or suggest enhancements on the [Issues Page](https://github.com/cswiercz/abelfunctions/issues).

## Extensions to Abelfunctions
- [CyclePainter](https://github.com/markopuza/abelfunctions-cyclepainter) - A tool for building custom paths on Riemann surfaces

## Citing this Software

> C. Swierczewski et. al., *Abelfunctions: A library for computing with Abelian
  functions, Riemann surfaces, and algebraic curves*,
  `http://github.com/abelfunctions/abelfunctions`, 2017.

BibTeX:

    @misc{abelfunctions,
      author = {C. Swierczewski and others},
      title = {Abelfunctions: A library for computing with Abelian functions, Riemann surfaces, and algebraic curves},
      note= {\tt http://github.com/abelfunctions/abelfunctions},
      year = 2017,
    }
