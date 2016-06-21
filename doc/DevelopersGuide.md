# Abelfunctions Developer's Guide

Want to contribute to Abelfunctions? Excellent! This document presents some
useful information on how to get started. This guide assumes that, at the very
least, you know how to write [Sage](http://www.sagemath.org) code, how to use
[git](http://git-scm.com), and that you have a [GitHub](http://www.github.com)
account.

Found a bug but don't want to fix it yourself? Simply create an Issue over on
the [Issues Page](https://github.com/abelfunctions/abelfunctions/issues).

### Contents

* [Workflow](#workflow)
* [Abelfunctions Overview](#abelfunctions-overview)
* [Writing Tests](#writing-tests)

## Workflow

Use the suggested workflow below when contributing to Abelfunctions. This
workflow follows the standard
[Fork-Commit-Pull Request workflow](https://guides.github.com/activities/contributing-to-open-source/)
used in many GitHub projects. To begin, before making any changes or
improvements, follow the instructions at
[GitHub - Fork a Repo](https://help.github.com/articles/fork-a-repo/) to create
a personal "fork" of Abelfunctions on GitHub. You only need to do this once.

1. **Create a new branch:**

   Create a new branch in which to commit your changes. This way, you always
   have a clean copy of the code in `master`.
   
   ```
   $ git checkout -b my-branch-name
   ```
   
2. **Make changes:**

   Remember to add tests to the test suite, especially when fixing a bug. See
   [Writing Tests](#writing-tests), below.

3. **Test changes:**

   Now that you've made your changes and written your tests, build and test
   Abelfunctions. This will run the entire test suite, checking for any
   propagating issues that may have cropped up.

   ```
   $ sage setup.py build_ext --inplace
   $ sage runtests.py
   ```

4. **Commit:**

   Commit your changes to your working branch. Repeat steps 2-4 if there are
   multiple "chunks" of work that need to be done. Ideally, each commit is for a
   separate unit of work. Also ideally, each commit results in working code.
   
5. **Push and create pull request:**

   Push your changes to a remote branch.
   
   ```
   $ git push origin my-branch-name
   ```

   Then, create a "Pull Request" of this remote branch against the main
   Ablefunctions repo. Read more about Pull Requests at
   [GitHub - Using pull requests](https://help.github.com/articles/using-pull-requests/).
   If during discussion some additional commits are needed to call the issue or
   enhancement "resolved" simply push them to the same remote branch.

## Abelfunctions Overview

Here we describe the library from a high-level perspective, outlining the
different major components and how they interact with each other. For details
about a particular module, class, or function, please see the source code of the
corresponding module.

*(Aside: I still need to figure out how to get Sphinx to interface with external
Sage packages and libraries, such as Abelfunctions, how how best to host the
output HTML documentation.)*

**TODO:** provide high-level library description.

```
abelfunctions/
  abelfunctions/    # Sage library source files
  doc/              # Abelfunctions documentation
  example/          # Example usage scripts
  notebooks/        # Demonstration notebooks
  .travis.yml       # TravisCI test configuration
  runtests.py       # Script for running Ablfunctions tests
  setup.py          # Installation script
```

## Writing Tests

Abelfunctions uses the built-in Python
[unittest](https://docs.python.org/2.7/library/unittest.html) module for
automated testing. The quickest way to learn how to write unit tests is to
follow the syntax in an existing test suite module, such as the one for testing
Riemann surface paths:
[`test_riemann_surface_path.py`](https://github.com/abelfunctions/abelfunctions/blob/master/abelfunctions/tests/test_riemann_surface_path.py).

To run the test suite simply execute

```
$ sage runtests.py
```

from the top-level directory. To only run the test suite on certain modules you
can optionally supply a partial filename. For example,

```
$ sage runtests.py puiseux
```

will run the test suite on every module matching the expression
"`test_*puiseux*.py`".

New test suite modules are automatically detected as long as the module name
begins with `test_` and lives in a `test/` directory. When creating a new
`TestCase` object in either a new module or an existing be sure to inherit from
`abelfunctions.tests.test_abelfunctions.AbelfunctionsTestCase`.

Every remote branch and pull request will create a new
[TravisCI instance](https://travis-ci.org/abelfunctions/abelfunctions) so that
you'll be notified if your contribution passes tests.
