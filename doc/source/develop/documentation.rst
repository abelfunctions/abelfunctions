Writing Documentation
=====================

This document describes how to contribute to Abelfunctions's documentation. The
following packages are required to build the documentation, in addition to the
usual Abelfunctions :ref:`prerequisites <prerequisites>`:

* `sphinx <http://sphinx-doc.org>`_
* `releases <http://releases.readthedocs.org/en/latest/index.html>`_


Documentation Structure
-----------------------

The abelfunctions documentation is found in the ``doc`` directory
located at the top of the file tree of the code repository. There are
several files and subdirectories within but the most important one is
the ``doc/source`` directory.

You'll find several files written in the reStructuredText (reST)
syntax. It's a simple and easy to use syntax similar to wiki or markdown
format. You can find a `reST primer on the Sphinx website
<http://sphinx-doc.org/rest.html>`_.

There are several subdirectories for specific types of documentation.

* ``doc/source/applications``: contains a list of applications
  demonstrating how to use abelfunctions to solve various problems,
* ``doc/source/reference``: auto-generated documentation of the
  abelfunctions API,
* ``doc/source/develop``: instructions on how to contribute to abelfunctions as
  well as some in-depth discussion on the structure and design of the
  software.

Building the Documentation
--------------------------

To build the documentation on your own machine, simply execute ::

    $ cd abelfunctions/doc
    $ make html

if you want to build the documentation website or ::

    $ make pdf

for a ``.pdf`` version. Also note that the documentation is auto-updated
at http://abelfunctions.cswiercz.info whenever a commit is pushed to the
master branch of the GitHub repository
https://github.com/cswiercz/abelfunctions .

Adding A New Page
-----------------

Adding a new page to the documentation is easy. For example, let's say
we wish to add a new application to the :doc:`../applications/index` page.

1. Create a ``mynewapplication.rst`` document in
   ``/doc/source/applications``. (Of course, using a more appropriate
   file name.)

2. Make this new document appear in the documentation by adding
   ``mynewapplication`` to the ``applications/index.rst`` table of
   contents tree (toctree) like so: ::

       .. toctree::
          :maxdepth: 1

          <other, already existing documents>
          mynewapplication

Rebuild the documentation to see a link to your new document appear in
the table of contents and root page.

Source-Level Documenation
-------------------------

Every module of abelfunctions (the code itself) in abelfunctions should
be documented using the `numpydoc style guide
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_. That
is, every function, class, and module itself should have a docstring
associated with it describing to both the user and the developer its
purpose, use, and any other pertinent information. The numpydoc style in
particular was chosen because its legibility in both the code itself and
the auto-generated documentation webpage.

To add an abelfunctions module ``mymodule`` to this documentation

1. Add the text ``mymodule`` to the table of contents tree in
   ``doc/source/reference/index.rst`` as if you were adding a new page.
2. Add a file ``mymodule.rst`` to ``doc/source/reference`` with the text ::

      .. automodule:: abelfunctions.mymodule
          :members:
          :show-inheritance:

For a decent example of code documentation syntax and layout see
``abelfunctions/puiseux.py``.

In brief, every function and class method should have a docstring with
the following fields. The details are given in the numpydoc guide: ::

    def myfunc(a, b):
        r"""A brief description of the function that fits in one line.

        More in-depth description of the function and its behavior that
        can span multiple lines. You can include mathematical
        expressions :math:`f(x,y) = 0` and notes

        .. note::

            This is a note.

        Parameters
        ----------
        a : the type of a
            Description of ``a`` as a function parameter.
        b : the type of b
            Description of ``b`` as a function parameter.

        Returns
        -------
        the type of the return
            A description of the return value.

        Raises
        ------
        (optional) If this function is desiged to raise errors, describe
        the errors and which conditions invoke them.

        Algorithm
        ---------
        (optional) Brief explanation of the algorithm used in this func.

        References
        ----------
        (optional) References for the source of the algorithm.

        """

Classes should have docstrings with a similar structure (though without
a ``Returns`` field, for instance) with the fields ``Attributes`` and
``Methods``: ::

    class MyClass(object):
        r"""A brief description of the class that fits in one line.

        More in-depth description of the class and its purpose that can
        span multiple lines.

        Attributes
        ----------
        myattr : type of attribute
            Description of attribute

        Methods
        -------
        mymethod1
        mymethod2
        mymethod3

        """

Again, consult the `numpydoc style guide
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for more information on docstring syntax and structure.
