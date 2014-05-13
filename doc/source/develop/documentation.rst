Writing Documentation
=====================

This document describes how to contribute to abelfunctions's
documentation. All of the documentation is written using the `Sphinx
<http://sphinx-doc.org/>`_ Python package.

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
