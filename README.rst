############################################
nphelper - convenient numpy helper functions
############################################

This package contains some convenient helper function for numpy.  Nothing
fancy, but quite useful.
It works with python 2.7, 3.4, and 3.5.

Usage / Features
================

Create block arrays (in matlabs's ``[A A; B B]`` spirit):

.. code:: python

    >>> from nphelper import block
    >>> A = np.array([[1, 2, 3]])
    >>> B = np.array([[2, 3, 4]])
    >>> C = np.ones((2, 6))
    >>> block([[A, B], [C]])
    array([[1, 2, 3, 2, 3, 4],
          [[1, 1, 1, 1, 1, 1],
          [[1, 1, 1, 1, 1, 1]])

Compute the cartesian product (similar to ``itertools.product``):

.. code:: python

    >>> from nphelper import cartesian_product
    >>> cartesian_product([[1, 2], [3, 4]])
    array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]])

Easily multiply many arrays without the cubersome ``dot`` syntax. It's also
much faster than dot because it selects the fastest evaluation order.
(This is part of numpy 1.10.0.)

.. code:: python

    >>> from nphelper import multi_dot
    >>> A = np.random.random((10000, 100))
    >>> B = np.random.random((100, 1000))
    >>> C = np.random.random((1000, 5))
    >>> D = np.random.random((5, 333))
    >>> # Sick of this?
    >>> np.dot(np.dot(np.dot(A, B), C), D)  # doctest: +SKIP
    >>> # Or this?
    >>> A.dot(B).dot(C).dot(D)  # doctest: +SKIP
    >>> # Use multi_dot
    >>> multi_dot([A, B, C, D])  # doctest: +SKIP

- TODO along, maxalong, minalong, sumalong, meanalong, stdalong, varalong

Dependencies
============

- numpy 1.8

Install
============

Simply install via ``pip``::

    pip install nphelper

Tests
------

Run ``tox`` to run the tests for python 2.7, 3.4, and 3.5::

    tox

You might need addiotional dependencies to run the tests.

- ``py.test``,
- ``nose``,
- ``python2.7-dev``,
- ``python3.4-dev``, and
- ``python3.5-dev``.
