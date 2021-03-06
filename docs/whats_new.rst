:orphan:

.. _whats_new:


What's new?
===========

Here we list a changelog of EEGIO.

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: eegio

.. _current:

Current
-------

Changelog
~~~~~~~~~

Bug
~~~

- Fixed `MANIFEST.in` file, `Makefile` and docstrings to adhere to pydocstyle. By `Adam Li`_ (`#11 <https://github.com/adam2392/eegio/pull/11>`_)

.. _changes_0_1:

Version 0.1
-----------

Changelog
~~~~~~~~~

- Add support for mne-bids and a corresponding refactoring to align with: :func:`write_raw_bids` and :func:`read_raw_bids` by `Adam Li`_ and `Patrick_Myers`_ (`#8 <https://github.com/adam2392/eztrack/pull/8>`_)

Bug
~~~

- Recreated datasets in `data/`, so that it would make tests pass that depend on it. By `Adam Li`_ (`#10 <https://github.com/adam2392/eztrack/pull/10>`_)

API
~~~

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* Adam Li
* Patrick Myers

.. _Adam Li: https://github.com/adam2392
.. _Patrick Myers: https://github.com/pmyers16
