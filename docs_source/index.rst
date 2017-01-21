.. Parallel Regression documentation master file, created by
   sphinx-quickstart on Sun Jan 15 23:08:41 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ParallelRegression
==================

ParallelRegression is a set of Python tools for using parallel processes to analyze a data set in shared memory.  mathDict is a set of tools for assembling a matrix in a single block of shared memory and then creating different matrix views that combine columns in shared memory with process-local columns for analysis.  termSet( ) and the more generic categorizedSetDict( ) and setList( ) are classes that facilitate tracking of metadata regarding the data set being analyzed.  For example, regression in floating point mathematics can be sensitive to the ordering of terms.  Using tools built on the ordered set class setList( ) to track term metadata facilitates the reproducability of results.  ParallelRegression also includes functions that simplify working with strings that contain bracketed substrings, such as formulas.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   mathDict
   formulas
   support
   ParallelRegression

Source Code
-----------

* `mathDict Example Source Code <mathDict_example.html>`_
* `ParallelRegression Module Source Code <_modules/ParallelRegression.html>`_