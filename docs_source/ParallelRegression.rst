.. module:: ParallelRegression

ParallelRegression API
======================

This page is automatically generated from the source code using Sphinx.  Source code documentation follows Numpy style guidelines, not all of which is supported by Sphinx, even when using the napoleon extension.

Functions
---------

.. automodule:: ParallelRegression
    :members: vCovMatrix, FStatistic, syncText, val_if_present, has_term, mask_brackets, masked_dict, masked_iter, masked_split, terms_in, formulas_match, termString
    :show-inheritance:
   
setList( )
----------

.. autoclass:: setList
   :show-inheritance:
   
   **Attributes**
   
   lastSIOutcome : bool
       True if the most recent call to .__setitem__( ) resulted in adding a value, or False if the call would have resulted in a duplicate value.
   
   **Methods** (see :class:`set` & :class:`list`)
   
   * add (alias for .append( ))
   * append (returns True if it results in adding a value, or False otherwise)
   * difference
   * discard (returns True if it resulted in removing a value, or False otherwise)
   * extend (alias for .update( ))
   * intersection
   * issubset
   * issuperset
   * pop
   * symmetric_difference
   * union
   * update (returns the number of new values added to the setList( ))
   * *and list methods inherited from UserList( )*

.. autoattribute:: setList.as_fsets
	
typedDict( )
------------

.. autoclass:: typedDict
   :show-inheritance:
   :members: __init__, itemLength, keys, union_update, pop, update

categorizedSetDict( )
---------------------

.. class:: categorizedSetDict
   
   Bases: :class:`typedDict`
   
   Ordered sets stored in a dict( ) in which each set and each set member potentially belongs to one or more category.

   Sets and set members can be retrieved by category. Categories designated as mutually exclusive restrict category membership to sets and set members without conflicting categories, when category membership is established via the .set_category( ) method.
   
   **Attributes**
   
   mutually_exclusive : set
       Set of frozensets of categories, where each category in a frozen set and all other categories in the same frozenset are are considered mutually exclusive.
   
   **Notes**
   
   Categories are assumed to be identified by strings. Code is only tested using string-identified categories. However, this is not strictly enforced.

.. automethod:: categorizedSetDict.__init__

.. automethod:: categorizedSetDict.__setitem__

.. automethod:: categorizedSetDict.del_category
   
.. automethod:: categorizedSetDict.get_categories

.. automethod:: categorizedSetDict.is_a

.. automethod:: categorizedSetDict.is_None

.. automethod:: categorizedSetDict.items_categorized

.. automethod:: categorizedSetDict.keys_categorized

.. automethod:: categorizedSetDict.make_mutually_exclusive

.. automethod:: categorizedSetDict.pop

.. automethod:: categorizedSetDict.set_categories

.. automethod:: categorizedSetDict.set_category

.. automethod:: categorizedSetDict.values_categorized

termSet( )
----------

.. class:: termSet
   
   Bases: :class:`categorizedSetDict`
   
   Manages a set of terms, each of which might have multiple representations.
   
   **Categories**
   
   dummy
       "Dummy" terms or representations of terms in which there are only two values.
	   
   Y
       Terms that are used on the LHS of formulas instead of the RHS.  Terms that are sometimes used on the LHS and sometimes used on the RHS are not supported.
	   
   required_X
       Terms that must be included on the RHS of all formulas derived from this termSet( ).
	   
   T
       RHS terms representing time/trend.

   **Read-Only Attributes**
   
   The following read-only attributes return setList( )s with the respective terms:
   
   * W_term_set (all terms)
   * Y_term_set
   * X_required_set
   * dummy_term_set
   * real_term_set (terms not categorized as dummy terms)
   * other_terms (terms neither categorized as Y terms nor as required X terms)

.. method:: termSet.__init__( terms, dterms, T=None )
   
   Creates a termSet( ) instance.
   
   :param dict terms: Dictionary in which each term is represented by a key for which the value is a sequence of forms in which the term might occur.  Ex: {'X': ['X', 'ln(X)']}.  Entries for which the value is a single string, e.g. {'d': 'd'},  will be treated as dummy terms.  To avoid this when a term has only one form, enclose the string in a list, e.g. {'X': ['X']}.
   :param string T: String identifying a single term that represents time/trend.

.. method:: termSet.__init__( formulas )
   
   Creates a termSet( ) instance.
   
   :param iterable formulas: Iterable of a formula strings from which to extract the terms and their forms.  Ex: ['y ~ x**2 + x', 'ln(y) ~ ln(x)']

.. automethod:: termSet.changeT

.. automethod:: termSet.require

.. automethod:: termSet.Y

.. automethod:: termSet.dummy

mathDataStore( )
----------------

.. autoclass:: mathDataStore
   :show-inheritance:
   :members:

mathDictMaker( )
----------------

.. autoclass:: mathDictMaker
   :show-inheritance:
   :members: make
   

mathDict( ) and additional supporting classes
---------------------------------------------

.. class:: mathDict
   
   Bases: :class:`object`
   
   **Attributes**
   
   hypothesis : See :class:`mathDictHypothesis`.

.. automethod:: mathDict.__init__

.. automethod:: mathDict.__getitem__( index )

.. automethod:: mathDict.add

.. automethod:: mathDict.add_from_RHS

.. automethod:: mathDict.columns

.. automethod:: mathDict.config_to_dict

.. automethod:: mathDict.crosspower

.. automethod:: mathDict.crossproduct

.. automethod:: mathDict.get_column

.. automethod:: mathDict.mask_all

.. automethod:: mathDict.power

.. automethod:: mathDict.rows

.. automethod:: mathDict.set_mask

.. automethod:: mathDict.shape

.. automethod:: mathDict.unmask_all

.. autoclass:: mathDictConfig
   :show-inheritance:
   :members:

.. autoclass:: mathDictHypothesis
   :show-inheritance:
   :members:

Exceptions and Warnings
-----------------------

.. autoclass:: CategoryError
   :show-inheritance:
   :members:
	
.. autoclass:: UnsupportedColumn
   :show-inheritance:
   :members:

.. autoclass:: mathDictKeyError
   :show-inheritance:
   :members:

.. autoclass:: RankError
   :show-inheritance:
   :members: