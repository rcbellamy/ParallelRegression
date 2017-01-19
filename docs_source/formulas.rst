.. module:: ParallelRegression

Working with terms and formulas
===============================

Parallel Regression provides a termSet( ) class that tracks a set of terms and for each term tracks the multiple forms in which the term might occur.  For example, a set of terms might include `y`, `x1`, and `x2`, and `x1` might occur in some places as `x1` and elsewhere as `x1**2 + x1` or as `ln(x1)`.

Parallel Regression also provides a set of tools for interpreting and manipulating formula strings based only on the characters outside of brackets by masking the bracket contents.

Using termSet( ) to track terms in their multiple forms
-------------------------------------------------------

termSet( ) uses a dict( ) structure to track a set of terms as dict( ) keys, with each of the various forms in which the term occurs listed in a single ordered set value associated with that key.  termSet( ) can also track which terms occur on the LHS of formulas, which are dummy terms, and which must be included on the RHS of all formulas generated using that termSet( ).  Whole terms can be associated with any of these categories, and individual forms in which a term can occur can also be associated with these categories.

The role of termSet( ) is simply tracking metadata about a set of terms.  It is up to the developer or analyst to incorporate the use of such metadata into analysis.  mathDict makes use of knowledge regarding which columns are dummy variables in order to allow analysts or developers to attempt to add squares of all columns in a matrix to a hypothesis about that matrix without producing a test matrix of less than full rank.  Tracking this information and using a logic-based test is substantially more efficient than mathematically evaluating the effect of a column on the rank of the matrix.  A termSet( ) can be linked to a mathDict( ) when directly instantiating a mathDict( ) or by assignment to the mathDict( ).terms attribute.

:class:`termSet` ( ) is a subclass of the more broadly-applicable :class:`categorizedSetDict` ( ) class.

Working with bracket masks
--------------------------

Formula strings are a convenient, human-readable manner in which to store information about a mathematical formula.  Existing string and regular expression methods provide most of the functionality needed for algorithmically acting on such formulas, except that these tools do not provide any recognition of brackets nesting strings within other strings.  Parallel Regression provides this functionality by creating a masked string in which bracketed substrings (including the brackets) are replaced with same-length sequences repeating '_'.  Various functions then allow some regular expression and string methods to be used on the masked string and the substring content then recovered from the original string. ::

    def testMaskBrackets(self):
        formula = 'ln( Hello )'
        masked = mask_brackets( formula )
        self.assertEqual( formula, 'ln( Hello )' )
        self.assertEqual( masked, 'ln_________' )
        square_squig = 'func{ #tag } + name[index]'
        masked = mask_brackets( square_squig )
        self.assertEqual( square_squig, 'func{ #tag } + name[index]' )
        self.assertEqual( masked, 'func________ + name_______' )
        nested = '( brackets ( nested {properly} ) ) // { text{a + b } }'
        masked = mask_brackets( nested )
        self.assertEqual( nested, '( brackets ( nested {properly} ) ) // { text{a + b } }' )
        self.assertEqual( masked, '__________________________________ // ________________' )
        nestwo = ' +( nested { incorrectly] ) + ( again ( ) '
        masked = mask_brackets( nestwo )
        self.assertEqual( nestwo, ' +( nested { incorrectly] ) + ( again ( ) ' )
        self.assertEqual( masked, ' +_________________________ + ( again ___ ' )

:func:`mask_brackets` is used to create the masked formula.  :func:`masked_dict` and :func:`masked_iter` take the original string and a regular expression match object where the regular expression matching was performed on the masked formula in order to recover the masked contents, and :func:`masked_split` provides str.split( ) functionality by using both the masked and unmasked versions of the formula.  See the source code or the API documentation for complete details. ::

    def testMaskedDict(self):
        powerpattern = re.compile( r'''\ *(?P<column_name>[a-zA-Z_][a-zA-Z_0-9]*(?![a-zA-Z_0-9\(\{\[]))
                             (?:\ *\*\*\ *(?P<power>[0-9]+)\ *)?''', re.X )
        formula    = ' log( (a + b) / c**3 ) ** 2'
        masked     = mask_brackets( formula )
        mobj_power = powerpattern.fullmatch( masked )
        self.assertNotEqual( mobj_power, None )
        mobj_dict  = masked_dict( formula, mobj_power )
        self.assertEqual( mobj_dict['column_name'], 'log( (a + b) / c**3 )' )
        self.assertEqual( mobj_dict['power'],       '2' )