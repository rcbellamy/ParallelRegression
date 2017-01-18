mathDict Support Classes
========================

setList( )
----------

setList( ) implements many common set operation methods of Python's set( ) type and prevents duplicate values.  However, it uses a list to store the set members, allowing the order in which members are listed to be preserved and allowing non-hashable set members.  It also supports all list( ) methods.

setList( ) quietly avoids duplicate values by declining to add a redundant value without raising an exception when an attempt to add a duplicate is made.  Whether or not an attempt to add a redundant value was made can be checked using the .lastSIOutcome( ) ("last .__\ **s**\ et\ **i**\ tem__( ) outcome") attribute, and by comparing the number of values added (the return value) to the number attempted in an extend or update operation::

    def testAddToSetList_asifList(self):
        sl = setList( [123, 'abc'] )
        self.assertFalse( sl.append( 123 ) )
        sl[0] = 'abc'
        self.assertFalse( sl.lastSIOutcome )
        sl[2] = 'xyz'
        self.assertTrue( sl.lastSIOutcome )
        self.assertTrue( sl.append( 789 ) )
        self.assertEqual( sl.extend( ['abc', 'mno', 'xyz'] ), 1 )
        self.assertSequenceEqual( sl, [123, 'abc', 'xyz', 789, 'mno'] )
        
    def testAddToSetList_asifSet(self):
        sl = setList( [123, 'abc'] )
        self.assertFalse( sl.add( 123 ) )
        self.assertTrue( sl.add( 789 ) )
        self.assertEqual( sl.update( ['abc', 'mno', 'xyz'] ), 2 )
        self.assertSequenceEqual( sl, [123, 'abc', 789, 'mno', 'xyz'] )

categorizedSetDict( )
---------------------

categorizedSetDict( ) is a dict( ) of setList( ) objects in which both setList( )s and their members can be categorized.  It enforces mutually-exclusive categories and provides versions of .keys( ), .values( ), and .items( ) that only retrieve setList( ) members associated with a specified category: .keys_categorized( ), .values_categorized( ), and .items_categorized( ).  See the source code or API documentation for details.

categorizedSetDict( ) was created to provide the functionality needed for termSet( ) and is a subclass of typedDict( ). ::

    def setUp(self):
        self.catSD = categorizedSetDict( )
        self.catSD['nums']    = ([1, 2, 3, 4, 5],
                                 {'numerical',})
        self.catSD['numbers'] = (['one', 'two', 'three', 'four', 'five'],
                                 [set( ),
                                  {'multiple', 'even'},
                                  {'multiple'},
                                  {'multiple', 'even'},
                                  {'multiple'}])
        self.catSD['food']    = (['apple', 'bananas', 'cucumber', 'dates'],
                                 [{'apples'},
                                  {'multiple'},
                                  None,
                                  {'multiple'}])
        self.catSD['sayings'] = (['fair and square', 'two birds with one stone', 'an apple a day'],
                                 [{'even'},
                                  {'multiple'},
                                  {'apples'}])

    def testGetCategory(self):
        self.assertSetEqual( self.catSD.get_categories( key='nums' ),
                             {'numerical',} )
        self.assertSetEqual( self.catSD.get_categories( key='numbers', value='one' ),
                             set( ) )
        self.assertSetEqual( self.catSD.get_categories( key='food', value='cucumber' ),
                             set( ) )
		####
        self.catSD.set_category( 'words', key='numbers' )
        self.catSD.set_category( 'numerical', key='numbers' )
        self.assertSetEqual( self.catSD.get_categories( key='numbers' ),
                             {'numerical', 'words'} )
        self.assertSetEqual( self.catSD.get_categories( key='numbers', value='two' ),
                             {'multiple', 'even', 'numerical', 'words'} )
							 
typedDict( )
------------

typedDict( ) restricts entries to objects of a specified type, and handles default values and missing keys in a different manner than Python's dict( ) type.  The defualt value for entries created for missing keys can be specified in the form of an object (as opposed to a type), which is then deepcopy( )d when creating new default entries.  Entries can be accessed via both integer indexing and string keys, and the length of the item associated with any given key can be checked without creating a new default entry if the key does not exist by using the .itemLength( ) method.  Additional features designed for specific situations are documented in the source code.