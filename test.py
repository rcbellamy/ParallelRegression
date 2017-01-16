import unittest
import ParallelRegression as tsm
import ParallelRegression as PR
from ParallelRegression import TestCase
from io import BytesIO, StringIO
import numpy as np
import array
from multiprocessing import sharedctypes
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

class TestGenericFunctions(tsm.TestCase):
    def testval_if_present(self):
        class testFixture(object):
            NoneVal = None
            number = 123
        testFix = None
        self.assertEqual( tsm.val_if_present( testFix, None, 'ABCdef' ), 'ABCdef' )
        testFix = testFixture( )
        self.assertEqual( tsm.val_if_present( testFix, None, 'ABCdef' ), testFix )
        self.assertEqual( tsm.val_if_present( testFix, 'arbitrary', 'ABCdef' ), 'ABCdef' )
        self.assertEqual( tsm.val_if_present( testFix, 'arbitrary' ), None )
        self.assertEqual( tsm.val_if_present( testFix, 'NoneVal', 'ABCdef' ), 'ABCdef' )
        self.assertEqual( tsm.val_if_present( testFix, 'NoneVal' ), None )
        self.assertEqual( tsm.val_if_present( testFix, 'number', 'ABCdef' ), 123 )
        self.assertEqual( tsm.val_if_present( testFix, 'number', ), 123 )
        
    def testTermString(self):
        termList = ['NN','x1','x2','x3','x4','x5','y','T']
        ts1 = 'x1, x2, x3, x4, y'
        ts2 = 'x1, x3, y, T'
        f1 = 'y ~ I(x1**3) + x3 + I(x2**2) + x4'
        f2 = 'ln(y) ~ I(x1**3) + x3 + I(x2**3) + x4'
        f3 = 'ln(y) ~ I(x1**3) + x3 + T'
        f4 = 'x3 ~ ln(y) + I(x1**3) + T'
        self.assertEqual( tsm.termString( f1, termList ), ts1 )
        self.assertEqual( tsm.termString( f2, termList ), ts1 )
        self.assertEqual( tsm.termString( f3, termList ), ts2 )
        self.assertEqual( tsm.termString( f4, termList ), ts2 )
        
    def testHasTerm(self):
        formula = 'y~ x1+ x2 + x3-x4 + ln(x6) + x55'# + I(x6**2) *x7'
        self.assertTrue( tsm.has_term( formula, 'y' ) )
        self.assertTrue( tsm.has_term( formula, 'x1' ) )
        self.assertTrue( tsm.has_term( formula, 'x2' ) )
        self.assertTrue( tsm.has_term( formula, 'x3' ) )
        self.assertTrue( tsm.has_term( formula, 'x4' ) )
        self.assertTrue( tsm.has_term( formula, 'x6' ) )
        self.assertFalse( tsm.has_term( formula, 'x5' ) )
        self.assertTrue( tsm.has_term( formula, 'x55' ) )
        self.assertFalse( tsm.has_term( formula, 'ln' ) )
        
    def testPatsyTerms(self):
        formula1 = 'x1 + x2 + x1:x3 + I(x2**2):x3 + a{b+c(d)}+e[f+g]'
        f1_terms = { frozenset(['x1']), frozenset(['x2']), frozenset(['x1', 'x3']), frozenset(['I(x2**2)', 'x3']), frozenset(['a{b+c(d)}']), frozenset(['e[f+g]']) }
        patsy_f1_terms = tsm.patsy_terms( formula1 )
        self.assertSetEqual( patsy_f1_terms.as_fsets, f1_terms )
        
    def testPatsyVarsInTerms(self):
        formula1 = 'x1 + x2 + x1:x3 + I(x2**2):x3 + a{b+c(d)}+e[f+g]'
        f1_terms = { frozenset(['x1']), frozenset(['x2']), frozenset(['x1', 'x3']), frozenset(['x2', 'x3']), frozenset(['b', 'd']), frozenset(['f', 'g']) }
        patsy_f1_terms = tsm.patsy_terms( formula1, reduce_to_vars=True )
        self.assertSetEqual( patsy_f1_terms.as_fsets, f1_terms )
        

    def testMaskBrackets(self):
        formula = 'ln( Hello )'
        masked = tsm.mask_brackets( formula )
        self.assertEqual( formula, 'ln( Hello )' )
        self.assertEqual( masked, 'ln_________' )
        square_squig = 'func{ #tag } + name[index]'
        masked = tsm.mask_brackets( square_squig )
        self.assertEqual( square_squig, 'func{ #tag } + name[index]' )
        self.assertEqual( masked, 'func________ + name_______' )
        nested = '( brackets ( nested {properly} ) ) // { text{a + b } }'
        masked = tsm.mask_brackets( nested )
        self.assertEqual( nested, '( brackets ( nested {properly} ) ) // { text{a + b } }' )
        self.assertEqual( masked, '__________________________________ // ________________' )
        nestwo = ' +( nested { incorrectly] ) + ( again ( ) '
        masked = tsm.mask_brackets( nestwo )
        self.assertEqual( nestwo, ' +( nested { incorrectly] ) + ( again ( ) ' )
        self.assertEqual( masked, ' +_________________________ + ( again ___ ' )

    def testTermsIn(self):
        formula = 'A + B( hello ) - 9Nine -> Hello world!'
        self.assertSequenceEqual( [term for term in tsm.terms_in( formula )],
                                  ['A', 'B( hello )', '9Nine', 'Hello world!'] )
        formula = 'X + Y / C(a + b) - Z**2'
        self.assertSequenceEqual( [term for term in tsm.terms_in( formula )],
                                  ['X', 'Y / C(a + b)', 'Z**2'] )
    
class TestGenericClasses(tsm.TestCase):
    def testTypedDictWriteOnceString(self):
        td = tsm.typedDict( str, writeOnce=True )
        self.assertEqual( len( td ), 0 )
        self.assertEqual( td.itemLength( 'Five' ), 0 )
        td['Five'] = 'Hello'
        self.assertEqual( td.itemLength( 'Five' ), 5 )
        self.assertEqual( len( td ), 1 )
        self.assertEqual( len( td['Five'] ), 5 )
        self.assertEqual( len( td['Four'] ), 0 )
        self.assertIsInstance( td['Four'], str )
    def testTypedDictWriteManyString(self):
        td = tsm.typedDict( str )
        self.assertEqual( len( td ), 0 )
        self.assertEqual( td.itemLength( 'Five' ), 0 )
        td['Five'] = 'Hello'
        self.assertEqual( td.itemLength( 'Five' ), 5 )
        self.assertEqual( len( td ), 1 )
        self.assertEqual( len( td['Five'] ), 5 )
        self.assertEqual( len( td['Four'] ), 0 )
        self.assertIsInstance( td['Four'], str )
        td['Four'] = 'What'
        td['Five'] = 'Do you want?'
        self.assertEqual( td['Four'], 'What' )
        self.assertEqual( td['Five'], 'Do you want?' )
    def testTypedDictWriteOnceList(self):
        td = tsm.typedDict( list, writeOnce=True )
        self.assertEqual( len( td ), 0 )
        self.assertEqual( td.itemLength( 'Five' ), 0 )
        td['Five'] = ['Hello']
        self.assertEqual( td.itemLength( 'Five' ), 1 )
        self.assertEqual( len( td ), 1 )
        self.assertEqual( td.itemLength( 'Four' ), 0 )
        self.assertEqual( len( td ), 1 )
        self.assertEqual( len( td['Four'] ), 0 )
        self.assertEqual( len( td ), 2 )
        td['Four'].append( 'You?' )
        self.assertEqual( len( td['Four'] ), 1 )
        td['Four'].extend( ['What do you want?', 'Another list item.'] )
        self.assertEqual( td.itemLength( 'Four' ), 3 )
        self.assertListEqual( td['Four'], ['You?', 'What do you want?', 'Another list item.'] )
        i = td.strKeys['Four']
        self.assertEqual( str( td[i] ), str( td['Four'] ) )
        self.assertEqual( len( td ), 2 )
        td['Four'].clear( )
        self.assertEqual( td.itemLength( 'Four' ), 0 )
    def testTypedDictWriteOnceDefault(self):
        td = tsm.typedDict( list, writeOnce=True, default=['First, '] )
        self.assertEqual( len( td ), 0 )
        self.assertListEqual( td['One'], ['First, '] )
        td['Two'].append( 'I would like to say hello.' )
        self.assertListEqual( td['Two'], ['First, ', 'I would like to say hello.'] )
        td['Three'] = list( ['I would like to say hello.'] )
        self.assertListEqual( td['Three'], ['I would like to say hello.'] )
        self.assertEqual( len( td ), 3 )
        self.assertEqual( td.itemLength( 0 ), 1 )
    def testTypedDictWriteOnceDefaultWithSubclass(self):
        d = PR.mathDataStore( )
        td = PR.typedDict( PR.mathDataStore, writeOnce=True, default=d )
        self.assertEqual( len( td ), 0 )
        self.assertSequenceEqual( td['One'], d )
        self.assertEqual( len( td ), 1 )
    
    def testTypedDictMethods(self):
        # Pop
        td = tsm.typedDict( frozenset )
        td['a'] = frozenset( ['A', 'Bee', 'C'] )
        td['x'] = frozenset( [1, 2, 3] )
        td['z'] = frozenset( [123, 456] )
        self.assertSetEqual( set( td.keys( ) ), set( ['a', 'x', 'z'] ) )
        x = td.pop( 'x' )
        self.assertSetEqual( x, frozenset( [1, 2, 3] ) )
        self.assertSetEqual( set( td.keys( ) ), set( ['a', 'z'] ) )
        td['x'] = x
        # Update
        two = tsm.typedDict( frozenset )
        two['a'] = frozenset( ['one', 'two', 'three'] )
        two['b'] = frozenset( [7, 8, 9] )
        two['c'] = frozenset( ['red', 'green', 'blue'] )
        td.update( two )
        self.assertSetEqual( set( td.keys( ) ), set( ['a', 'b', 'c', 'x', 'z'] ) )
        self.assertSetEqual( td['a'], frozenset( ['one', 'two', 'three'] ) )
        self.assertSetEqual( td['b'], frozenset( [7, 8, 9] ) )
        self.assertSetEqual( td['x'], frozenset( [1, 2, 3] ) )
        # Union Update
        h1 = tsm.typedDict( set )
        h1['a'] = set( [1, 2, 3] )
        h1['b'] = set( [4, 5, 6] )
        h1['c'] = set( [7, 8, 9] )
        h2 = tsm.typedDict( set )
        h2['a'] = set( [4, 5, 6] )
        h2['b'] = set( [6, 7, 8] )
        h2['c'] = set( [8, 9, 0] )
        h1.union_update( h2 )
        self.assertSetEqual( h1['a'], set( [1, 2, 3, 4, 5, 6] ) )
        self.assertSetEqual( h1['b'], set( [4, 5, 6, 7, 8] ) )
        self.assertSetEqual( h1['c'], set( [7, 8, 9, 0] ) )
    
    def testTypedDictErrors(self):
        with self.assertRaises( TypeError ) as cm:
            td = tsm.typedDict( )
        self.assertEqual( cm.exception.args[0], "__init__() missing 1 required positional argument: 'typeRequirement'" )
        with self.assertRaises( TypeError ) as cm:
            td = PR.typedDict( PR.mathDataStore, default=dict( ) )
        self.assertEqual( cm.exception.args[0], 'The default item must be of the required type.' )
        td = tsm.typedDict( str, writeOnce=True, default='Set.' )
        self.assertEqual( len( td['Hi'] ), 4 )
        with self.assertRaises( KeyError ) as cm:
            td['Hi'] = 'Hello.'
        self.assertEqual( cm.exception.args[0], 'This typedDict is write-once.  The key, {:s}, has already been set.'.format( 'Hi' ) )
        with self.assertRaises( TypeError ) as cm:
            td['A'] = 123
        self.assertEqual( cm.exception.args[0], "This typedDict has been configured to only accept list items of type <class 'str'>.  Cannot accept a <class 'int'>." )
        with self.assertRaises( TypeError ) as cm:
            td['A'] = list( )
        self.assertEqual( cm.exception.args[0], "This typedDict has been configured to only accept list items of type <class 'str'>.  Cannot accept a <class 'list'>." )
        with self.assertRaises( KeyError ) as cm:
            print( td[0.] )
        self.assertEqual( cm.exception.args[0], 'typedDict only supports string keys and their integer indexes.' )
        with self.assertRaises( KeyError ) as cm:
            if not 0. in td:
                print( 'Hello there.' )
        self.assertEqual( cm.exception.args[0], 'typedDict only supports string keys and their integer indexes.' )
        with self.assertRaises( KeyError ) as cm:
            if td.itemLength( 0. ) > 0:
                print( 'Hello there.' )
        self.assertEqual( cm.exception.args[0], 'typedDict only supports string keys and their integer indexes.' )

class TestMathDictMaker(tsm.TestCase):
    def setUp(self):
        numZip = zip( [1, 2, 3, 10],
                      [4, 5, 6, 11],
                      [7, 8, 9, 12])
        self.numList = list( )
        try:
            while True:
                self.numList.extend( numZip.__next__( ) )
        except StopIteration:
            pass
        
    def testWorkingInt(self):
        i_arr = sharedctypes.RawArray( 'b', 12*8 )
        i_arr[:] = array.array( 'q', self.numList ).tobytes( )
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1, 4, 7]
        mdMaker['b'] = array.array( 'q', np.array( [2, 5, 8], dtype='i8' ) )
        mdMaker['c'] = [3, 6, 9]
        mdMaker['d'] = [10, 11, 12]
        arr, md = mdMaker.make( )
        self.assertEqual( arr[:], i_arr[:] )
        self.assertEqual( md.mask, [False] )
        self.assertSequenceEqual( md._column_names, ['a', 'b', 'c', 'd'] )
        self.assertSequenceEqual( md.columns, ['Intercept', 'a', 'b', 'c', 'd'] )
        dt = [np.dtype( 'i8' ) for x in range( 4 )]
        self.assertSequenceEqual( md.dtypes, dt )
        
    def testWorkingFloat(self):
        f_arr = sharedctypes.RawArray( 'b', 12*8 )
        f_arr[:] = array.array( 'd', self.numList ).tobytes( )
        mdMaker = tsm.mathDictMaker( )
        mdMaker['x1'] = [1., 4., 7.]
        mdMaker['x2'] = [2., 5., 8.]
        mdMaker['x3'] = [3., 6., 9.]
        mdMaker['x4'] = [10., 11., 12.]
        arr, md = mdMaker.make( )
        self.assertEqual( arr[:], f_arr[:] )
        self.assertEqual( md.mask, [False] )
        self.assertSequenceEqual( md._column_names, ['x1', 'x2', 'x3', 'x4'] )
        self.assertSequenceEqual( md.columns, ['Intercept', 'x1', 'x2', 'x3', 'x4'] )
        dt = [np.dtype( 'f8' ) for x in range( 4 )]
        self.assertSequenceEqual( md.dtypes, dt )
        
    def testWorkingMixedTypes(self):
        m_arr = sharedctypes.RawArray( 'b', 12*8 )
        m_arr[:6*8] = array.array( 'd', self.numList[:6] ).tobytes( )
        m_arr[6*8:] = array.array( 'q', self.numList[6:] ).tobytes( )
        mdMaker = tsm.mathDictMaker( )
        mdMaker['x_3'] = [3, 6, 9]
        mdMaker['x(2!)'] = [2., 5., 8.]
        mdMaker['x(1+2)'] = [1., 4., 7.]
        mdMaker['x_4'] = [10, 11, 12]
        arr, md = mdMaker.make( )
        self.assertEqual( arr[:], m_arr[:] )
        self.assertEqual( md.mask, [False] )
        self.assertSequenceEqual( md._column_names, ['x(1+2)', 'x(2!)', 'x_3', 'x_4'] )
        self.assertSequenceEqual( md.columns, ['Intercept', 'x(1+2)', 'x(2!)', 'x_3', 'x_4'] )
        dt = [np.dtype( 'f8' ) for x in range( 2 )]
        dt.extend( [np.dtype( 'i8' ) for x in range( 2 )] )
        self.assertSequenceEqual( md.dtypes, dt )
    
    def testErrors(self):
        mdMaker = tsm.mathDictMaker( )
        with self.assertRaisesWithMessage( KeyError,
                  "Keys must be valid Python variable names, alone or immediately followed by brackets.  'Two Words' is not." ):
            mdMaker['Two Words'] = [1, 2, 3, 4]
        with self.assertRaisesWithMessage( KeyError,
                  "Keys must be valid Python variable names, alone or immediately followed by brackets.  '2Errors' is not." ):
            mdMaker['2Errors'] = [1, 2, 3, 4]
        with self.assertRaisesWithMessage( KeyError,
                  "Keys must be valid Python variable names, alone or immediately followed by brackets.  'Cubed_Error ** 3' is not." ):
            mdMaker['Cubed_Error ** 3'] = [1, 2, 3, 4]
        with self.assertRaisesWithMessage( KeyError,
                  'Keys must be strings because they will be used as column names.' ):
            mdMaker[1] = [1]
        mdMaker['long'] = [1, 2, 3, 4]
        with self.assertRaisesWithMessage( ValueError,
                  'The sequence you are trying to add has a different length, 2, than existing sequence(s), 4.' ):
            mdMaker['short'] = [1, 2]
        with self.assertRaisesWithMessage( TypeError,
                  'All values must be non-string sequences of the same length.' ):
            mdMaker['int'] = 1
        with self.assertRaisesWithMessage( TypeError,
                  'All values must be non-string sequences of the same length.' ):
            mdMaker['string'] = 'Hello'
        with self.assertRaisesWithMessage( TypeError, 'One or more values in column numbers is neither an integer nor a float.' ):
            mdMaker['numbers'] = [1., 'a', 3, 4.]
        with self.assertRaisesWithMessage( TypeError, 'One or more values in column intfirst is not an integer.' ):
            mdMaker['intfirst'] = [1, 2., 3, 4]
        with self.assertRaisesWithMessage( TypeError, "All values must be ndarrays, sequences of ints, or sequences of numbers that start with a float.  mixed starts with neither an int nor a float.  'Good day.' is a <class 'str'>." ):
            mdMaker['mixed'] = ['Good day.', 2, 3, 4]

class TestMathDict(tsm.TestCase):        
    def setUp(self):
        self.dictClass = tsm.mathDict
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1., 4., 7.]
        mdMaker['b'] = [2, 5, 8]
        mdMaker['c'] = [3., 6., 9.]
        mdMaker['d'] = [10, 11, 12]
        self.arr, self.md = mdMaker.make( )
        numZip = zip( [ 1,  2,  3,  4],
                      [ 5,  6,  7,  8],
                      [ 9, 10, 11, 12],
                      [13, 14, 15, 16],
                      [17, 18, 19, 20],
                      [21, 22, 23, 24] )
        self.numList = list( )
        try:
            while True:
                self.numList.extend( numZip.__next__( ) )
        except StopIteration:
            pass
        
    def testBasicStorageRetrieval(self):
        arr = sharedctypes.RawArray( 'd', array.array( 'd', [1, 2, 3, 4, 5, 6, 7, 8, 9] ) )
        #
        md = self.dictClass( arr, 9, ['a'], [False] )
        self.assertTupleEqual( md.shape, (9,2) )
        self.assertEqual( md.rows, 9 )
        md.mask = [True]
        self.assertTupleEqual( md.shape, (9,1) )
        self.assertEqual( md.rows, 9 )
        #assert_almost_equal( md['a'], [[1], [2], [3], [4], [5], [6], [7], [8], [9]] )
        #
        self.assertTupleEqual( self.md.shape, (3,5) )
        self.assertEqual( self.md.rows, 3 )
        assert_almost_equal( self.md['b'], [[2], [5], [8]] )
        assert_almost_equal( self.md[2:4], [[2, 3],
                                            [5, 6],
                                            [8, 9]] )
        assert_almost_equal( self.md[2], [[2], [5], [8]] )
        assert_almost_equal( self.md[:], [[1, 1, 2, 3, 10],
                                          [1, 4, 5, 6, 11],
                                          [1, 7, 8, 9, 12]] )
        assert_almost_equal( self.md[2:], [[2, 3, 10],
                                           [5, 6, 11],
                                           [8, 9, 12]] )
        assert_almost_equal( self.md[:3], [[1, 1, 2],
                                           [1, 4, 5],
                                           [1, 7, 8]] )
        assert_almost_equal( self.md[1:3], [[1, 2],
                                            [4, 5],
                                            [7, 8]] )
        assert_almost_equal( self.md[0], [[1], [1], [1]] )
        md = self.dictClass( self.arr, 12, ['a', 'b', 'c', 'd'], [False, False, True, False, True] )
        assert_almost_equal( md[:], [[1, 1, 3],
                                     [1, 4, 6],
                                     [1, 7, 9]] )
        self.assertTupleEqual( md.shape, (3,3) )
        self.assertEqual( md.rows, 3 )
        md = self.dictClass( self.arr, 12, ['a', 'b', 'c', 'd'], [True, False, True, False, True] )
        assert_almost_equal( md[:], [[1, 3],
                                     [4, 6],
                                     [7, 9]] )
        assert_almost_equal( md[0], [[1], [4], [7]] )
        self.assertTupleEqual( md.shape, (3,2) )
    
    def testColumnNames(self):
        md = self.dictClass( self.arr, 12, ['ln( a )', 'b(@)', 'c(d!)', 'd'],
                             [False, False, False, False, False],
                             dtypes=['f8', 'i8', 'f8', 'i8'])
        assert_almost_equal( md['ln( a )'], [[1], [4], [7]] )
        assert_almost_equal( md['b(@)'], [[2], [5], [8]] )
        assert_almost_equal( md['c(d!)'], [[3], [6], [9]] )
        
    def testMasking(self):
        self.md.mask_all( )
        self.assertTupleEqual( self.md.shape, (3,0) )
        self.md.unmask_all( )
        self.assertTupleEqual( self.md.shape, (3,5) )
        assert_almost_equal( self.md[:], [[1, 1, 2, 3, 10],
                                          [1, 4, 5, 6, 11],
                                          [1, 7, 8, 9, 12]] )
        self.md.mask_all( except_intercept=True )
        self.assertTupleEqual( self.md.shape, (3,1) )
        assert_almost_equal( self.md[:], [[1],
                                          [1],
                                          [1]] )
        self.md.unmask_all( )
        self.md.set_mask( 'Intercept' )
        self.assertTupleEqual( self.md.shape, (3,4) )
        assert_almost_equal( self.md[:], [[1, 2, 3, 10],
                                          [4, 5, 6, 11],
                                          [7, 8, 9, 12]] )
        self.md.set_mask( 'b' )
        self.assertTupleEqual( self.md.shape, (3,3) )
        assert_almost_equal( self.md[:], [[1, 3, 10],
                                          [4, 6, 11],
                                          [7, 9, 12]] )
        self.md.set_mask( 'd' )
        self.assertTupleEqual( self.md.shape, (3,2) )
        assert_almost_equal( self.md[:], [[1, 3],
                                          [4, 6],
                                          [7, 9]] )
        self.md.set_mask( 'b', False )
        self.assertTupleEqual( self.md.shape, (3,3) )
        assert_almost_equal( self.md[:], [[1, 2, 3],
                                          [4, 5, 6],
                                          [7, 8, 9]] )
        self.md.set_mask( 'Intercept', False )
        self.assertTupleEqual( self.md.shape, (3,4) )
        assert_almost_equal( self.md[:], [[1, 1, 2, 3],
                                          [1, 4, 5, 6],
                                          [1, 7, 8, 9]] )
        self.md.mask_all( except_intercept=True )
        self.md.add_from_RHS( 'c + b**2 + b * c' )
        assert_array_almost_equal( self.md[:], [[1, 3, 2*2, 2*3],
                                                [1, 6, 5*5, 5*6],
                                                [1, 9, 8*8, 8*9]] )
        self.md.mask_all( except_intercept=True, clear_calculated=True )
        self.assertTupleEqual( self.md.shape, (3,1) )
        assert_almost_equal( self.md[:], [[1],
                                          [1],
                                          [1]] )
    
    def testAdd(self):
        self.md.mask_all( )
        self.assertTupleEqual( self.md.shape, (3,0) )
        self.md.add( 'Intercept' )
        self.assertTupleEqual( self.md.shape, (3,1) )
        assert_almost_equal( self.md[:], [[1],
                                          [1],
                                          [1]] )
        self.md.add( 'c' )
        self.assertTupleEqual( self.md.shape, (3,2) )
        assert_almost_equal( self.md[:], [[1, 3],
                                          [1, 6],
                                          [1, 9]] )
        self.md.add( 'b ** 2' )
        assert_array_almost_equal( self.md[:], [[1, 3, 2*2],
                                                [1, 6, 5*5],
                                                [1, 9, 8*8]] )  
        self.md.add( 'b*c' )
        assert_array_almost_equal( self.md[:], [[1, 3, 2*2, 2*3],
                                                [1, 6, 5*5, 5*6],
                                                [1, 9, 8*8, 8*9]] )
        with self.assertRaisesWithMessage( tsm.UnsupportedColumn, "mathDict( ) neither contains a column named, nor supports the calculation of, 'xyz'." ):
            result = self.md.add( 'xyz' )
    
    def testAddFromRHS(self):
        self.md.mask_all( except_intercept=True )
        self.assertTupleEqual( self.md.shape, (3,1) )
        assert_almost_equal( self.md[:], [[1],
                                          [1],
                                          [1]] )
        result = self.md.add_from_RHS( 'c + b**2 + b * c' )
        self.assertTrue( result )
        assert_almost_equal( self.md[:], [[1, 3, 2*2, 2*3],
                                          [1, 6, 5*5, 5*6],
                                          [1, 9, 8*8, 8*9]] )
        self.md.mask_all( except_intercept=True, clear_calculated=True )
        result = self.md.add_from_RHS( 'ln(y) ~ c + b**2 + b * c' )
        self.assertEqual( result, 'ln(y)' )
        assert_almost_equal( self.md[:], [[1, 3, 2*2, 2*3],
                                          [1, 6, 5*5, 5*6],
                                          [1, 9, 8*8, 8*9]] )
        self.md.mask_all( except_intercept=True, clear_calculated=True )
        self.assertTupleEqual( self.md.shape, (3,1) )
        assert_almost_equal( self.md[:], [[1],
                                          [1],
                                          [1]] )
        with self.assertRaises( tsm.UnsupportedColumn ) as cm:
            self.md.add_from_RHS( 'y ~ c + b**2 + xyz + b * c + b@' )
        self.assertEqual( cm.exception.msg, "One or more RHS terms could not be added.  RHS: ' c + b**2 + xyz + b * c + b@'." )
        self.assertEqual( cm.exception.LHS, 'y' )
        self.assertListEqual( cm.exception.columns, ['xyz', 'b@'] )
        assert_almost_equal( self.md[:], [[1, 3, 2*2, 2*3],
                                          [1, 6, 5*5, 5*6],
                                          [1, 9, 8*8, 8*9]] )
    
    def testPower(self):
        assert_almost_equal( self.md['a**1'],
                             [[1], [4], [7]] )
        assert_almost_equal( self.md['a**2'],
                             [[1*1], [4*4], [7*7]] )
        assert_almost_equal( self.md.power( 'b', 2 ),
                             [[2*2], [5*5], [8*8]] )
        assert_almost_equal( self.md.power( 'b', 0 ),
                             [[1], [1], [1]] )
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1., 4., 7.]
        mdMaker['b'] = [2, 5, 8]
        mdMaker['c'] = [3., 6., 9.]
        mdMaker['d'] = [10, 11, 12]
        RA, md = mdMaker.make( cache_crossproducts=True, cache_powers=3 )
        assert_almost_equal( md['a**4'],
                             [[1*1*1*1], [4*4*4*4], [7*7*7*7]] )
        assert_almost_equal( md['a**3'],
                             [[1*1*1], [4*4*4], [7*7*7]] )
        assert_almost_equal( md.power( 'b', 2 ),
                             [[2*2], [5*5], [8*8]] )
    
    def testCrossproduct(self):
        assert_almost_equal( self.md['a*b'],
                             [[1*2], [4*5], [7*8]] )
        assert_almost_equal( self.md.crossproduct( 'b', 'c' ),
                             [[2*3], [5*6], [8*9]] )
        assert_almost_equal( self.md['a*a'],
                             [[1*1], [4*4], [7*7]] )
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1., 4., 7.]
        mdMaker['b'] = [2, 5, 8]
        mdMaker['c'] = [3., 6., 9.]
        mdMaker['d'] = [10, 11, 12]
        RA, md = mdMaker.make( cache_crossproducts=True )
        assert_almost_equal( md['a*b'],
                             [[1*2], [4*5], [7*8]] )
        assert_almost_equal( md.crossproduct( 'b', 'c' ),
                             [[2*3], [5*6], [8*9]] )        
        
    def testLocalPowerCrossproduct(self):
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1., 4., 7.]
        mdMaker['b'] = [2, 5, 8]
        mdMaker['c'] = [3., 6., 9.]
        mdMaker['d'] = [10, 11, 12]
        RA, md = mdMaker.make( cache_crossproducts=True, cache_powers=3 )
        md['lcl_one'] = [11, 22, 33]
        md['lcl_two'] = [77, 88, 99]
        assert_almost_equal( md['lcl_one**2'],
                             [[11*11], [22*22], [33*33]] )        
        assert_almost_equal( md['lcl_one*b'],
                             [[11*2], [22*5], [33*8]] )
        assert_almost_equal( md['lcl_one*lcl_two'],
                             [[11*77], [22*88], [33*99]] )
        assert_almost_equal( md.crossproduct( 'b', 'lcl_two' ),
                             [[2*77], [5*88], [8*99]] )
    
    def testCrosspower(self):
        assert_array_equal( self.md['a**3*b'],
                             [[1**3*2], [4**3*5], [7**3*8]] )
        assert_array_equal( self.md.crosspower( 'b', 1, 'c', 2 ),
                             [[2*3**2], [5*6**2], [8*9**2]] )
        assert_array_equal( self.md['a*a**2'],
                             [[1**3], [4**3], [7**3]] )
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1., 4., 7.]
        mdMaker['b'] = [2, 5, 8]
        mdMaker['c'] = [3., 6., 9.]
        mdMaker['d'] = [10, 11, 12]
        RA, md = mdMaker.make( cache_crossproducts=True )
        assert_array_equal( md['a**3*b'],
                             [[1**3*2], [4**3*5], [7**3*8]] )
        assert_array_equal( md.crosspower( 'b', 1, 'c', 2 ),
                             [[2*3**2], [5*6**2], [8*9**2]] )
        assert_array_equal( md['a*a**2'],
                             [[1**3], [4**3], [7**3]] )

    def testCalculatedColumns(self):
        self.md.calculated_columns = ['b**2', 'c*d']
        assert_array_equal( self.md[:], [[1, 1, 2, 3, 10, 2*2, 3*10],
                                          [1, 4, 5, 6, 11, 5*5, 6*11],
                                          [1, 7, 8, 9, 12, 8*8, 9*12]] )
        
    def testProduct(self):
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1., 4., 7.]
        mdMaker['b'] = [2, 5, 8]
        mdMaker['c'] = [3., 6., 9.]
        mdMaker['d'] = [10, 11, 12]
        RA1, md1 = mdMaker.make( cache_crossproducts=True,
                                 interaction_columns=['a', 'b', 'c'] )
        RA2, md2 = mdMaker.make( cache_crossproducts=False,
                                 interaction_columns=['a', 'b', 'c'] )
        assert_array_equal( md1.product( ('a', 'b') ), [[1*2],
                                                        [4*5],
                                                        [7*8]] )
        assert_array_equal( md2.product( ('a', 'b') ), [[1*2],
                                                        [4*5],
                                                        [7*8]] )
        assert_array_equal( md1.product( ('a', 'b', 'c') ), [[1*2*3],
                                                             [4*5*6],
                                                             [7*8*9]] )
        assert_array_equal( md2.product( ('d', 'b', 'c') ), [[10*2*3],
                                                             [11*5*6],
                                                             [12*8*9]] )
        assert_array_equal( md1.product( ('d', 'b', 'c') ), [[10*2*3],
                                                             [11*5*6],
                                                             [12*8*9]] )
        assert_array_equal( md1['a:b:c'], [[1*2*3],
                                           [4*5*6],
                                           [7*8*9]] )
        assert_array_equal( md1['d:b:c'], [[10*2*3],
                                           [11*5*6],
                                           [12*8*9]] )
        assert_array_equal( md1['a:b:c ** 2'], [[1*2*3**2],
                                                [4*5*6**2],
                                                [7*8*9**2]] )
        assert_array_equal( md1['a:b*d:c'], [[1*2*10*3],
                                             [4*5*11*6],
                                             [7*8*12*9]] )
    
    def testHypothesis(self):
        self.md.hypothesis.add( 'b' )
        self.md.hypothesis.add( 'b ** 2' )
        self.md.hypothesis.add( 'b * c', 1 )
        X, R, r = self.md.hypothesis.make( )
        assert_array_almost_equal( X, [[1, 1, 2, 3, 10, 2*2, 2*3],
                                       [1, 4, 5, 6, 11, 5*5, 5*6],
                                       [1, 7, 8, 9, 12, 8*8, 8*9]] )
        assert_array_equal( R, [[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]] )
        assert_array_equal( r, [0, 0, 1] )

    def testHypothesisMasked_NoOrig(self):
        self.md.set_mask( 'b' )
        self.md.hypothesis.add( 'b' )
        self.md.hypothesis.add( 'b ** 2' )
        self.md.hypothesis.add( 'b * c', 1 )
        X, R, r = self.md.hypothesis.make( )
        np.testing.assert_array_almost_equal( X, [[1, 1, 3, 10, 2, 2*2, 2*3],
                                                  [1, 4, 6, 11, 5, 5*5, 5*6],
                                                  [1, 7, 9, 12, 8, 8*8, 8*9]] )
        np.testing.assert_array_equal( R, [[0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 1]] )
        np.testing.assert_array_equal( r, [0, 0, 1] )

    def testHypothesisMasked_WithOrig(self):
        self.md.set_mask( 'c' )
        self.md.hypothesis.add( 'b' )
        self.md.hypothesis.add( 'b ** 2' )
        self.md.hypothesis.add( 'b * c', 1 )
        X, R, r = self.md.hypothesis.make( )
        np.testing.assert_array_almost_equal( X, [[1, 1, 2, 10, 2*2, 2*3],
                                                  [1, 4, 5, 11, 5*5, 5*6],
                                                  [1, 7, 8, 12, 8*8, 8*9]] )
        np.testing.assert_array_equal( R, [[0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]] )
        np.testing.assert_array_equal( r, [0, 0, 1] )

    def testHypothesisMaskOrigLocal(self):
        self.md.set_mask( 'c' )
        self.md['lcl'] = [77, 88, 99]
        self.md.set_mask( 'lcl' )
        self.md.hypothesis.add( 'b' )
        self.md.hypothesis.add( 'b ** 2' )
        self.md.hypothesis.add( 'b * c', 1 )
        self.md.hypothesis.add( 'lcl**2' )
        assert_almost_equal( self.md.get_column( 'lcl' ), [[77], [88], [99]] )
        X, R, r = self.md.hypothesis.make( )
        assert_almost_equal( X, [[1, 1, 2, 10, 2*2, 2*3, 77*77],
                                 [1, 4, 5, 11, 5*5, 5*6, 88*88],
                                 [1, 7, 8, 12, 8*8, 8*9, 99*99]] )
        assert_array_equal( R, [[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]] )
        assert_array_equal( r, [0, 0, 1, 0] )

    def testConfigToDict(self):
        config = self.md.config_to_dict( )
        self.assertEqual( config['items'], 12 )
        self.md.items = 9
        self.assertEqual( config['items'], 12 )
        self.assertSequenceEqual( config['_column_names'], ['a', 'b', 'c', 'd'] )
        self.md._column_names = ['w', 'x', 'y', 'z']
        self.assertSequenceEqual( config['_column_names'], ['a', 'b', 'c', 'd'] )
        self.assertSequenceEqual( config['mask'], [False] )
        self.assertSequenceEqual( config['dtypes'], [np.dtype( 'f8' ),
                                                     np.dtype( 'i8' ),
                                                     np.dtype( 'f8' ),
                                                     np.dtype( 'i8' )] )
        self.assertSequenceEqual( config['calculated_columns'], [] )
        self.assertFalse( config['cache_crossproducts'] )
        self.assertEqual( config['cache_powers'], 1 )
        self.assertEqual( config['max_lag'], 0 )
        #
        md = self.dictClass( self.arr, 12, ['w', 'x', 'y', 'z'], [True, False, True, False, True], calculated_columns=['w**2', 'x*y'], cache_crossproducts=True, cache_powers=3 )
        md.max_lag = 3
        config2 = md.config_to_dict( )
        self.assertSequenceEqual( config2['_column_names'], ['w', 'x', 'y', 'z'] )
        self.assertSequenceEqual( config2['mask'], [True, False, True, False, True] )
        self.assertSequenceEqual( config2['calculated_columns'], ['w**2', 'x*y'] )
        self.assertTrue( config2['cache_crossproducts'] )
        self.assertEqual( config2['cache_powers'], 3 )
        self.assertEqual( config2['max_lag'], 3 )
    
    def testConfigFromDict(self):
        config = self.md.config_to_dict( )
        self.md._column_names = ['w', 'x', 'y', 'z']
        self.md.mask = [True]
        rebuilt_md = config.rebuild( self.arr )
        self.assertTupleEqual( rebuilt_md.shape, (3,5) )
        assert_almost_equal( rebuilt_md['b'], [[2], [5], [8]] )
        assert_almost_equal( rebuilt_md[:], [[1, 1, 2, 3, 10],
                                             [1, 4, 5, 6, 11],
                                             [1, 7, 8, 9, 12]] )
        mdMaker = tsm.mathDictMaker( )
        mdMaker['a'] = [1., 4., 7.]
        mdMaker['b'] = [2, 5, 8]
        mdMaker['c'] = [3., 6., 9.]
        mdMaker['d'] = [10, 11, 12]
        arr2, md2 = mdMaker.make( cache_crossproducts=True, cache_powers=2 )
        md2.calculated_columns = ['b**2', 'b*c']
        md2.max_lag = 1
        config2 = md2.config_to_dict( )
        rebuilt2 = config2.rebuild( arr2 )
        assert_array_almost_equal( rebuilt2[:], [[1, 4, 5, 6, 11, 5*5, 5*6],
                                                 [1, 7, 8, 9, 12, 8*8, 8*9]] )

    
    def testErrors(self):
        with self.assertRaisesWithMessage( KeyError,
                  'a + b is not a valid key.' ):
            self.md['a + b']
    
    def testLocalColumns(self):
        self.assertTupleEqual( self.md.shape, (3,5) )
        self.md['lcl'] = [77, 88, 99]
        self.assertTupleEqual( self.md.shape, (3,6) )
        self.assertEqual( self.md.rows, 3 )
        assert_almost_equal( self.md.get_column( 'lcl' ), [[77], [88], [99]] )
        self.assertSequenceEqual( self.md.columns, ['Intercept', 
                                                    'a', 
                                                    'b', 
                                                    'c', 
                                                    'd', 
                                                    'lcl'] )
        assert_almost_equal( self.md[:], [[1, 1, 2, 3, 10, 77],
                                          [1, 4, 5, 6, 11, 88],
                                          [1, 7, 8, 9, 12, 99]] )
        self.md.calculated_columns = ['c*d']
        assert_almost_equal( self.md[:], [[1, 1, 2, 3, 10, 77, 3*10],
                                          [1, 4, 5, 6, 11, 88, 6*11],
                                          [1, 7, 8, 9, 12, 99, 9*12]] )
        self.md.calculated_columns.append( 'c*lcl' )
        assert_almost_equal( self.md[:], [[1, 1, 2, 3, 10, 77, 3*10, 3*77],
                                          [1, 4, 5, 6, 11, 88, 6*11, 6*88],
                                          [1, 7, 8, 9, 12, 99, 9*12, 9*99]] )
        self.assertSequenceEqual( self.md.columns, ['Intercept', 
                                                    'a', 
                                                    'b', 
                                                    'c', 
                                                    'd',
                                                    'lcl',
                                                    'c*d',
                                                    'c*lcl'] )
        self.md.set_mask( 'lcl' )
        self.assertTupleEqual( self.md.shape, (3,7) )
        self.assertEqual( self.md.rows, 3 )
        assert_almost_equal( self.md[:], [[1, 1, 2, 3, 10, 3*10, 3*77],
                                          [1, 4, 5, 6, 11, 6*11, 6*88],
                                          [1, 7, 8, 9, 12, 9*12, 9*99]] )        
        self.assertSequenceEqual( self.md.columns, ['Intercept', 
                                                    'a', 
                                                    'b', 
                                                    'c', 
                                                    'd',
                                                    'c*d',
                                                    'c*lcl'] )
        self.md.mask_all( clear_calculated=False )
        
        self.assertSequenceEqual( self.md.columns, ['c*d', 'c*lcl'] )
        self.assertTupleEqual( self.md.shape, (3,2) )
        self.md.unmask_all( )
        assert_almost_equal( self.md[:], [[1, 1, 2, 3, 10, 77, 3*10, 3*77],
                                          [1, 4, 5, 6, 11, 88, 6*11, 6*88],
                                          [1, 7, 8, 9, 12, 99, 9*12, 9*99]] )
        self.assertSequenceEqual( self.md.columns, ['Intercept', 
                                                    'a', 
                                                    'b', 
                                                    'c', 
                                                    'd',
                                                    'lcl',
                                                    'c*d',
                                                    'c*lcl'] )
    
    def testLags(self):
        arr = sharedctypes.RawArray( 'd', array.array( 'd', self.numList ) )
        md = self.dictClass( arr, 24, ['a', 'b', 'c', 'd'], [True] )
        assert_array_equal( md['a'], [[1],
                                      [5],
                                      [9],
                                      [13],
                                      [17],
                                      [21]] )
        assert_array_equal( md.get_column( 'c' ), [[3],
                                                   [7],
                                                   [11],
                                                   [15],
                                                   [19],
                                                   [23]] )
        self.assertTupleEqual( md.shape, (6, 4) )
        self.assertEqual( md.rows, 6 )
        md.max_lag = 2
        self.assertTupleEqual( md.shape, (4, 4) )
        self.assertEqual( md.rows, 6 )
        assert_array_equal( md['a'], [[9],
                                      [13],
                                      [17],
                                      [21]] )
        assert_array_equal( md.get_column( 'c' ), [[11],
                                                   [15],
                                                   [19],
                                                   [23]] )
        assert_array_equal( md[:], [[ 9, 10, 11, 12],
                                    [13, 14, 15, 16],
                                    [17, 18, 19, 20],
                                    [21, 22, 23, 24]] )
        md.add( 'L2@a' )
        assert_array_equal( md[:], [[ 9, 10, 11, 12,  1],
                                    [13, 14, 15, 16,  5],
                                    [17, 18, 19, 20,  9],
                                    [21, 22, 23, 24, 13]] )
        md.add( 'L2@a*b' )
        assert_array_equal( md[:], [[ 9, 10, 11, 12,  1, 1*10],
                                    [13, 14, 15, 16,  5, 5*14],
                                    [17, 18, 19, 20,  9, 9*18],
                                    [21, 22, 23, 24, 13, 13*22]] )        
    
    def testLocalLags(self):
        arr = sharedctypes.RawArray( 'd', array.array( 'd', self.numList ) )
        md = self.dictClass( arr, 24, ['a', 'b', 'c', 'd'], [True] )
        md['lcl'] = [11, 22, 33, 44, 55, 66]
        self.assertTupleEqual( md.shape, (6, 5) )
        self.assertEqual( md.rows, 6 )
        md.max_lag = 2
        self.assertTupleEqual( md.shape, (4, 5) )
        self.assertEqual( md.rows, 6 )
        assert_array_equal( md['a'], [[9],
                                      [13],
                                      [17],
                                      [21]] )
        assert_array_equal( md[2], [[11],
                                    [15],
                                    [19],
                                    [23]] )
        assert_array_equal( md.get_column( 'lcl' ), [[33],
                                                     [44],
                                                     [55],
                                                     [66]] )
        assert_array_equal( md[:], [[ 9, 10, 11, 12, 33],
                                    [13, 14, 15, 16, 44],
                                    [17, 18, 19, 20, 55],
                                    [21, 22, 23, 24, 66]] )
        md.add( 'L2@lcl' )
        assert_array_equal( md[:], [[ 9, 10, 11, 12, 33, 11],
                                    [13, 14, 15, 16, 44, 22],
                                    [17, 18, 19, 20, 55, 33],
                                    [21, 22, 23, 24, 66, 44]] )
        md.add( 'L2@lcl*b**2' )
        md.set_mask( 'lcl' )
        assert_array_equal( md[:], [[ 9, 10, 11, 12, 11, 11*10**2],
                                    [13, 14, 15, 16, 22, 22*14**2],
                                    [17, 18, 19, 20, 33, 33*18**2],
                                    [21, 22, 23, 24, 44, 44*22**2]] )   

class TestTestCase(tsm.TestCase):    
    def testAssertRaisesWMessage(self):
        with self.assertRaisesWithMessage( TypeError, 'Hello Universe.' ):
            raise TypeError( 'Hello Universe.' )

    def testAssertStreamFileEqual(self):
        with self.assertStreamFileEqual( self, 'aSFE_test.txt' ) as f:
            print( 'Hello World.', file=f )
            print( '', file=f )
            print( 'Testing 1... 2... 3... ', file=f )
            
    @TestCase.assertStdoutFileEqual( 'aSFE_test.txt' )
    def testAssertStdoutFileEqual(self):
        print( 'Hello World.' )
        print( '' )
        print( 'Testing 1... 2... 3... ' )
    
    def testAssertFileLineSetEqual(self):
        f1 = 'exp_aSFE_test.txt'
        f2a = 'exp_aSFE_test - Copy.txt'
        f2b = 'exp_otherFile.txt'
        self.assertFileLineSetEqual( f1, f2a )
        with self.assertRaises( AssertionError ):
            self.assertFileLineSetEqual( f1, f2b )

class TestCategorizedSetDict(tsm.TestCase):
    def setUp(self):
        self.catSD = tsm.categorizedSetDict( )
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
    
    def testBasic(self):
        self.assertSequenceEqual( self.catSD['nums'], [1, 2, 3, 4, 5] )
        self.assertSequenceEqual( self.catSD['numbers'], ['one', 'two', 'three', 'four', 'five'] )
        self.catSD['one'] = 'word'
        self.assertSequenceEqual( self.catSD['one'], ['word'] )
        self.catSD['food'] = ['vegetables', 'seafood']
        self.assertSequenceEqual( self.catSD['food'], ['vegetables', 'seafood'] )
        self.catSD['food'] = tsm.setList( ['vegetables', 'seafood', 'pasta'] )
        self.assertSequenceEqual( self.catSD['food'], ['vegetables', 'seafood', 'pasta'] )
        with self.assertRaisesWithMessage( TypeError, "Items in the value category sequence must be sets, not <class 'dict'>." ):
            self.catSD['new'] = (['list'], [{}])
    
    def testGetCategory(self):
        self.assertSetEqual( self.catSD.get_categories( key='nums' ),
                             {'numerical',} )
        self.assertSetEqual( self.catSD.get_categories( key='numbers', value='one' ),
                             set( ) )
        self.assertSetEqual( self.catSD.get_categories( key='food', value='cucumber' ),
                             set( ) )
        self.assertTrue( self.catSD.is_None( key='food', value='cucumber' ) )
        self.assertFalse( self.catSD.is_None( key='food', value='apple' ) )
        self.assertTrue( self.catSD.is_a( 'numerical', key='nums' ) )
        self.assertFalse( self.catSD.is_a( 'numerical', key='numbers' ) )
        self.assertTrue( self.catSD.is_a( 'apples', key='food', value='apple' ) )
        self.assertFalse( self.catSD.is_a( 'apples', key='food', value='bananas' ) )
        self.catSD['newSet'] = [7, 8, 9]
        self.assertTrue( self.catSD.is_None( key='newSet' ) )
        self.assertFalse( self.catSD.is_None( key='nums' ) )
        self.catSD.set_category( 'words', key='numbers' )
        self.catSD.set_category( 'numerical', key='numbers' )
        self.assertSetEqual( self.catSD.get_categories( key='numbers' ),
                             {'numerical', 'words'} )
        self.assertSetEqual( self.catSD.get_categories( key='numbers', value='two' ),
                             {'multiple', 'even', 'numerical', 'words'} )

    def testSingularCategory(self):
        self.catSD['one'] = 'word'
        self.assertSequenceEqual( self.catSD['one'], ['word'] )
        self.assertFalse( self.catSD.is_a( 'single', key='one' ) )
        self.assertTrue( self.catSD.is_None( key='one' ) )
        catSD_one = tsm.categorizedSetDict( singular_category='single' )
        catSD_one['one'] = 'word'
        self.assertSequenceEqual( catSD_one['one'], ['word'] )
        self.assertTrue( catSD_one.is_a( 'single', key='one' ) )
        self.assertFalse( catSD_one.is_None( key='one' ) )
        catSD_two = tsm.categorizedSetDict( singular_category='single' )
        catSD_two['another'] = ['word']
        self.assertSequenceEqual( catSD_two['another'], ['word'] )
        self.assertFalse( catSD_two.is_a( 'single', key='another' ) )
        self.assertTrue( catSD_two.is_None( key='another' ) )

    def testCategorizedKVI(self):
        self.assertSequenceEqual( self.catSD.keys_categorized( 'numerical' ), ['nums'] )
        self.assertSequenceEqual( self.catSD.keys_categorized( 'multiple' ), [] )
        self.assertSequenceEqual( self.catSD.values_categorized( None ),
                                  ['one', 'cucumber'] )
        self.catSD.set_category( 'numerical', key='numbers', items={'sayings': 'two birds with one stone'} )
        self.assertSequenceEqual( self.catSD.values_categorized( None ),
                                  ['cucumber'] )
        self.assertSequenceEqual( self.catSD.values_categorized( 'numerical' ),
                                  [1, 2, 3, 4, 5, 'one', 'two', 'three', 'four', 'five', 'two birds with one stone'] )
        self.assertDictUnsortedEqual( self.catSD.items_categorized( 'numerical' ),
                                  {'nums': [1, 2, 3, 4, 5],
                                   'numbers': ['one', 'two', 'three', 'four', 'five'],
                                   'sayings': ['two birds with one stone']} )

    def testSetCategory(self):
        self.assertSequenceEqual( self.catSD.keys_categorized( 'food' ), [] )
        self.assertSequenceEqual( self.catSD.values_categorized( 'food' ), [] )
        self.catSD.set_category( 'food', keys=['food', 'sayings'] )
        self.assertSequenceEqual( self.catSD.keys_categorized( 'food' ),
                                  ['food', 'sayings'] )
        self.assertSequenceEqual( self.catSD.values_categorized( 'food' ),
                                  ['apple', 'bananas', 'cucumber', 'dates', 'fair and square', 'two birds with one stone', 'an apple a day'] )
        self.catSD.set_category( 'numerical', key='numbers', items={'sayings': 'two birds with one stone'} )
        self.assertDictUnsortedEqual( self.catSD.items_categorized( 'numerical' ),
                                  {'nums': [1, 2, 3, 4, 5],
                                   'numbers': ['one', 'two', 'three', 'four', 'five'],
                                   'sayings': ['two birds with one stone']} )
        self.catSD.set_category( 'newCategory', key='food', value='bananas' )
        self.assertSequenceEqual( self.catSD.values_categorized( 'newCategory' ),
                                  ['bananas'] )

    def testDelCategory(self):
        self.assertSequenceEqual( self.catSD.values_categorized( None ),
                                  ['one', 'cucumber'] )
        self.catSD.del_category( 'apples', items={'food': 'apple', 'sayings': 'an apple a day'} )
        self.assertSequenceEqual( self.catSD.values_categorized( None ),
                                  ['one', 'apple', 'cucumber', 'an apple a day'] )
        self.assertSequenceEqual( self.catSD.keys_categorized( 'numerical' ), ['nums'] )
        self.catSD.del_category( 'numerical', key='nums' )
        self.assertSequenceEqual( self.catSD.keys_categorized( 'numerical' ), [] )
        self.catSD.set_category( 'food', keys=['food', 'sayings', 'cooking'] )
        self.assertSequenceEqual( self.catSD.keys_categorized( 'food' ),
                                  ['food', 'sayings', 'cooking'] )
        self.catSD.del_category( 'food', keys=['food', 'sayings'] )
        self.assertSequenceEqual( self.catSD.keys_categorized( 'food' ),
                                  ['cooking'] )

    def testDelCategory2(self):
        self.catSD.del_category( 'apples', key='sayings', value='an apple a day' )
        self.assertSequenceEqual( self.catSD.values_categorized( 'apples' ), ['apple'] )

    def testMutuallyExclusive(self):
        self.catSD.make_mutually_exclusive( ['even', 'odd'] )
        self.catSD.set_category( 'odd', items={'numbers': ['one', 'three', 'five']} )
        with self.assertRaisesWithMessage( tsm.CategoryError,
                    "Cannot add category 'odd' to ['numbers']'two' because it and an existing category are mutually exclusive." ):
            self.catSD.set_category( 'odd', items={'numbers': 'two'} )
        with self.assertRaisesWithMessage( tsm.CategoryError, "Cannot add category 'odd' to ['sayings'] because one or more values already has a mutually exclusive category." ):
            self.catSD.set_category( 'odd', key='sayings' )
        self.catSD.make_mutually_exclusive( ['numerical', 'words'] )
        self.catSD.set_category( 'words', key='numbers' )
        with self.assertRaisesWithMessage( tsm.CategoryError, "Cannot add category 'numerical' to ['numbers'] because it already has a mutually exclusive category." ):
            self.catSD.set_category( 'numerical', key='numbers' )

class TestTerms(tsm.TestCase):
    def testTermSetWorking(self):
        W = set( ['T', 'y', 'x1', 'x2', 'x3', 'x4', 'd5', 'd6'] )
        T = 'T'
        dterms = set( ['d5', 'd6'] )
        terms = {'y': ['y', 'ln(y)'], 'x1':['x1', 'ln(x1)'], 'x2':['x2', 'ln(x2)'], 'x3':['x3', 'ln(x3)'], 'x4':['x4', 'ln(x4)'] }
        ts = tsm.termSet( terms, dterms, T )
        self.assertEqual( len( ts ), 8 )
        self.assertSetEqual( ts.W_term_set, W )
        self.assertSetEqual( ts.X_required_set, set( ) )
        self.assertSetEqual( ts.Y_term_set, set( ) )
        self.assertSetEqual( ts.other_terms, W )
        ts.Y( 'y' )
        ts.require( 'x1', 'd6' )
        self.assertSetEqual( ts.Y_term_set, set( 'y' ) )
        self.assertSetEqual( ts.X_required_set, set( ['x1', 'd6'] ) )
        self.assertSetEqual( ts.other_terms, W.difference( set( ['y', 'x1', 'd6'] ) ) )
        ts.require( 'd6', make=False )
        self.assertSetEqual( ts.X_required_set, set( ['x1'] ) )
        self.assertSetEqual( ts.other_terms, W.difference( set( ['y', 'x1'] ) ) )
        ts.changeT( 'x4' )
        self.assertSetEqual( ts.W_term_set, W )
        self.assertEqual( ts['x1'], ['x1', 'ln(x1)'] )
        self.assertEqual( ts['d5'], ['d5'] )
        self.assertEqual( ts['T'], ['T'] )
        self.assertEqual( ts['x4'], ['x4', 'ln(x4)'] )
        ts['x99'] = ['a', 'b']
        self.assertEqual( ts['x99'], ['a', 'b'] )
        self.assertSetEqual( ts.dummy_term_set, set( ['d5', 'd6'] ) )
        self.assertSetEqual( ts.real_term_set, set( ['T', 'y', 'x1', 'x2', 'x3', 'x4', 'x99'] ) )
        ts.dummy( 'T', make=True )
        self.assertSetEqual( ts.dummy_term_set, set( ['d5', 'd6', 'T'] ) )
        ts['d7'] = 'd7'
        self.assertSetEqual( ts.dummy_term_set, set( ['d5', 'd6', 'T', 'd7'] ) )
    
    def testOneRepresentationDummy(self):
        formula = "ln(UNITS1) ~ ln(REGPR1) + FEAT1 + DISP1 + (CUT1 > 0)"
        formulas = [formula]
        ts = tsm.termSet( formulas=formulas )
        ts.dummy( key='CUT1', value='(CUT1>0)' )
        self.assertSequenceEqual( ts.values_categorized( 'dummy' ),
                                  ['(CUT1>0)'] )
        self.assertSetEqual( ts.keys_categorized( None ),
                             {'REGPR1', 'FEAT1', 'DISP1'} )
        
    
    def testTermSetChangeTNone(self):
        W = set( ['T', 'y', 'x1', 'x2', 'x3', 'x4', 'd5', 'd6'] )
        T = 'T'
        dterms = set( ['d5', 'd6'] )
        terms = {'y': ['y', 'ln(y)'], 'x1':['x1', 'ln(x1)'], 'x2':['x2', 'ln(x2)'], 'x3':['x3', 'ln(x3)'], 'x4':['x4', 'ln(x4)'] }
        ts = tsm.termSet( terms, dterms, T )
        self.assertEqual( len( ts ), 8 )
        self.assertSetEqual( ts.W_term_set, W )
        ts.Y( 'y' )
        ts.changeT( 'x4' )
        self.assertIn( 'T', ts.real_term_set )
        ts.changeT( None )
        ts['x4'] = None
        self.assertEqual( len( ts ), 7 )
        self.assertNotIn( 'x4', ts.W_term_set )
        self.assertNotIn( None, ts.W_term_set )
        
    def testTermSetErrors(self):
        W = set( ['T', 'y', 'x1', 'x2', 'x3', 'x4', 'd5', 'd6'] )
        T = 'T'
        dterms = set( ['d5', 'd6'] )
        terms = {'y': ['y', 'ln(y)'], 'x1':['x1', 'ln(x1)'], 'x2':['x2', 'ln(x2)'], 'x3':['x3', 'ln(x3)'], 'x4':['x4', 'ln(x4)'] }
        with self.assertRaises( TypeError ):
            ts = tsm.termSet( terms, T=T )
        with self.assertRaises( TypeError ) as cm:
            ts = tsm.termSet( terms, T, dterms )
        self.assertEqual( cm.exception.args[0], 'The set of dummy variables must be a non-string iterable.' )
        with self.assertRaises( TypeError ) as cm:
            ts = tsm.termSet( T, terms, dterms )
        self.assertEqual( cm.exception.args[0], 'The set of real terms must be a dictionary.' )
        with self.assertRaises( TypeError ) as cm:
            ts = tsm.termSet( terms, T )
        self.assertEqual( cm.exception.args[0], 'The set of dummy variables must be a non-string iterable.' )
        ts = tsm.termSet( terms, dterms )
        with self.assertRaises( TypeError ) as cm:
            ts = tsm.termSet( terms, list( ), dterms )
        self.assertEqual( cm.exception.args[0], 'The time index term must be identified by a string not in the set of dummy terms and not used as a key in the dictionary of real terms.' )
        ts = tsm.termSet( terms, dterms )
        with self.assertRaises( TypeError ) as cm:
            ts['x99'] = 88
        self.assertEqual( cm.exception.args[0], "Value is not a supported type.  Type: <class 'int'>, length: 0." )           
        ts.Y( 'y' )
        with self.assertRaises( KeyError ) as cm:
            ts.require( 'y' )
        self.assertEqual( cm.exception.args[0], 'That key is already in Y.  It cannot both be in Y and a required member of X.' )
        with self.assertRaises( KeyError ) as cm:
            ts.require( 'x3', 'y' )
        self.assertEqual( cm.exception.args[0], 'That key is already in Y.  It cannot both be in Y and a required member of X.' )
        ts.require( 'x2' )
        with self.assertRaises( KeyError ) as cm:
            ts.Y( 'x2' )
        self.assertEqual( cm.exception.args[0], 'That key is already a required member of X.  It cannot be in Y while it is a required member of X.' )
        
    def testDummyInteractions(self):
        dterms = tsm.setList( ['a', 'b', 'c'] )
        terms = dict( )
        ts = tsm.termSet( terms=terms, dterms=dterms )
        dt_interactions = set( ['a:b', 'a:c', 'b:c', 'a:b:c', 'a', 'b', 'c'] )
        self.assertSetEqual( ts.dummy_interactions, dt_interactions )
    
    def testDummyNumInteractions(self):
        dterms = tsm.setList( ['a', 'b', 'c'] )
        terms = {'X1': ['X1', 'ln(X1)'], 'X2': ['X2', 'I(X2**2)']}
        ts = tsm.termSet( terms=terms, dterms=dterms )
        X1_complete = dict( )
        X1_complete['X1'] = ['a:b:X1', 'a:c:X1', 'b:c:X1', 'a:b:c:X1', 'a:X1', 'b:X1', 'c:X1']
        X1_complete['ln(X1)'] = ['a:b:ln(X1)',
                                 'a:c:ln(X1)',
                                 'b:c:ln(X1)',
                                 'a:b:c:ln(X1)',
                                 'a:ln(X1)',
                                 'b:ln(X1)',
                                 'c:ln(X1)']
        self.assertDictUnsortedEqual( ts.get( 'X1', interactions=True ), X1_complete )
        X2_complete = dict( )
        X2_complete['X2'] = ['a:b:X2', 'a:c:X2', 'b:c:X2', 'a:b:c:X2', 'a:X2', 'b:X2', 'c:X2']
        X2_complete['I(X2**2)'] = ['a:b:I(X2**2)',
                                   'a:c:I(X2**2)',
                                   'b:c:I(X2**2)',
                                   'a:b:c:I(X2**2)',
                                   'a:I(X2**2)',
                                   'b:I(X2**2)',
                                   'c:I(X2**2)']
        self.assertDictUnsortedEqual( ts.get( 'X2', interactions=True ), X2_complete )
    
    def testGetNoInteractions(self):
        dterms = set( ['a', 'b', 'c'] )
        terms = {'X1': ['X1', 'ln(X1)'], 'X2': ['X2', 'I(X2**2)']}
        ts = tsm.termSet( terms=terms, dterms=dterms )
        self.assertSetEqual( set( ts.get( 'X1', interactions=False ) ),
                             set( ['X1', 'ln(X1)'] ) )
        self.assertSetEqual( set( ts.get( 'X2', interactions=False ) ),
                             set( ['X2',
                                   'I(X2**2)'] ) )

    def testExpandedDummyNumInteractions(self):
        dterms = tsm.setList( ['a', 'b', 'c'] )
        terms = {'X1': ['X1', 'ln(X1)'], 'X2': ['X2', 'I(X2**3)']}
        ts = tsm.termSet( terms=terms, dterms=dterms )
        X2_complete = dict( )
        X2_complete['X2'] = ['a:b:X2', 'a:c:X2', 'b:c:X2', 'a:b:c:X2', 'a:X2', 'b:X2', 'c:X2']
        X2_complete['I(X2**3) + I(X2**2) + X2'] = ['a:b:I(X2**3) + a:b:I(X2**2) + a:b:X2',
                                                   'a:c:I(X2**3) + a:c:I(X2**2) + a:c:X2',
                                                   'b:c:I(X2**3) + b:c:I(X2**2) + b:c:X2',
                                                   'a:b:c:I(X2**3) + a:b:c:I(X2**2) + a:b:c:X2',
                                                   'a:I(X2**3) + a:I(X2**2) + a:X2',
                                                   'b:I(X2**3) + b:I(X2**2) + b:X2',
                                                   'c:I(X2**3) + c:I(X2**2) + c:X2']
        self.assertDictUnsortedEqual( ts.get( 'X2', interactions=True, expand=True ), X2_complete )

    def testGetExpandedNoInteractions(self):
        dterms = set( ['a', 'b', 'c'] )
        terms = {'X1': ['X1', 'ln(X1)'], 'X2': ['X2', 'I(X2**3)']}
        ts = tsm.termSet( terms=terms, dterms=dterms )
        self.assertSetEqual( set( ts.get( 'X1', interactions=False, expand=True ) ),
                             set( ['X1', 'ln(X1)'] ) )
        self.assertSetEqual( set( ts.get( 'X2', interactions=False, expand=True ) ),
                             set( ['X2',
                                   'I(X2**3) + I(X2**2) + X2'] ) )
        
    def testInitFromFormulas_PricePromo(self):
        formulas = ['ln(UNITS1) ~ ln(REGPR1) + FEAT1 + DISP1 + (CUT1 > 0) + T',
                    'ln(UNITS1) ~ ln(REGPR1) + FEAT1 + DISP1 + (CUT1 > 0)',
                    'ln(UNITS1) ~ ln(REGPR1) + FEAT1 + DISP1 + CUT1',
                    'ln(UNITS1) ~ ln(REGPR1) + FEAT1 + DISP1',
                    'ln(UNITS1) ~ ln(REGPR1) + FEAT1',
                    'ln(UNITS1) ~ ln(REGPR1) + DISP1',
                    'ln(UNITS1) ~ ln(REGPR1)',
                    'UNITS1 ~ REGPR1 + FEAT1 + DISP1 + CUT1',
                    'UNITS1 ~ REGPR1 + FEAT1 + DISP1',
                    'UNITS1 ~ REGPR1 + FEAT1',
                    'UNITS1 ~ REGPR1',
                    'UNITS1 ~ ADVPR1']
        ts = tsm.termSet( formulas=formulas )
        self.assertEqual( len( ts ), 7 )
        self.assertSequenceEqual( ts['UNITS1'], ['ln(UNITS1)', 'UNITS1'] )
        self.assertSequenceEqual( ts['CUT1'], ['(CUT1>0)', 'CUT1'] )
        self.assertSequenceEqual( ts['ADVPR1'], ['ADVPR1'] )
        self.assertSetEqual( ts.Y_term_set, set( ['UNITS1'] ) )

class TestSetList(tsm.TestCase):
    def testAddToSetList_asifList(self):
        sl = tsm.setList( [123, 'abc'] )
        self.assertFalse( sl.append( 123 ) )
        sl[0] = 'abc'
        self.assertFalse( sl.lastSIOutcome )
        sl[2] = 'xyz'
        self.assertTrue( sl.lastSIOutcome )
        self.assertTrue( sl.append( 789 ) )
        self.assertEqual( sl.extend( ['abc', 'mno', 'xyz'] ), 1 )
        self.assertSequenceEqual( sl, [123, 'abc', 'xyz', 789, 'mno'] )
        
    def testAddToSetList_asifSet(self):
        sl = tsm.setList( [123, 'abc'] )
        self.assertFalse( sl.add( 123 ) )
        self.assertTrue( sl.add( 789 ) )
        self.assertEqual( sl.update( ['abc', 'mno', 'xyz'] ), 2 )
        self.assertSequenceEqual( sl, [123, 'abc', 789, 'mno', 'xyz'] )
    
    def testRemoveFromSetList(self):
        sl = tsm.setList( [123, 'abc', 789, 'mno', 'xyz'] )
        sl.remove( 789 )
        self.assertSequenceEqual( sl, [123, 'abc', 'mno', 'xyz'] )
        with self.assertRaises( ValueError ):
            sl.remove( 'ab' )
        self.assertFalse( sl.discard( 'yz' ) )
        self.assertTrue( sl.discard( 'xyz' ) )
        self.assertSequenceEqual( sl, [123, 'abc', 'mno'] )
        with self.assertRaises( TypeError ):
            sl.pop( 1 )
        with self.assertRaisesWithMessage( KeyError, 'One of `index` or `value` must be specified by keyword arguement when calling setList.pop( ).' ):
            sl.pop( )
        self.assertEqual( sl.pop( index=1 ), 'abc' )
        self.assertEqual( sl.pop( value='mno' ), 'mno' )
        self.assertEqual( sl.pop( value=123 ), 123 )
        self.assertEqual( len( sl ), 0 )
    
    def testSetMethods(self):
        s1 = set( [1, 2, 3, 4] )
        slist = tsm.setList( s1 )
        s2 = set( [2, 4, 6, 8] )
        self.assertSetEqual( s1.difference( s2 ),
                             set( slist.difference( s2 ) ) )
        self.assertSetEqual( s1.union( s2 ),
                             set( slist.union( s2 ) ) )
        self.assertSetEqual( s1.intersection( s2 ),
                             set( slist.intersection( s2 ) ) )
        self.assertSetEqual( s1.symmetric_difference( s2 ),
                             set( slist.symmetric_difference( s2 ) ) )
        self.assertFalse( slist.issubset( s2 ) )
        self.assertFalse( slist.issuperset( s2 ) )
        s3 = set( [1, 2] )
        s4 = set( [1, 2, 3, 4, 5, 6] )
        self.assertTrue( slist.issubset( s4 ) )
        self.assertTrue( slist.issuperset( s3 ) )

    def testInit(self):
        sl = tsm.setList( )
        self.assertEqual( len( sl ), 0 )
        sl = tsm.setList( [1, 2, 2, 'three', 'four'] )
        self.assertEqual( len( sl ), 4 )
        self.assertSequenceEqual( sl, [1, 2, 'three', 'four'] )
        sl = tsm.setList( {'1', '3', '2', 'four'} )
        self.assertSequenceEqual( list( sl ), ['1', '2', '3', 'four'] )

if __name__ == "__main__":
    ts = unittest.TestLoader( ).loadTestsFromName( 'tools_test.TestMathDict' )
    tr = unittest.TextTestRunner( )
    tr.run( ts )