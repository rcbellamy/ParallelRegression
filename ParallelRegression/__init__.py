## (C) 2017 by Richard Bellamy
##
## This software is licensed under the terms of version 3 of the GNU Affero
## General Public License.
from copy import deepcopy
from collections import abc, UserList, Sequence, OrderedDict
from contextlib import suppress, ContextDecorator, contextmanager
from io import StringIO, BytesIO
import unittest
import sys
import re
import numpy as np
from multiprocessing import sharedctypes, Queue, Process, cpu_count
import array
import math, itertools, random, keyword

__all__ = ['vCovMatrix',                          # Functions
           'FStatistic',
           'syncText',
           'val_if_present',
           'has_term',
           'mask_brackets',
           'masked_dict',
           'masked_iter',
           'masked_split',
           'terms_in',
           'formulas_match',
           'termString',
           'despace',
           'CategoryError',                       # Errors & warnings
           'UnsupportedColumn',
           'mathDictKeyError',
           'RankError',
           'setList',                             # Building block classes
           'typedDict',
           'categorizedSetDict',
           'laggedAccessor',
           'termSet',                             # Parallel regression-
           'mathDataStore',                       # oriented classes
           'mathDictMaker',
           'mathDictHypothesis',
           'mathDictConfig',
           'mathDict']

def vCovMatrix( X, u, vcType='White1980' ):
    '''Computes a variance-covariance matrix.

    Parameters
    ----------
    X : 2-dimensional array
        Matrix of X values.
    u : vector array
        Vector of residuals.
    vcType : {'White1980', 'Classical'}, optional
        Type of variance-covariance matrix requested.  'Classical' for the
        classical statistics formula, or 'White1980' for the
        heteroskedasticity-robust formula originally proposed by Halbert White
        in his 1980 paper, 'A Heteroskedasticity-Consistent Covariance Matrix
        Estimator and a Direct Test for Heteroskedasticity'.

    Notes
    -----
    The heteroskedasticity-robust formula supported is the formula explained
    in the documentation to the "car" R package's "hccm" function:

    "The classical White-corrected coefficient covariance matrix ("hc0") (for
    an unweighted model) is

    "V(b) = inv(X'X) X' diag(e^2) X inv(X'X)

    "where e^2 are the squared residuals, and X is the model matrix."

    This is the same formula proposed by White in 1980.  However, the car
    pachage documentation is substantially more clear and concise than either
    the original paper or most textbook discussions.
    '''
    if len( getattr( X, 'shape', '' ) ) != 2:
        raise ValueError('X must be a 2-dimensional ndarray with exactly two '
                         'dimensions.')
    if len( X[:,0] ) != len( u ):
        raise ValueError('Length of `u` must match the number of rows in `X`.')
    else:
        nobs = len( u )
    XpXi = np.linalg.inv( X.T.dot( X ) )

    horse = np.zeros( ( nobs, nobs ) )
    u = np.array( u )
    sigma_sq = u.T.dot( u ) / ( len( u ) - len( X[0,:] ) )
    if vcType == 'White1980':
        for i in range( nobs ):
            horse[i,i] = u[i]**2
    else:
        for i in range( nobs ):
            horse[i,i] = sigma_sq
    inner = X.T.dot( horse ).dot( X )

    vCovM = XpXi.dot( inner ).dot( XpXi )
    return( vCovM )

def FStatistic( X, u, coefs, R, r, vcType='White1980' ):
    '''Computes an F statistic that by default is heteroskedasticity-robust.

    Parameters
    ----------
    X : 2-dimensional array
        Matrix of all regressors including the intercept.
    u : vector array
        Vector of all residuals.
    coefs : vector array
        All coefficients from the model that is being tested, including the
        intercept and untested parameters.
    R : 2-dimensional array
        Linear restrictions in matrix form.
    r : vector array
        Linear restriction values in vector form.
    vcType : {'White1980', 'Classical'}, optional
        Type of variance-covariance matrix requested.  Keep the default setting
        for a heteroskedasticity-robust result.  See vCovMatrix function for
        details.
    '''
    vc = vCovMatrix( X, u, vcType )
    Rterm = R.dot( coefs ) - r
    Lterm = Rterm.T
    Mterm = R.dot( vc ).dot( R.T )
    if len( r ) > 1:
        Mterm = np.linalg.inv( Mterm )
    else:
        Mterm = Mterm**-1
    F = Lterm.dot( Mterm ).dot( Rterm ) / len( r )
    if math.isnan( F ):
        raise ValueError( 'F Statistic calculation resulted in nan.', vcType,
                                                                      vc.shape,
                                                                      F,
                                                                      Mterm,
                                                                      Rterm,
                                                                      r )
    return( F )

def syncText( strA, strB, addA, addB, pre='' ):
    '''Adds necessary spacing to align simultaneous additions to two strings.

    Parameters
    ----------

    strA : string
        The first of the two strings.
    strB : string
        The second of the two strings.
    addA : string
        The string to be appended to the first string.
    addB : string
        The string to be appended to the second string.
    pre : string, optional
        This string is appended to `strA` and `strB` immediately before `addA`
        and `addB`.

    Returns
    -------

    strA : string
        The first string, with `addA` appended.
    strB : string
        The second string, with `addB` appended, starting at the same index as
        `addA` in the first string.

    Example
    -------

    >>> upper, lower = ('John', 'Proper Noun')
    >>> for a, b in [('ate', 'Verb'),('an', 'Article'),('apple.', 'Noun')]:
    >>>     upper, lower = syncText( upper, lower, a, b, ' ' )
    >>> print( upper, '\\n', lower )
    John        ate  an      apple.
    Proper Noun Verb Article Noun
    '''
    ## Please pardon the inconsistency in the headings' line spacing between
    ## this and other docstrings, and the escaping the backslash in '\n'.  Both
    ## were necessitated by Sphinx.
    spaces = abs( len( strA ) - len( strB ) )
    spaces = ''.join( [' ' for i in range( spaces )] )
    if len( strA ) > len( strB ):
        strB += spaces
    else:
        strA += spaces
    strA += pre + addA
    strB += pre + addB
    return( strA, strB )

def val_if_present( obj, attr=None, alt=None ):
    '''Returns the requested value if it is set and is not None.  Otherwise
    `alt` is returned.

    Avoids errors if the requested value does not exist, while handling the
    presence of a None value and a default value in a different manner than
    getattr( ).

    Parameters
    ----------
    obj
        An object from which to attempt to retrieve an attribute value.
    attr : string, optional
        The name of the attribute to be retrieved.  If `attr` is not set, then
        the value of `obj` will be returned unless it is equal to None.  If '.'
        is present in this string, child objects will be retrieved recursively.
        See example below.
    alt, optional
        The value to be returned if the requested attribute does not exist, or
        is equal to None, or if the object's value is requested but is equal to
        None.  If this is not set, then None will be returned in these
        scenarios.

    Returns
    -------
    obj
        If the requested value is set and is not equal to None, then the
        requested value is returned.  Otherwise, `alt` is returned.  No error
        is raised if `obj` does not have an attribute named `attr`.

    Examples
    --------
    >>> class testFixture(object):
    >>>     NoneVal = None
    >>>     number = 123
    >>> testFix = testFixture( )
    >>> val_if_present( testFix, 'NoneVal', 'ABCdef' ) == 'ABCdef'
    True
    >>> testFix.fixtureTwo = testFixture( )
    >>> testFix.fixtureTwo.dict_object = {'a_key': 'has a value.'}
    >>> val_if_present( testFix, 'fixtureTwo.dict_object.a_key' ) \\
    >>>               == 'has a value.'
    True
    '''
    if obj == None:
        return( alt )
    if attr == None:
        return( obj )
    if isinstance( attr, str ) and attr.count( '.' ) > 0:
        attr = attr.split( '.', maxsplit=1 )
        if attr[0] in obj.__dir__( ):
            tpl = ( getattr( obj, attr[0] ), attr[1], alt )
            return( val_if_present( getattr( obj, attr[0] ), attr[1], alt ) )
        else:
            return( alt )
    if not attr in obj.__dir__( ):
        if isinstance( obj, dict ) and attr in obj.keys( ):
            return( obj[attr] )
        return( alt )
    attrVal = getattr( obj, attr )
    if attrVal == None:
        return( alt )
    else:
        return( attrVal )

def has_term( formula, term ):
    '''Returns True if `formula` either starts with `term` followed by one of
    [ )+-~*:] or contains `term` followed by one those characters, preceeded by
    one of [ (+-~*:].
    '''
    before = ' (+-~*:<>'
    after = ' )+-~*:<>'
    if formula.startswith( term ):
        if len( formula ) == len( term ) or formula[len( term )] in after:
            return( True )
    for b in before:
        i = formula.find( b )
        while i >= 0:
            if formula[i+1:].startswith( term ):
                if len( formula[i+1:] ) == len( term ):
                    return( True )
                if ( formula[i+len( term )+1] in after ):
                    return( True )
            i = formula.find( b, i + 1 )
    return( False )

splitter_rstr = r'[\+\-\~]+'
splitter = re.compile( splitter_rstr )
splitted_rstr = r'(?:[^\+\-\~\ ]+\ )*[^\+\-\~\ ]+'
splitted = re.compile( splitted_rstr )
one_star = re.compile( r'\*(?!\*)' )
paren = re.compile( r'(?:\([^\(]*?\))|(?:\{[^\{]*?\})|(?:\[[^\[]*?\])' )
varpattern_rstr = r'[a-zA-Z_][a-zA-Z_0-9]*(?![a-zA-Z_0-9\(\{\[\@])'
varpattern = re.compile( varpattern_rstr )
termpat_rstr = r'[a-zA-Z_0-9]+(?:[ ]*[^ \+\-\(\{\[]+)*(?![a-zA-Z_0-9\(\{\[])'
termpattern = re.compile( termpat_rstr )

def mask_brackets( string ):
    ''' Mask anything inside brackets, including nested brackets, by replacing
    the brackets and their contents with a same-length string of repeated
    underscores.
    '''
    def repl( mobj ):
        replacement = ''.join( ['_' for i in \
                                range( mobj.end( ) - mobj.start( ) )] )
        return( replacement )
    nsubstitutions = 1
    while nsubstitutions > 0:
        string, nsubstitutions = paren.subn( repl, string )
    return( string )

def masked_dict( string, mobj ):
    '''Recovers the corresponding contents from the original string based on a
    regular expressions match object produced using a masked string.  Compare
    to mobj.groupdict( ).

    Parameters
    ----------
    string : string
        The unmasked string from which content is to be recovered.
    mobj : regular expression match object
        The match object resulting from a regular expression pattern matched to
        a masked version of `string`.

    Returns
    -------
    dict
        Dictionary containing the substrings of `string` corresponding to the
        named subgroups of the match, keyed by the subgroup name.
    '''
    if mobj == None:
        return( None )
    ret = dict( )
    for k, v in mobj.re.groupindex.items( ):
        a, t = mobj.span( v )
        ret[k] = string[a:t]
    return( ret )

def masked_iter( string, mobj_iter ):
    '''Recovers the corresponding contents from the original string based on
    regular expression match objects produced by an iterable returned from
    re.finditer( ) or from a pattern object's .finditer( ) method.

    Parameters
    ----------
    string : string
        The unmasked string from which content is to be recovered.
    mobj_iter : iterable of regular expression match objects
        The iterable of regular expression match objects resulting from a
        regular expression pattern matched to a masked version of `string`.

    Returns
    -------
    list
        List containing the substrings of `string` corresponding to the
        substring matched by each match object produced by the iterable.
    '''
    ret = list( )
    for mobj in mobj_iter:
        a, t = mobj.span( )
        ret.append( string[a:t] )
    return( ret )

def masked_split( string, mask, split ):
    '''Splits `string` based on the location(s) at which `split` is located in
    `mask`.  Compare to str.split( ).

    Parameters
    ----------
    string : string
        The unmasked string from which content is to be recovered.
    mask : string
        The masked version of `string` to be used to determine the the
        location(s) at which to split `string`.
    split : string
        The string identifying the location(s) at which to split `string`.

    Returns
    -------
    list
        List of substrings resulting from splitting `string` based on the
        presence of `split` in `mask`.
    '''
    mask = mask.split( split )
    ret = list( )
    a = 0
    for m in mask:
        t = a + len( m )
        ret.append( string[a:t] )
        a = t + len( split )
    return( ret )

def terms_in( formula ):
    '''Generator that yields individual terms in `formula`.
    '''
    mask = mask_brackets( formula )
    for mobj in termpattern.finditer( mask ):
        a, t = mobj.span( )
        yield formula[a:t]

def _regex_split( string, pattern ):
    '''Like str.split( ) except that it splits on anything matching the regular
    expression pattern object.

    Parameters
    ----------
    string : string
        The string to be split.
    pattern : regular expression pattern object
        The pattern object identifying the locations at which to split the
        string.  This is the object that is returned by re.compile( ).

    Yields
    ------
    string
        The strings in `string` that are between each pattern match.
    '''
    a = 0
    wall = pattern.search( string )
    while wall != None:
        t = wall.start( )
        yield( string[a:t] )
        a = wall.end( )
        wall = pattern.search( string, a )
    if a < len( string ):
        yield( string[a:] )

def despace( string ):
    '''Removes inconsequential spaces without removing spaces that might impact
    the interpretation of a formula or line of code.
    '''
    string = string.strip( )
    string = string.replace( '( ', '(' )
    string = string.replace( ' )', ')' )
    string = string.replace( ' *', '*' )
    string = string.replace( '* ', '*' )
    return( string )

def _patsy_terms( formula, reduce_to_vars=False ):
    '''Splits a formula string into a set of terms by splitting on `+`, `-` and
    `~` characters.  Then splits each term into a set of `factors` (as the
    Python package patsy uses the term) by splitting on `:` characters.

    Parameters
    ----------
    formula : string
        The formula string to be split.
    reduce_to_vars : bool, optional
        If True, then each factor will be further split into Python variable
        names, with extraneous characters discarded.  In this scenario, the
        setList( ) representing the term contains all substrings representing
        valid Python variable names occuring in any factor in the term.
        * This is done without regard to whether or not a variable actually
        exists by the name in any given scope.
        * Python reserved keywords are discarded along with extraneous
        characters, but valid Python variable names that are the name of a
        builtin are included.

    Returns
    -------
    setList( ) of setList( )s
    '''
    ## Mask brackets so that only patsy operators are processed.
    mask = mask_brackets( formula )
    start = 0
    final = False
    patsySet = setList( )
    wall = splitter.search( mask, start )
    ## Treating `+`, `-`, and `~` as walls between terms,
    ## create a set of factors in each term, splitting on ':'.
    while wall != None or final == False:
        if wall == None:
            final = True
            endpos = wallpos = len( formula )
        else:
            wallpos = wall.start( )
            endpos = wall.end( )
        factorStr = formula[start:wallpos]
        factorList = factorStr.split( ':' )
        factorList2 = setList( )
        for f in factorList:
            f = despace( f )
            if reduce_to_vars == True:
                factorList2.extend( _vars_in_factor( f ) )
            else:
                factorList2.append( f )
        patsySet.append( factorList2 )
        start = endpos
        if wall != None:
            wall = splitter.search( mask, start )
    return( patsySet )

poly_rs = r' *([a-zA-Z_][a-zA-Z_0-9]*(?![a-zA-Z_0-9\(\{\[])) *\*\* *([0-9]+) *'
polypattern   = re.compile( poly_rs )
polypattern_I = re.compile( r'I\(' + poly_rs + r'\)' )

def pre_rstr( name=None ):                        # initial underscore omitted
    if name == None:                              # for brevity, treat as a
        return( r'(?:L(?:[0-9]+))@' )             # private function.
    return( r'(?:L(?P<' + name + r'>[0-9]+))@' )

prefixed_varpattern_rstr = r'(?:' + pre_rstr( ) + r')?' + varpattern_rstr
prefixed_varpattern = re.compile( prefixed_varpattern_rstr )
has_prefix_pat = re.compile( r'(?:' + pre_rstr( r'lag' ) + r''')
                               (?P<column_name>''' + varpattern_rstr + r')',
                               re.X )
prefixed_pat = re.compile( r'(?:' + pre_rstr( r'lag' ) + r''')?
                             (?P<column_name>''' + varpattern_rstr + r')',
                             re.X )
powerpattern = re.compile( r'\ *(?:' + pre_rstr( r'lag' ) + r''')?
                                (?P<column_name>''' + varpattern_rstr + r''')
                   (?:\ *\*\*\ *(?P<power>[0-9]+)\ *)?''', re.X )
###############################################################################
crosspower = re.compile( r'(?:' + pre_rstr( r'lag_a' ) + r''')?
                           (?P<column_a>''' + varpattern_rstr + r''')
              (?:\ *\*\*\ *(?P<power_a>[0-9]+)\ *)?
               \ *\*\ *''' + # multiplication
                         r'(?:' + pre_rstr( r'lag_b' ) + r''')?
                           (?P<column_b>''' + varpattern_rstr + r''')
              (?:\ *\*\*\ *(?P<power_b>[0-9]+)\ *)?''', re.X )
###############################################################################
cross_Aor_power = re.compile( r'(?:' + pre_rstr( r'lag_a' ) + r''')?
                                (?P<column_a>''' + varpattern_rstr + r''')
                   (?:\ *\*\*\ *(?P<power_a>[0-9]+)\ *)?
                  \ *\*?\ *''' + # multiplication
                              r'(?:' + pre_rstr( r'lag_b' ) + r''')?
                                (?P<column_b>''' + varpattern_rstr + r''')?
                   (?:\ *\*\*\ *(?P<power_b>[0-9]+)\ *)?''', re.X )
###############################################################################
crosspattern = re.compile( r'\ *(?P<column_a>' + varpattern_rstr + r''')
                   \ *\*\ *''' + # multiplication
                           r'\ *(?P<column_b>' + varpattern_rstr + r')\ *',
                           re.X )

def _vars_in_factor( factor ):
    '''Identifies valid Python variable names.

    This includes those used by builtins, but excludes reserved keywords.  The
    function's name refers to "factors" as patsy uses the term, as the function
    dates back to when patsy was used more extensively in the project for which
    mathDict was developed.
    '''
    ret = list( )
    for mobj in varpattern.finditer( factor ):
        if not keyword.iskeyword( mobj.group( ) ):
            ret.append( mobj.group( ) )
    return( ret )

impr_space_pattern = re.compile( r'([a-zA-Z_0-9]+) +([a-zA-Z_0-9]+)' )
def formulas_match( formA, formB ):
    '''Determines whether or not two formula strings are likely to be the same
    formula despite differences in the order of terms and/or spacing.
    '''
    def re_sort( item ):
        '''Divides a string into elements by splitting on single asterisks,
        sorts elements alphabetically, and reassembles the string by joining
        the elements with an asterisk.

        This way, terms consisting of multiple python variables multiplied
        together compare as equal regardless of the order in which the python
        variables are listed.
        '''
        strList = [str( x ) for x in _regex_split( item[0], one_star )]
        strList.sort( )
        item[0] = '*'.join( strList )
        return( item )
    def prep( string ):
        mobj = impr_space_pattern.search( string )
        if mobj and not (keyword.iskeyword( mobj.group( 1 ) ) or \
           keyword.iskeyword( mobj.group( 2 ) )):
            raise ValueError( 'There is an improper space in `%s`.' % string )
        return( string.split( '~' ) )
    def terms_match( A, B ):
        '''Separates each formula into a set( ) of terms, with each term
        represented as a frozenset( ).  Terms that are a product of multiple
        subterms have those subterms sorted in alphabetical order, but terms
        are otherwise left intact.
        '''
        termsA = _patsy_terms( A, reduce_to_vars=False )
        termsB = _patsy_terms( B, reduce_to_vars=False )
        termsA._re_sort = re_sort
        termsB._re_sort = re_sort
        return( termsA.as_fsets == termsB.as_fsets )
    def power_match( A, B ):
        '''Rejects the equality of the two formulas if they contain the same
        variable raised to different powers.

        Returns
        -------
        False
            If there is one or more variables raised to different powers in the
            different formulas.
        True
            If the equality of the two formulas cannot be rejected on this
            narrow basis.
        '''
        mobj_A = powerpattern.fullmatch( mask_brackets( A ) )
        if mobj_A == None:
            return( True )
        dict_A = masked_dict( A, mobj_A )
        mobj_B = powerpattern.fullmatch( mask_brackets( B ) )
        if mobj_B == None:
            return( False )
        dict_B = masked_dict( B, mobj_B )
        if dict_A['lag'] != dict_B['lag'] or \
           not terms_match( dict_A['column_name'], dict_B['column_name'] ) or \
           dict_A['power'] != dict_B['power']:
            return( False )
        return( True )
    formA = prep( formA )
    formB = prep( formB )
    if len( formA ) != len( formB ):
        return( False )
    for i in range( len( formA ) ):
        if not power_match( formA[i], formB[i] ):
            return( False )
        if not terms_match( formA[i], formB[i] ):
            return( False )
    return( True )

def _soft_in( checking, container, else_unchanged=True, else_val=None ):
    '''Checks for membership in a collection, without requiring exact string
    equality.

    If a member of the collection is determined to be a match based on
    formulas_match( ) finding that the strings are likely to be the same
    formula, then a tuple consisting of the formula as represented by the
    collection member, followed by True.

    Otherwise, it returns a tuple with either the original value for which
    membership is being tested or the value of the else_val parameter if set,
    followed by False.
    '''
    ## Currently only used once in mathDictHypothesis.add( ).
    if checking in container:
        return( checking, True )
    for member in container:
        if formulas_match( checking, member ):
            return( member, True )
    if else_unchanged:
        else_val = checking
    return( else_val, False )

def termString( formula, termList ):
    '''Returns the subset of terms in `termList` that occur in `formula`.
    '''
    termString = ''
    for t in termList:
        if has_term( formula, t ):
            if len( termString ) > 0:
                termString += ', '
            termString += t
    return( termString )

def _mapper( ProcessQueue,
             ReturnQueue,
             SharedDataArray,
             mDictCfg,
             func,
             placement,
             number_results=False ):
    '''Used by mathDict.iter_map( ) and .map( ) as a wrapper around the
    function to be mapped to the rows of the mathDict( ).

    See mathDict.iter_map( ) for usage details.
    '''
    mDict = mDictCfg.rebuild( SharedDataArray )
    matrix = mDict[:]
    QueueObject = ProcessQueue.get( )
    while QueueObject != 'Terminate.':
        rid, args, kwargs = QueueObject
        if isinstance( placement, int ):
            if len( args ) < placement:
                raise ValueError( 'Not enough positional arguments to insert '
                                  'the matrix at %d.' % placement )
            pargs = list( args[:placement] )
            pargs.append( matrix )
            pargs.extend( args[placement:] )
        elif isinstance( placement, str ):
            kwargs[placement] = matrix
            pargs = args
        ReturnQueue.put( (rid, func( *pargs, **kwargs )) )
        QueueObject = ProcessQueue.get( )
    ReturnQueue.put( 'Terminated.' )

class CategoryError(Exception):
    '''Raised by categorizedSetDict( ) when an error results from an invalid
    category as opposed to an invalid key or value that would raise a KeyError
    or ValueError.
    '''
    pass

class UnsupportedColumn(Warning):
    '''Raised by mathDict( ) when .add( ) or .add_from_RHS( ) is used in an
    attempt to add a string as a column that is not understood as a column by
    mathDict( ).

    Attributes
    ----------
    msg, args[0] : string
        Error message.
    columns, args[1] : list
        Lists the column or columns that are not understood by mathDict( ).
    LHS : string
        The left-hand-side of a formula string provided to .add_from_RHS( )
        when the formula string contained at least one tilde ('~') character.
    '''
    def __init__(self, *args, LHS=None ):
        Warning.__init__( self, *args )
        if len( args ) > 0:
            self.msg = self.args[0]
        if len( args ) > 1:
            self.columns = self.args[1]
        self.LHS = LHS

class mathDictKeyError(KeyError):
    '''Subclass of KeyError used in error handling to distinguish between
    calling mathDict( ).__getitem__( ) with an invalid key/index, resulting in
    mathDictKeyError, or a facially-valid key/index the handling of which
    causes a KeyError for some other reason.
    '''
    pass

class RankError(Warning):
    '''Raised by mathDictHypothesis( ) when mathDictHypothesis( ).add( ) is
    able to determine that adding a hypothesis about the specified column would
    result in a matrix of insufficient rank for computing an F statistic
    evaluating the hypothesis.

    To enable mathDict's ability to anticipate matrices of less than full rank,
    use the .terms termSet( ) attribute of mathDict( ) to inform
    mathDictHypothesis( ) which terms in which forms are dummy variables.  This
    has been shown in profiles to be substantially more efficient than
    mathematically determining the impact of the additional column on the rank
    of the matrix.

    Example
    -------
    >>> RA, MD = mathDictMaker( [details omitted] ).make( )
    >>> MD.terms = termSet( [details omitted] )
    >>> MD.hypothesis.add( [details omitted] )
    >>> # Link the mathDict( ) to an existing termSet( ) using simple attribute
    >>> # assignment.
    '''
    pass

class setList(UserList):
    '''List that eliminates redundant list items and implements set comparison
    methods.
    '''
    def __init__(self, values=None ):
        UserList.__init__( self )
        if values != None:
            self.extend( values )
        if isinstance( values, (set, frozenset, type( dict( ).keys( ) )) ):
            self.data.sort( )

    def __setitem__(self, key, value ):
        if value in self.data:
            self.lastSIOutcome = False
        else:
            if key == len( self.data ):
                self.data.append( value )
            else:
                self.data[key] = value
            self.lastSIOutcome = True

    def append(self, value ):
        if value in self.data:
            return( False )
        else:
            self.data.append( value )
            return( True )

    def extend(self, values ):
        return( self.update( values ) )

    def add(self, value ):
        return( self.append( value ) )

    def discard(self, value ):
        if value in self.data:
            self.data.remove( value )
            return( True )
        else:
            return( False )

    def pop(self, *, value=None, index=None ):
        if index != None and not isinstance( index, int ):
            raise KeyError( 'Index must be specified as an integer.', index )
        if value != None and index != None:
            raise KeyError( 'Pop can remove an item by index or value but not'
                            ' both.  Value specified: %s, index specified: %d.'
                            % (value, index) )
        if index != None:
            return( self.data.pop( index ) )
        elif value != None:
            if not value in self.data:
                raise KeyError( '`%s` not in %s.' % (value, self) )
            i = self.data.index( value )
            return( self.data.pop( i ) )
        else:
            raise KeyError( 'One of `index` or `value` must be specified by '
                            'keyword arguement when calling setList.pop( ).' )

    def update(self, values ):
        if isinstance( values, (set, frozenset, type( dict( ).keys( ) )) ):
            values = list( values )
            values.sort( )
        counter = 0
        for v in values:
            if self.append( v ) == True:
                counter += 1
        return( counter )

    def difference(self, other ):
        new = setList( )
        for value in self:
            if not value in other:
                new.append( value )
        return( new )

    def union(self, other ):
        new = setList( )
        new.update( self )
        new.update( other )
        return( new )

    def intersection(self, other ):
        new = setList( )
        for value in self:
            if value in other:
                new.append( value )
        return( new )

    def issubset(self, other ):
        for value in self:
            if not value in other:
                return( False )
        return( True )

    def issuperset(self, other ):
        for value in other:
            if not value in self:
                return( False )
        return( True )

    def symmetric_difference(self, other ):
        new = self.difference( other )
        for value in other:
            if not value in self:
                new.append( value )
        return( new )

    @staticmethod
    def _re_sort( item ):
        ## Exists so that it can be replaced by assignment on specific
        ## instances.
        return( item )

    @property
    def as_fsets(self):
        '''set( ) of frozenset( )s of the items in each setList( ) member.  The
        use of frozenset( )s enables the set( ) to contain otherwise non-
        hashable objects.  Useful for order-insensitive equality testing.
        '''
        ret = set( )
        for item in self.data:
            ret.add( frozenset( self._re_sort( item ) ) )
        return( ret )

class typedDict(dict):
    '''dict( ) that is restricted to entries consisting of values of a
    specified type.

    typedDict also supports default values whereby new entries are created by
    deepcopy( )ing an object as opposed to creating a new instance of a class,
    and supports a write-once mode in which keys that have a value associated
    with them cannot be changed, but values that are mutable objects may still
    mutate.

    Each item has an integer key, and may also have a string key associated
    with it, but a string key is not required.  I.e., there is a (zero-or-one)-
    to-one relationship between string keys and dictionary entries, as well as
    a one-to-one relationship between integer keys and dictionary entries.

    Integer keys are not preserved when typedDict( ) is copied.
    '''

    def __init__(self, typeRequirement, writeOnce=False, default=None ):
        '''Creates a typedDict( ) instance.

        Parameters
        ----------
        typeRequirement : type
            Dictionary entries will only be accepted if they satisfy
            `isinstance( obj, typeRequirement )`.
        writeOnce : bool
            If True, then once a dictionary entry has been created for a key,
            the dictionary entry cannot be changed.  If the entry consists of a
            mutable object, the object may still mutate.
        default : object of type `typeRequirement`, optional
            If set, then attempting to access a dictionary entry that does not
            yet exist will result in a deepcopy of this object being used to
            create an entry for the requested key.
        '''
        self.typeRequirement = typeRequirement
        self.writeOnce = writeOnce
        self.strKeys = OrderedDict( )
        self._intAvail = 0
        if default != None:
            if isinstance( default, typeRequirement ):
                self.default = default
            else:
                raise TypeError( 'The default item must be of the required '
                                 'type.' )
        self.pickle = False

    def __deepcopy__(self, memo_dict ):
        if 'default' in self.__dir__( ):
            d = deepcopy( self.default, memo_dict )
        else:
            d = None
        new = typedDict( self.typeRequirement, self.writeOnce, d )
        new.union_update( self )
        return( new )

    def __setitem__(self, key, val ):
        if not 'pickle' in self.__dir__( ):
            if isinstance( key, int ):
                dict.__setitem__( self, key, val )
                return( key )
            else:
                raise KeyError( 'Only integer keys are supported during '
                                'unpickling.' )
        ## Translate string keys into integer keys.
        if key in self:
            if self.writeOnce == True:
                raise KeyError( 'This typedDict( ) is write-once.  The key, '
                                '`%s`, has already been set.' % key )
            else:
                intKey = self._int_key( key )
        ## For new string keys: find an available integer key.
        elif isinstance( key, str ) and \
             isinstance( val, self.typeRequirement ):
            while self._intAvail in self:
                self._intAvail += 1
            intKey = self._intAvail
            self.strKeys[key] = intKey
        ## Having ascertained the appropriate integer key, set the item.
        elif isinstance( key, int ):
            intKey = key
        if isinstance( val, self.typeRequirement ):
            dict.__setitem__( self, intKey, val )
            return( intKey )
        else:
            raise TypeError( 'This typedDict( ) has been configured to only '
                           'accept list items of type %s.  Cannot accept a %s.'
                           % (str( self.typeRequirement ),
                              str( type( val ) )) )

    def __getitem__(self, key ):
        ik = self._int_key( key )
        if ik == None:
            return( self.__missing__( key ) )
        return( dict.__getitem__( self, ik ) )

    def __contains__(self, key ):
        key = self._int_key( key )
        return( dict.__contains__( self, key ) )

    def __missing__(self, key ):
        if 'default' in self.__dir__( ) and \
           isinstance( self.default, self.typeRequirement ):
            newObj = deepcopy( self.default )
        else:
            newObj = self.typeRequirement( )
        typedDict.__setitem__( self, key, newObj )
        return( self.__getitem__( key ) )

    def itemLength(self, key ):
        '''Only checks the length of an entry if the entry already exists,
        otherwise returns 0.

        Parameters
        ----------
        key : int or string
            A key for a dictionary entry that need not exist.

        Returns
        -------
        int
            If there already exists a dictionary entry with the requested key,
            the length of the entry is returned.  If there is no dictionary
            entry already existing with the requested key, then 0 will be
            returned without creating an entry for the key.
        '''
        if key in self:
            key = self._int_key( key )
            return( len( self[key] ) )
        else:
            return( 0 )

    def __getstate__(self):
        tR = self.typeRequirement
        wO = self.writeOnce
        strK = self.strKeys
        _intA = self._intAvail
        if 'default' in self.__dir__( ):
            d = self.default
        else:
            d = None
        dataDict = {}
        for k, v in self.items( ):
            dataDict[k] = v
        attrDict = dict( )
        for key, value in vars( self ).items( ):
            if key.startswith( '_s_' ):
                attrDict[key] = value
        tple = (tR, wO, strK, _intA, d, dataDict, attrDict)
        return( tple )

    def __setstate__(self, state ):
        tR, wO, strK, _intA, d, dataDict, attrDict = state
        self.typeRequirement = tR
        self.writeOnce = wO
        if d != None:
            self.default = d
        self._intAvail = _intA
        for k, v in dataDict.items( ):
            self[k] = v
        self.strKeys = dict( )
        for k, v in strK.items( ):
            self.strKeys[k] = v
        for k, v in attrDict.items( ):
            if k.startswith( '_s_' ):
                setattr( self, k, v )
        self.pickle = False
        return( )

    def _int_key(self, key ):
        if isinstance( key, int ):
            return( key )
        elif isinstance( key, str ):
            if key in self.strKeys:
                return( self.strKeys[key] )
        else:
            raise KeyError( 'typedDict( ) only supports string keys and their '
                            'integer indexes.' )

    def keys(self, key_type=None ):
        '''Returns a setList( ) of keys for which there currently exists an
        entry.

        Parameters
        ----------
        key_type : {None, 'integer', 'string', 'union'}, optional
            The type of keys to return.  If `key_type==None` and at least one
            entry has a string key, then only string keys will be returned.
            Otherwise if `key_type==None`, integer keys will be returned.  If
            `key_type=='union'`, then a setList( ) consisting of both integer
            and string keys will be returned.

        Returns
        -------
        setList( )
        '''
        if key_type == None:
            if len( self.strKeys.keys( ) ) == 0:
                key_type = 'integer'
            else:
                key_type = 'string'
        if key_type == 'string':
            return( setList( self.strKeys.keys( ) ) )
        elif key_type == 'integer':
            return( setList( dict.keys( self ) ) )
        elif key_type == 'union':
            return( setList( self.strKeys.keys( )
                            ).union( dict.keys( self ) ) )

    def pop(self, key ):
        '''Returns and removes the entry associated with the specified key.
        Accepts both integer and string keys.
        '''
        ik = self._int_key( key )
        if isinstance( key, str ):
            self.strKeys.pop( key )
        return( dict.pop( self, ik ) )

    def update(self, other ):
        '''Copies entries in `other` into this typedDict( ), replacing existing
        entries that use the same key.
        '''
        for key in other.keys( ):
            n = self.__setitem__( key, other[key] )

    def union_update(self, other ):
        '''Similar to dict( ).update( other ) except that for keys with which
        an entry is associated in both this typedDict( ) and `other`, the new
        entry will be this typedDict[key].union( other[key] ).
        '''
        sk = setList( self.strKeys.keys( ) )
        ok = setList( other.keys( ) )
        for key in sk.union( ok ):
            self[key].update( other[key] )
        for key in ok.difference( sk ):
            self[key] = deepcopy( other[key] )

class categorizedSetDict(typedDict):
    '''Ordered sets stored in a dict( ) in which each set and each set member
    potentially belongs to one or more category.

    Sets and set members can be retrieved by category.  Categories designated
    as mutually exclusive restrict category membership to sets and set members
    without conflicting categories.

    Attributes
    ----------
    mutually_exclusive : set
        set( ) of frozenset( )s of categories, where each category in a
        frozenset( ) and all other categories in the same frozenset( ) are are
        considered mutually exclusive.

    Notes
    -----
    Categories are assumed to be identified by strings, so the code is only
    tested using string-identified categories.  However, this is not strictly
    enforced.
    '''
    def __new__( cls, *args, **kwargs ):
        obj = super( categorizedSetDict, cls ).__new__( cls, *args, **kwargs )
        obj._s_ctg_keys = dict( )
        obj._s_ctg_values = dict( )
        obj.mutually_exclusive = setList( )
        typedDict.__init__( obj, setList )
        return( obj )

    def __init__(self, singular_category=None ):
        '''Creates a categorizedSetDict( ) instance.

        Parameters
        ----------
        singular_category : string, optional
            When a set is instantiated with a single item (as opposed to a
            sequence of one member), this category is assigned to the set.
        '''
        self.singular_category = singular_category

    def __setitem__(self, key, value ):
        '''Sets dict( ) entries.

        Parameters
        ----------
        key : str
            String identifying a set.
        value : str or tuple (list-like : set members[, list-like : set member
        categories][, set : whole-set categories )
            list( ), set( ), or setList( ) of set members associated with the
            key, alone or combined with categories that apply to individual set
            members and/or categories that apply to the whole set.

        Raises
        ------
        CategoryError
            If an attempt is made to associate a set or set member with a
            category that is already associated with a mutually-exclusive
            category.  In this scenario, the entire .__setitem__( ) operation
            will fail and the categorizedSetDict( ) will not be altered.

        Examples
        --------
        >>> c = categorizedSetDict( )
        >>> c['vocab'] = ['apple', 'bee', 'cabin']
        >>> c['vocab']
        ['apple', 'bee', 'cabin']
        >>> c.get_categories( key='vocab', value='apple' )
        {None}

        >>> c['vocab'] = (['apple', 'bee', 'cabin'],
        >>>               {'words'})
        >>> c.get_categories( key='vocab', value='apple' )
        {'words'}

        >>> c['vocab'] = (['apple', 'bee', 'cabin'],
        >>>               [{'food'}, {'animal'}, {'building'}])
        >>> c.get_categories( key='vocab', value='apple' )
        {'food'}

        >>> c['vocab'] = (['apple', 'bee', 'cabin'],
        >>>               [{'food'}, {'animal'}, {'building'}],
        >>>               {'words'})
        >>> c.get_categories( key='vocab', value='apple' )
        {'food', 'words'}
        '''
        ## Convert entries without category information and 2-tuples with
        ## either whole-key categories or single-value categories into
        ## 3-tuples.
        if value == None:
            if key in self.keys( ):
                self.__delitem__( key )
            return( )
        if isinstance( value, (list, set) ):
            value = setList( value )
        if isinstance( value, str ):
            value = (setList( [value] ), [], {self.singular_category})
        elif isinstance( value, setList ):
            value = (value, [], {None,})
        if isinstance( value, (tuple, list) ) and len( value ) == 2:
            if not isinstance( value[0], setList ):
                value = (setList( value[0] ), value[1])
            if isinstance( value[1], list ):
                value = (value[0], value[1], set( ))
            elif isinstance( value[1], set ):
                value = (value[0], [], value[1])
        ## Take 3-tuples, perform some validation, and then store the data.
        if isinstance( value, tuple ) and len( value ) == 3:
            if isinstance( value[0], list ):
                value = (setList( value[0] ), value[1], value[2])
            if isinstance( value[0], setList ) and \
               isinstance( value[1], list ) and \
               isinstance( value[2], set ):
                ## Validate:
                if isinstance( value[1], list ) and \
                   len( value[1] ) > len( value[0] ):
                    raise ValueError( 'Categories list has a length of %d.  '
                          'It cannot be longer than the items list, which has '
                          'a length of %d.' % (len( values[1] ),
                                               len( values[0] )) )
                for i in range( len( value[1] ) ):
                    if value[1][i] == None:
                        value[1][i] = set( )
                    elif not isinstance( value[1][i], (set, setList) ):
                        raise TypeError( 'Items in the value category sequence'
                              ' must be sets, not %s.' % type( value[1][i] ) )
                ## Make backup of current values for key
                B_itm = deepcopy( self[key] )
                if key in self._s_ctg_keys:
                    B_key = deepcopy( self._s_ctg_keys[key] )
                else:
                    B_key = None
                if key in self._s_ctg_values:
                    B_val = deepcopy( self._s_ctg_values[key] )
                else:
                    B_val = None
                try:
                    ## Attempt to store new data
                    self.__missing__( key )
                    typedDict.__setitem__( self, key, value[0] )
                    self.set_categories( *value[2], key=key )
                    for i in range( len( value[1] ) ):
                        self.set_categories( *value[1][i],
                                             key=key,
                                             value=value[0][i] )
                except CategoryError as e:
                    ## Use backups to restore prior value if attempt was
                    ## unsuccesful.
                    typedDict.__setitem__( self, key, B_itm )
                    self._s_ctg_keys[key]   = B_key
                    self._s_ctg_values[key] = B_val
                    raise e
                return( )
        raise TypeError( 'Value is not a supported type.  '
                         'Type: %s, length: %d.'
                         % (type( value ), getattr( value, '__len__', 0 )) )

    def get_categories(self, key, value=None ):
        '''Returns the set of categories associated with the whole set or a
        particular member thereof.

        Parameters
        ----------
        key : str
            The key associated with the set about which the inquiry is being
            made.
        value, optional
            If specified, the categories specifically associated with the
            specified set member are returned along with the categories
            associated with the whole set.

        Returns
        -------
        categories : set
            The set of all categories associated with the specified key or with
            the specified key/value pair.  If there is a set identified by the
            key `key`, then the set of categories associated with it will be
            returned regardless of whether or not `value` is in the set `key`.
        None
            If there is no entry with `key`'s value as its key.
        '''
        if key not in self:
            return( None )
        if value == None or value not in self[key] or \
           self[key].index( value ) >= len( self._s_ctg_values[key] ):
            return( self._s_ctg_keys[key] )
        else:
            return( self._s_ctg_keys[key].union(
                self._s_ctg_values[key][self[key].index( value )] ) )

    def is_a(self, category, key, value=None ):
        '''Returns True if the specified key or key/value pair is associated
        with `category`, or False otherwise, so long as there is an entry with
        `key`'s value as its key.  If not, None is returned.  The set
        identified by `key` need not contain `value`.
        '''
        categories = self.get_categories( key, value )
        if categories == None:
            return( None )
        return( category in categories )

    def is_None(self, key, value=None ):
        '''Returns True if the specified key or key/value pair is not
        associated with any category, or False otherwise.  See .is_a( ) for
        handling of non-existant values.
        '''
        categories = self.get_categories( key=key, value=value )
        if not categories == set( ) and not categories == set( [None] ):
            return( False )
        return( True )

    def set_category(self, category, *, key=None, value=None,
                                        keys=None,
                                        items=None ):
        '''Associates key(s) and/or key/value pairs with the specified
        category.

        Parameters
        ----------
        category : string
            The category to associate with the key(s) and/or key/value pairs.
        key : str, optional
            If key but not value is specified, then this key will be associated
            with the specified category.
        value, optional
            If key and value are both specified, then this key/value pair will
            be associated with the specified category.
        keys : Sequence
            The keys in this sequence will be associated with the specified
            category.
        items : dict
            The key/value pairs in this dict( ) will be associated with the
            specified category.  Each value in this dict( ) can be either a
            single set member or a sequence of set members to associate with
            the specified category.

        Raises
        ------
        CategoryError
            If the category specified and one or more categories already
            associated with one or more of the specified key(s) or key/value
            pairs are considered mutually exclusive.
        KeyError
            If a specified key/value pair does not identify an existing set
            member.  Note that assigning a category to a key for which there is
            no pre-existing entry results in the creation of a default entry
            instead of of a KeyError.
        '''
        def set_key( key, category ):
            ## Validates and sets a single category for a single key.
            if key not in self.keys( ):
                self.__missing__( key )
            for ctgset in self.mutually_exclusive:
                if category in ctgset:
                    if len( ctgset.intersection( self._s_ctg_keys[key] )\
                                  .difference( set( [category] ) ) ) > 0:
                        raise CategoryError( "Cannot add category '%s' to "
                              "['%s'] because it already has a mutually "
                              "exclusive category." % (category, key) )
                    for val in self._s_ctg_values[key]:
                        if len( ctgset.intersection( val )\
                                      .difference( set( [category] ) ) ) > 0:
                            raise CategoryError( "Cannot add category '%s' to "
                                  "['%s'] because one or more values already "
                                  "has a mutually exclusive category."
                                  % (category, key), ctgset )
            self._s_ctg_keys[key].add( category )
        def set_val( key, val, category ):
            ## Validates & sets a single category for a single key/value pair.
            if val in typedDict.__getitem__( self, key ):
                index = typedDict.__getitem__( self, key ).index( val )
            else:
                raise KeyError( "Set identified by key '%s' has no value '%s'."
                      "  It has: %s."
                      % (key, val, typedDict.__getitem__( self, key )) )
            while len( self._s_ctg_values[key] ) <= index:
                self._s_ctg_values[key].append( set( ) )
            for ctgset in self.mutually_exclusive:
                if category in ctgset:
                    if len( ctgset.intersection( self.get_categories( key=key,
                                                                 value=val ) )\
                                  .difference( set( [category] ) ) ) > 0:
                        raise CategoryError( "Cannot add category '%s' to "
                              "['%s']'%s' because it and an existing category "
                              "are mutually exclusive."
                              % (category, key, val), ctgset )
            self._s_ctg_values[key][index].add( category )
        #######################################################################
        ## Body of method starts here:
        if key:
            if value:
                set_val( key, value, category )
            else:
                set_key( key, category )
        if isinstance( keys, Sequence ) and not isinstance( keys, str ):
            for key in keys:
                set_key( key, category )
        elif keys != None:
            raise KeyError( '`keys` must be a non-string Sequence.  Use the '
                            'singular `key` to set the category for one key.' )
        if isinstance( items, dict ):
            for key, values in items.items( ):
                if not isinstance( values, Sequence ) or \
                   isinstance( values, str ):
                    values = [values]
                for value in values:
                    set_val( key, value, category )
        elif items != None:
            raise KeyError( '`items` must be a dict( ).' )

    def set_categories(self, *categories, key=None, value=None,
                                          keys=None,
                                          items=None ):
        '''Associates key(s) and/or key/value pairs with each of the specified
        categories.

        .set_categories( ) makes a separate call to .set_category( ) for each
        category.  As such, calls to .set_categories( ) are not atomic.  For
        atomic behavior, use .__setitem__( ) instead of .set_categories( ).

        Parameters
        ----------
        positional arguments
            Categories to associate with the key(s) and/or key/value pairs.
        key : str, optional
            If key but not value is specified, then this key will be associated
            with the specified category.
        value, optional
            If key and value are both specified, then this key/value pair will
            be associated with the specified category.
        keys : Sequence
            The keys in this sequence will be associated with the specified
            category.
        items : dict
            The key/value pairs in this dict( ) will be associated with the
            specified category.  Each value in this dict( ) can be either a
            single set member or a sequence of set members to associate with
            the specified category.

        Raises
        ------
        CategoryError
            If the category specified and one or more categories already
            associated with one or more of the specified key(s) or key/value
            pairs are considered mutually exclusive.
        KeyError
            If a specified key/value pair does not identify an existing set
            member.  Note that assigning a category to a key for which there is
            no pre-existing entry results in the creation of a default entry
            instead of of a KeyError.
        '''
        for c in categories:
            self.set_category( c, key=key, value=value,
                                  keys=keys,
                                  items=items )

    def del_category(self, category, *, key=None, value=None,
                                        keys=None,
                                        items=None ):
        '''Disassociates the specified category from the specified key(s) and/
        or key/value pairs.  See .set_category( ) documentation for usage.

        If there is no existing association between `category` and a specified
        key and/or key/value pair, no Exception is raised.

        Raises
        ------
        CategoryError
            If an attempt is made to disassociate an individual set member with
            a category that is associated with the whole set.
        KeyError
            If a specified key/value pair does not identify an existing set
            member.
        '''
        def del_key( key, category ):
            ## Disassociates a single key from a single category.
            self._s_ctg_keys[key].discard( category )
        def del_val( key, val, category ):
            ## Disassociates a single key/value pair from a single category.
            if val in typedDict.__getitem__( self, key ):
                index = typedDict.__getitem__( self, key ).index( val )
            else:
                raise KeyError( "Set identified by key '%s' has no value '%s'."
                                "  It has: %s."
                                % (key,
                                   val,
                                   typedDict.__getitem__( self, key )) )
            while len( self._s_ctg_values[key] ) <= index:
                self._s_ctg_values[key].append( set( ) )
            self._s_ctg_values[key][index].discard( category )
        #######################################################################
        ## Body of method starts here:
        if key:
            if value:
                if self.is_a( category, key=key ):
                    raise CategoryError( "Attempted disassociation of ['%s']%s"
                          " and category '%s' when said category is associated"
                          " with the whole set ['%s']."
                          % (key, value, category, key) )
                del_val( key, value, category )
            else:
                del_key( key, category )
        if isinstance( keys, Sequence ) and not isinstance( keys, str ):
            for key in keys:
                del_key( key, category )
        elif keys != None:
            raise KeyError( '`keys` must be a non-string Sequence.  Use the '
                  'singular `key` to disassociate the category for one key.' )
        if isinstance( items, dict ):
            for key, value in items.items( ):
                del_val( key, value, category )
        elif items != None:
            raise KeyError( '`items` must be a dict( ).' )

    def keys_categorized(self, category ):
        '''Returns the keys that are associated with the specified category.'''
        if category == None:
            return( self._keys_categorized_none( ) )
        ret = setList( )
        for key in self.keys( ):
            if category in self._s_ctg_keys[key]:
                ret.append( key )
        return( ret )

    def items_categorized(self, category ):
        '''Returns the key/value pairs consisting of keys associated with the
        specified category with all of their set members, as well as set
        members that are individually associated with the specified category
        (along with their keys).'''
        if category == None:
            return( self._items_categorized_none( ) )
        ret = typedDict( setList )
        for key in self.keys( ):
            if key in self._s_ctg_keys.keys( ):
                if category in self._s_ctg_keys[key] or category == ('ALL',):
                    ret[key] = typedDict.__getitem__( self, key )
                if key in self._s_ctg_values.keys( ):
                    if key not in ret.keys( ):
                        for i in range( len( self._s_ctg_values[key] ) ):
                            if category in self._s_ctg_values[key][i]:
                                ret[key].append( typedDict.__getitem__( self,
                                                                     key )[i] )
        return( ret )

    def values_categorized(self, category ):
        '''Returns set members from all sets where either the set member or the
        set is associated with the specified category.'''
        items = self.items_categorized( category )
        ret = setList( )
        for value in items.values( ):
            ret.extend( value )
        return( ret )

    def make_mutually_exclusive(self, categories ):
        '''Designates the specified set of categories as mutually exclusive.

        Parameters
        ----------
        categories : set or Sequence
            The set of categories to be considered mutually exclusive.  If this
            is a superset of an existing set of mutually exclusive categories,
            it will replace the existing subset.  If an existing set is a
            superset of this one, then no action is taken.
        '''
        categories = frozenset( categories )
        for ctgset in self.mutually_exclusive:
            if categories.issuperset( ctgset ):
                self.mutually_exclusive.remove( ctgset )
            elif ctgset.issuperset( categories ):
                return( )
        self.mutually_exclusive.add( categories )

    def _keys_categorized_none(self):
        items_none = self._items_categorized_none( )
        return( items_none.keys( ) )

    def _values_categorized_none(self):
        ret = setList( )
        for key in self.keys( ):
            if key not in self._s_ctg_keys.keys( ) or \
               self._s_ctg_keys[key] == None or \
               len( self._s_ctg_keys[key] ) == 0:
                for i in range( len( typedDict.__getitem__( self, key ) ) ):
                    if len( self._s_ctg_values[key] ) <= i or \
                       self._s_ctg_values[key][i] == None or \
                       len( self._s_ctg_values[key][i] ) == 0:
                        ret.append( typedDict.__getitem__( self, key )[i] )
        return( ret )

    def _items_categorized_none(self):
        ret = typedDict( setList )
        for key in self.keys( ):
            if key not in self._s_ctg_keys.keys( ) or \
               self._s_ctg_keys[key] == {None,} or \
               len( self._s_ctg_keys[key] ) == 0:
                for i in range( len( typedDict.__getitem__( self, key ) ) ):
                    if len( self._s_ctg_values[key] ) <= i or \
                       self._s_ctg_values[key][i] == None or \
                       len( self._s_ctg_values[key][i] ) == 0:
                        ret[key].append( typedDict.__getitem__( self,
                                                                key )[i] )
        return( ret )

    def pop(self, key ):
        '''Removes and returns the setList( ) associated with the provided key
        and deletes category information.
        '''
        if key not in self:
            raise KeyError( 'Invalid key: `%s`.' % key )
        self._s_ctg_keys.pop( key )
        self._s_ctg_values.pop( key )
        return( typedDict.pop( self, key ) )

    def __delitem__(self, key ):
        self.pop( key )

    def __missing__(self, key ):
        self._s_ctg_keys[key] = set( )
        self._s_ctg_values[key] = list( )
        return( typedDict.__missing__( self, key ) )

    def keys(self):
        return( setList( values=typedDict.keys( self ) ) )

def get_term_vars( formula ):
    '''Returns a flat setList( ) containing all valid Python variable names in
    the provided string.
    '''
    INTERNAL_NOTE = '''Originally written for termSet( )'s initializing from a
    set of formulas, and moved out of termSet( ) so that it could also be used
    by laggedAccessor.rewrite( ).
    '''
    patsySet = _patsy_terms( formula, reduce_to_vars=True )
    ret = setList( )
    for patsyTerm in patsySet:
        for var in patsyTerm:
            ret.add( var )
    return( ret )

class termSet(categorizedSetDict):
    '''Manages a set of terms, each of which might have multiple
    representations.

    Categories
    ----------
    dummy
        "Dummy" terms or representations of terms in which there are only two
        values.
    Y
        Terms that are used on the LHS of formulas instead of the RHS.  Terms
        that are sometimes used on the LHS and sometimes used on the RHS are
        not supported.  (However, there is no need to make use of the 'Y'
        category in order to make full use of mathDict( ).)
    required_X
        Terms that must be included on the RHS of all formulas derived from
        this termSet( ).
    T
        RHS term(s) representing time/trend.
    '''
    def __init__(self, *args, interactions=False, expand=True, **kwargs ):
        '''Creates a termSet( ) instance.

        Parameters
        ----------
        formulas : Iterable of formula strings
            Iterable of a formula strings from which to extract the terms and
            their forms.  Ex: ['y ~ x**2 + x', 'ln(y) ~ ln(x)'].
        *or*
        terms : dict
            Dictionary in which each term is represented by a key for which the
            value is a sequence of forms in which the term might occur.  Ex:
            {'X': ['X', 'ln(X)']}.  Entries for which the value is a single
            string, e.g. {'d': 'd'},  will be treated as dummy terms.  To avoid
            this when a term has only one form, enclose the string in a list,
            e.g. {'X': ['X']}.
        T : string, optional
            String identifying a single term that represents time/trend.
        '''
        categorizedSetDict.__init__( self, singular_category='dummy' )
        self.make_mutually_exclusive( ['Y', 'required_X'] )
        if 'terms' in kwargs.keys( ):
            self._init_from_terms( **kwargs )
        elif 'formulas' in kwargs:
            self._init_from_formulas( **kwargs )
        else:
            self._init_from_terms( *args, **kwargs )

    def _init_from_formulas(self, formulas ):
        def get_term_reps( formula ):
            patsySet = _patsy_terms( formula )
            ret = typedDict( setList )
            for patsyTerm in patsySet:
                for factor in patsyTerm:
                    for var in _vars_in_factor( factor ):
                        ret[var].append( factor )
            return( ret )
        #######################################################################
        ## Body of method starts here:
        Y = setList( )
        for formula in formulas:
            self.union_update( get_term_reps( formula ) )
            sides = formula.split( '~' )
            Y = Y.union( get_term_vars( sides[0] ) )
        for y in Y:
            self.set_category( 'Y', key=y )

    def _init_from_terms(self, terms, dterms=None, T=None ):
        DEPRECATED_PARAMETERS = '''
        dterms : Iterable, optional
            Iterable of dummy terms that only occur in one form.  Dummy terms
            listed in `dterms` need not also be listed in `terms`.
        T : string, optional
            Same story as dterms, except for the 'T' category and only one term
            can be specified in this manner.

        The `dterms` parameter is deprecated as of the first release of
        Parallel Regression.  It exists to support code that predates
        categorizedSetDict( ) and will be removed in a subsequent release.  Use
        entries in the `terms` dict( ) for which the value is a single string
        to identify single-form dummy terms at initialization time.
        '''
        if not isinstance( terms, dict ):
            raise TypeError( 'The set of real terms must be a dictionary.' )
        if dterms != None and \
           ((not isinstance( dterms, abc.Iterable )) or \
            isinstance( dterms, str )):
            raise TypeError( 'The set of dummy variables must be a non-string'
                             ' iterable.' )
        if (not isinstance( T, str )) and T != None:
            raise TypeError( 'The time index term must be identified by a '
                  'string not in the set of dummy terms and not used as a key '
                  'in the dictionary of real terms.' )
        for term in dterms:
            self[term] = term
        if T:
            self[T] = ([T], {'T',})
        self.update( terms )

    def changeT(self, T ):
        '''Disassociates any term(s) currently associated with the category 'T'
        and associates `T` with the category 'T'.
        '''
        old_T = self.keys_categorized( 'T' )
        for key in old_T:
            self.del_category( 'T', key=key )
        if T:
            if T not in self.keys( ):
                self[T] = (T, {'T',})
            else:
                self.set_category( 'T', key=T )

    def require(self, *args, make=True ):
        '''Associates keys listed in *args with the category 'required_x' if
        `make` is True, or disassociates them if `make` is False.
        '''
        try:
            if isinstance( args, str ):
                if make:
                    self.set_category( 'required_X', key=args )
                else:
                    self.del_category( 'required_X', key=args )
            else:
                for key in args:
                    if make:
                        self.set_category( 'required_X', key=key )
                    else:
                        self.del_category( 'required_X', key=key )
        except CategoryError:
            raise KeyError( 'That key is already in Y.  It cannot be used on '
                            'the LHS and required on the RHS.' )

    def Y(self, key, value=None, make=True ):
        ''' Associates the specified term with the category 'Y' if `make` is
        True, or disassociates it if `make` is False.
        '''
        if key not in self.keys( ):
            self[key] = [key]
        try:
            if make:
                self.set_category( 'Y', key=key, value=value )
            else:
                self.del_category( 'Y', key=key, value=value )
        except CategoryError:
            raise KeyError( 'That key is already required on the RHS.  It '
                 'cannot be used on the LHS while it is required on the RHS.' )

    def dummy(self, key, value=None, make=True ):
        ''' Associates the specified term with the category 'dummy' if `make`
        is True, or disassociates it if `make` is False.
        '''
        if make:
            self.set_category( 'dummy', key=key, value=value )
        else:
            self.del_category( 'dummy', key=key, value=value )

    @property
    def W_term_set(self):
        return( self.keys( ) )

    @property
    def W_termRep_set(self):
        ret = setList( )
        for k in self.keys( ):
            ret.update( self[k] )
        return( ret )

    @property
    def Y_term_set(self):
        return( self.keys_categorized( 'Y' ) )

    @property
    def X_required_set(self):
        return( self.keys_categorized( 'required_X' ) )

    @property
    def all_X_terms_set(self):
        return( self.keys( ).difference( self.keys_categorized( 'Y' ) ) )

    @property
    def dummy_term_set(self):
        return( self.keys_categorized( 'dummy' ) )

    @property
    def dummy_termRep_set(self):
        return( self.values_categorized( 'dummy' ) )

    @property
    def real_term_set(self):
        return( self.keys( ).difference( self.keys_categorized( 'dummy'
                                        ).union( self.keys_categorized( 'T' ) )
                                        ) )

    @property
    def real_termRep_set(self):
        return( self.values_categorized( ('ALL',) ).difference(
                                      self.values_categorized( 'dummy' ).union(
                                           self.values_categorized( 'T' ) ) ) )

    @property
    def other_terms(self):
        return( self.keys( ).difference(
        self.keys_categorized( 'Y' ).union( self.keys_categorized( 'required_X'
                                                                        ) ) ) )

PR_BYTESIZE = 8
PR_NP_INT   = 'i8'
PR_PY_INT   = 'q'
PR_NP_FLT   = 'f8'
PR_PY_FLT   = 'd'

NoneSet = {None, ''}
class laggedAccessor(object):
    '''Allows lagged values to be retrieved from a dict( ) or
    pandas.dataframe( ) object using the same `L#@column_name` notation used by
    other mathDict classes.
    '''
    def __init__(self, data, max_lag=None ):
        '''Initializes the laggedAccessor( ).

        Parameters
        ----------
        data : dict( ) or pandas.dataframe( )
            The collection of columns from which values are to be retrieved.
        max_lag : int, optional
            The maximum number of lags to be provided.  The first `max_lag`
            number of rows will be hidden from each column retrieved through
            the laggedAccessor so that retrieving `column` and retrieving
            `L1@column` results in columns of the same length.  This number
            must either be set explicitly or by using one of the .findMaxLag( )
            or .rewrite( ) methods.
        '''
        self.data = data
        self.max_lag = max_lag
        self.key_map = False
        self.dict_columns = []

    def __getitem__(self, index ):
        '''Returns a row, column, or slice of a column from the linked
        collection of columns.

        Data is always retrieved via a call to .get_column( ), so "row" has the
        meaning it has to that method.

        **Supported Notation**

        ------------------
        int -> returns a dict( ) of columns values for one row

        str -> returns the requested column

        (int or slice, str) tuple -> returns the specified row(s) from the
        requested column
        '''
        if isinstance( index, int ):
            ret = dict( )
            for column in self.dict_columns:
                try:
                    ret[column] = self.get_column( column, row=index )
                except KeyError:
                    pass
            if self.key_map:
                for column in self.key_map:
                    ret[column] = self[index, column]
            return( ret )
        if isinstance( index, tuple ) and \
           isinstance( index[0], (int, slice) ) and \
           len( index ) == 2:
            row = index[0]
            index = index[1]
        else:
            row = None
        if isinstance( index, str ):
            if self.key_map and index in self.key_map:
                mobj_dict = self.key_map[index]
            else:
                mobj_dict = masked_dict( index,
                                         prefixed_pat.fullmatch(
                                                     mask_brackets( index ) ) )
            if mobj_dict == None:
                raise KeyError( '`%s` is not a valid access pattern.' % index )
            elif mobj_dict['lag'] in NoneSet:
                mobj_dict['lag'] = 0
            else:
                mobj_dict['lag'] = int( mobj_dict['lag'] )
            return( self.get_column( row=row, **mobj_dict ) )

    def get_column(self, column_name, lag=0, row=None ):
        '''Returns the requested column or slice thereof.

        Parameters
        ----------
        column_name : string
            The name of the column in the linked collection of columns from
            which to retrieve data.
        lag : int, optional
            The column representing this many lags of `column_name` will be
            returned.  A `lag` of zero refers to the column that is the subset
            of rows of `column_name` in the linked collection starting at the
            index location equal to the `max_lag` value that this
            laggedAccessor( ) is configured to support.
        row : int or slice, optional
           The row or slice to be retrieved.  This refers to row numbers in the
           column identified by the combination of the `column_name` and `lag`
           parameters and the configured `max_lag` value.
        '''
        if isinstance( row, int ) and row < 0:
            raise ValueError( 'Negative row indexes such as the requested row,'
                              ' %d, are not supported.' % row )
        if lag > self.max_lag:
            raise KeyError( '%d exceeds the maximum lag of %d.'
                            % (lag, self.max_lag) )
        a = self.max_lag - lag
        t = len( self.data[column_name] ) - lag
        if row != None:
            ret = self.data[column_name][a:t]
            if 'values' in ret.__dir__( ):
                ret = ret.values
            return( ret[row] )
        return( self.data[column_name][a:t] )

    def findMaxLag(self, formula_string, in_place=False ):
        '''Determines the maximum lag requested for any column in the provided
        formula string(s).

        Parameters
        ----------
        formula_string : string or Sequence
            The formula(s) to be searched for lags.
        in_place : bool, optional
            If True, then the max_lag attribute of this laggedAccessor( ) will
            be updated based on the maximum lag in the provided formula(s).
            Otherwise, a new laggedAccessor( ) instance linked to the same data
            object will be created, and its max_lag attribute will be set.

        Returns
        -------
        laggedAccessor( )
        '''
        if isinstance( formula_string, Sequence ) and \
           not isinstance( formula_string, str ):
            if in_place:
                for formula in formula_string:
                    self.findMaxLag( formula, in_place=True )
                return( self )
            else:
                la_find = deepcopy( self )
                for formula in formula_string:
                    la_find = la_find.findMaxLag( formula, in_place=True )
                return( la_find )
        max_lag = 0
        mobj = prefixed_pat.search( formula_string )
        while mobj != None:
            lag = mobj.groupdict( )['lag']
            if lag in NoneSet:
                lag = 0
            else:
                lag = int( lag )
            max_lag = max( max_lag, lag )
            mobj = prefixed_pat.search( formula_string, mobj.start( ) + 1 )
        max_lag = max( max_lag, val_if_present( self, 'max_lag', 0 ) )
        if in_place:
            self.max_lag = max_lag
            return( self )
        return( laggedAccessor( data=self.data, max_lag=max_lag ) )

    def _find_available_name(self, column_name, lag ):
        '''(for internal use): used by .rewrite( ) to find a name for lagged
        columns that doesn't conflict with a column in the linked data object
        or any other lagged column.
        '''
        def valid_name( check ):
            if not check in self.key_map and not check in self.data.keys( ):
                return( True )
            if self.key_map and \
               self.key_map[check] == {'lag': lag,
                                       'column_name': column_name}:
                return( True )
            return( False )
        check = 'L%d_%s' % (lag, column_name)
        if valid_name( check ):
            return( check )
        for l in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b',
                  'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
            check = 'L%d_%s_%s' % (lag, l, column_name)
            if valid_name( check ):
                return( check )
        raise KeyError( 'Could not find an availabe column name for L%d@%s.'
                        % (lag, column_name) )

    def rewrite( self, formula_string ):
        '''Rewrites as formula string that uses mathDict's lag notation to
        instead use Python variable name aliases for the lagged columns, so
        that the rewritten formula string can be processed by, e.g., patsy.

        Returns
        -------
        (formula_string, laggedAccessor( )) tuple
           The rewritten formula string followed by a new laggedAccessor( )
           instance linked to the same data object containing the lagged column
           aliases.  The new laggedAccessor( ) instance's max_lag attribute is
           the greater of this laggedAccessor's max_lag and the maximum lag
           used in the formula_string.
        '''
        la_rewritten = self.findMaxLag( formula_string )
        la_rewritten.key_map = dict( )
        mobj = has_prefix_pat.search( formula_string )
        while mobj != None:
            mobj_dict = mobj.groupdict( )
            if mobj_dict['lag'] in NoneSet:
                mobj_dict['lag'] = 0
            else:
                mobj_dict['lag'] = int( mobj_dict['lag'] )
            new_name = la_rewritten._find_available_name( **mobj_dict )
            la_rewritten.key_map[new_name] = mobj_dict
            formula_string = formula_string[:mobj.start( )] \
                             + new_name + formula_string[mobj.end( ):]
            mobj = has_prefix_pat.search( formula_string, mobj.start( ) + 1 )
        la_rewritten.max_lag = max( val_if_present( la_rewritten,
                                                    'max_lag',
                                                    0 ),
                                    val_if_present( self,
                                                    'max_lag',
                                                    0 ) )
        la_rewritten.dict_columns = get_term_vars( formula_string )
        return( formula_string, la_rewritten )

class mathDataStore(dict):
    def __init__(self, mathDict=None ):
        self.mathDict = mathDict
        self.padding = dict( )

    @property
    def key_list(self):
        '''Returns the list of columns currently in this data store in sorted
        list form.
        '''
        l = list( self.keys( ) )
        l.sort( )
        return( l )

    @property
    def padding_list(self):
        '''Lists the amount of padding associated with each column in the same
        order as the columns are listed in mathDataStore( ).key_list.
        '''
        ret = list( )
        for key in self.key_list:
            if key in self.padding.keys( ):
                ret.append( self.padding[key] )
            else:
                ret.append( 0 )
        return( ret )

    @property
    def rows(self):
        '''Returns the number of rows that the matrix of original columns has
        or will have, as determined by the first column provided to
        mathDictMaker( ).  All subsequent columns must have the same number of
        rows.  Returns None before the first column is stored.
        '''
        if self.mathDict:
            return( self.mathDict.rows )
        if len( self ) == 0:
            return( None )
        return( len( self[self.key_list[0]] ) )

    @property
    def max_lag(self):
        if self.mathDict:
            max_lag = self.mathDict.max_lag
        else:
            max_lag = 0
        for k, v in self.padding.items( ):
            if k in self.keys( ):
                max_lag = max( max_lag, v )
        return( max_lag )

    @property
    def itemsize(self):
        '''Returns the bytes-per-cell.  Currently hard-coded at eight bytes.'''
        return( PR_BYTESIZE )

    def __setitem__(self, key, value ):
        '''See mathDictMaker( ) documentation.'''
        if (self.mathDict and key in self.mathDict._column_names) or \
           key == 'Intercept':
            raise KeyError( 'The shared data array already has a column named '
                            '%s.  The shared data array cannot be modified.'
                            % key )
        if value == None:
            return( self.__delitem__( key ) )
        if not isinstance( key, str ):
            raise KeyError( 'Keys must be strings because they will be used as'
                            ' column names.' )
        if not isinstance( value, (Sequence, array.array, np.ndarray) ) or \
           isinstance( value, str ):
            raise TypeError( 'All values must be non-string sequences of the '
                             'same length.', type( value ) )
        elif isinstance( value[0], (float) ):
            for v in value:
                if not isinstance( v, (int, float) ):
                    raise TypeError( 'One or more values in column %s is '
                                     'neither an integer nor a float.' % key )
        elif isinstance( value[0], int ):
            for v in value:
                if not isinstance( v, int ):
                    raise TypeError( 'One or more values in column %s is not '
                                     'an integer.' % key )
        elif isinstance( value, np.ndarray ):
            pass
        else:
            raise TypeError( "All values must be ndarrays, sequences of ints, "
                             "or sequences of numbers that start with a float."
                             "  %s starts with neither an int nor a float.  "
                             "'%s' is a %s."
                             % (key, str( value[0] ), type( value[0] )) )
        mobj = varpattern.fullmatch( mask_brackets( key ) )
        if mobj == None:
            raise KeyError( "Keys must be valid Python variable names, alone "
                            "or immediately followed by brackets.  "
                            "'%s' is not." % key )
        if self.rows == None or len( value ) == self.rows:
            if self.mathDict:
                self.mathDict._last_columns = None
            dict.__setitem__( self, key, value )
        else:
            raise ValueError( 'The sequence you are trying to add has a '
                              'different length, %d, than existing sequence(s)'
                              ', %d.' % (len( value ), self.rows), key )

    def savePaddedColumn(self, key, value, padding=None ):
        '''Stores a column that is shorter than the number of rows of other
        columns in this mathDataStore( ).

        Zeros are prepended to the column before it is store so that it is the
        same length as the other columns.  The number of prepended zeros is
        stored in self.padding so that those zeros can be treated as NAs when
        the column is retrieved, and yet new columns can be calculated by
        multiplying padded and full-length columns together without special
        handling of the padding zeros.

        Padding zeros are replaced with None when a specific row is retrieved
        via the mathDataStore( ).__getitem__( ) method.

        Parameters
        ----------
        key : string
            Name of the column to store.
        value
            The column data.
        padding : int, optional
            If specified, then this method will skip calculating the
            appropriate amount of padding and pad the column by this many
            zeros.  mathDataStore.__setitem__( ) will raise a ValueError if the
            resulting padded column is not the correct length.
        '''
        ## Determine length of padding
        if padding == None:
            padding = self.rows - len( value )
        ## Prepend zero-padding based on type of value
        padded = [0 for i in range( padding )]
        if isinstance( value, (list, UserList, array.array) ):
            padded.extend( value )
        elif isinstance( value, np.ndarray ):
            padded = np.array( padded, dtype=value.dtype )
            padded = np.concatenate( (padded, value) )
        else:
            raise TypeError( 'Cannot add column of type %s.' % type( value ) )
        ## Store padded column and information on the amount of padding
        self.__setitem__( key, padded )
        self.padding[key] = padding

    def __delitem__(self, key ):
        if self.mathDict:
            self.mathDict.local_mask.discard( key )
        if key in self.padding:
            del self.padding[key]
        return( dict.__delitem__( self, key ) )

    def __getitem__(self, key ):
        if isinstance( key, str ):
            return( dict.__getitem__( self, key ) )
        elif isinstance( key, int ) and key < self.rows:
            ret = dict( )
            for column in self.keys( ):
                ret[column] = self[column][key]
            return( ret )
        elif isinstance( key, int ):
            raise IndexError( 'Row index %d is out of bounds for mathDataStore'
                              '( ) with %d rows.' % (key, self.rows) )
        else:
            raise TypeError( 'mathDataStore( ) supports string indexes '
                             'referring to column names, and integer indexes '
                             'referring to row numbers.  It does not support '
                             '%s.' % type( key ) )

    def _toNDArray(self, key, lag='cheat' ):
        ## currently only used in mathDataStore( ).get_column( ).
        if isinstance( self[key][0], int ):
            column_datatype = PR_NP_INT
        else:
            column_datatype = PR_NP_FLT
        ret = np.asarray( self[key], column_datatype )
        return( ret )

    def _toBytes(self, key, lag='cheat' ):
        ## currently only used in mathDict( )._vec( )
        ## and once in mathDictMaker.make( ).
        if isinstance( self[key][0], int ):
            column_datatype = PR_PY_INT
            np_datatype = np.dtype( PR_NP_INT )
        else:
            column_datatype = PR_PY_FLT
            np_datatype = np.dtype( PR_NP_FLT )
        ret = array.array( column_datatype, self[key] ).tobytes( )
        if lag != 'cheat':
            ret = ret[(self.max_lag - lag)*self.itemsize
                      :len( ret ) - lag*self.itemsize]
        return( ret, np_datatype )

    def get_column(self, column_name, lag=0 ):
        # Currently only used in mathDataStore( )'s interaction matrix methods
        # to retrieve an NDArray without knowing if the column is local or not.
        if column_name in self.keys( ):
            ret = self._toNDArray( column_name )
            if lag != 'cheat':
                ret = ret[self.max_lag - lag:len( ret ) - lag]
            return( ret )
        elif self.mathDict != None and \
             column_name in self.mathDict._column_names:
            return( self.mathDict.get_column( column_name,
                                              lag,
                                              vector='row' ) )
        raise mathDictKeyError( '%s is not a valid column name.'
                                % column_name )

    def _bins(self, column_names ):
        '''(for internal use): used exclusively by .interactions_matrix( ) to
        convert dummy term columns with any two arbitrary values into a column
        of integer values 0 and 1.
        '''
        ret = list( )
        for col in column_names:
            bn = self.get_column( col, lag='cheat' )
            values = set( bn )
            if len( values ) != 2:
                raise ValueError( 'Cannot treat %s as a binary column.  It '
                                'does not contain exactly two unique values.' )
            one = max( values )
            bnList = list( )
            for i in bn:
                if i == one:
                    bnList.append( 1 )
                else:
                    bnList.append( 0 )
            bn = np.asarray( bnList, PR_NP_INT )
            ret.append( bn )
        ret = np.stack( ret, 1 )
        return( ret )

    def interactions_matrix(self, binary_columns, interact_with=None ):
        '''(for internal use): calculates the columns representing the
        interactions between one-or-more dummy terms, alone or with one non-
        dummy term.

        This method returns nothing.  The results are stored as columns in the
        data store.  If this mathDataStore( ) is linked to a mathDict( ), then
        it informs the mathDict( ) that the interaction matrix for this set of
        columns has been calculated and stored locally by appending the colon-
        separated list of terms as a single string to
        mathDict( ).local_calculated.

        Parameters
        ----------
        binary_columns : sequence of strings
            Strings identifying the columns to be used as dummy terms.
        interact_with : string
            String identifying the column for the dummy terms to interact with.
        '''
        binary_columns.sort( )                # Obtains binary_columns as
        bins = self._bins( binary_columns )   # columns of integers 0 and 1.
        if interact_with != None:             # Obtains the non-dummy term
            if interact_with in self.keys( ): # whether local or shared.
                intr = self.get_column( interact_with, lag='cheat' )
            elif self.mathDict != None:
                intr = self.mathDict.__getitem__( interact_with,
                                                  lag='cheat',
                                                  vector='row' )
            else:
                mathDictKeyError( '%s is not a valid column string for use in '
                                  'an interactions matrix.' % interact_with )
            intr = intr.reshape( self.rows )
            dt = intr.dtype
        else:                                 # Creates a column of integer 1s
            dt = PR_NP_INT                    # to use in place of the non-
            intr = np.ones( self.rows, dtype=dt ) # dummy term if there isn't a
                                              # non-dummy term.
        r, c = bins.shape
        binInts = np.zeros( r, dtype=PR_NP_INT )  # Turns each row into an int
        for i in range( r ):                  # representing the unique combi-
            s = [str( j ) for j in bins[i,:]] # nation of dummy categories by
            s = str( ''.join( s ) )           # reading the 0 & 1 values as a
            binInts[i] = int( s, 2 )          # single base 2 number.
        c2 = max( binInts )
        col_names = list( )                   # Generate the list of strings
        for i in range( 1, c2 + 1 ):          # identifying the columns in this
            bn = len( binary_columns )        # interactions matrix.
            bn = '{0:0' + str( bn ) + 'b}'
            bn = bn.format( i )
            col_name = list( )
            for j in range( c ):
                col_name.append( '%s=%s' % (binary_columns[j], bn[j]) )
            if interact_with:
                col_name.append( interact_with )
            col_names.append( '(' + ':'.join( col_name ) + ')' )
        #######################################################################
        ## Each column is created as a numpy array of zeros in the list of    #
        ## columns named `ret`.  The first for loop flips only the one column #
        ## representing the combination of dummy categories matching that row #
        ## from 0 to 1.  Then each column numpy array is multiplied by the    #
        ## interact_with column.                                              #
        ret = [np.zeros( r, dtype=dt ) for i in range( c2 )]                 ##
        for i in range( r ):                                                 ##
            for j in range( c2 ):                                            ##
                if binInts[i] == ( j + 1 ):                                  ##
                    ret[j][i] = 1                                            ##
        for i in range( len( ret ) ):          # Always multiplying by intr  ##
            self[col_names[i]] = ret[i] * intr # Keeps the code simple.      ##
        if self.mathDict != None: #................. mathDict( ) needs to know
            calculated = binary_columns[:]         # so that these locally-
            if interact_with != None:              # stored calculated columns
                calculated.append( interact_with ) # can be rebuilt if the
            calculated = ':'.join( calculated )    # mathDict( ) is rebuilt.
            self.mathDict.local_calculated.append( calculated )

    def _inter_col_terms(self, index, masked_index ):
        '''(for internal use): reads a masked, colon-separated list of values
        as terms and determines which one value is not a dummy term.

        Takes the original and masked strings as inputs because it is only used
        in situations where the masked string has already been generated.
        '''
        if self.mathDict != None:
            tSet = self.mathDict.terms
        else:
            tSet = self.terms
        terms = masked_split( index, masked_index, ':' )
        real_term = None
        wSet = set( tSet.W_termRep_set )          # Retrieving and converting
        dSet = set( tSet.dummy_termRep_set )      # these to sets before the
        for t in terms:                           # loop substantially improves
            if t in wSet:                         # performance.
                d = t in dSet
            else:
                if t in self.keys( ):
                    bn = self.get_column( t )
                else:
                    bn = self.mathDict.get_column( t, vector='row' )
                values = set( bn )
                d = len( values ) == 2
            if not d and real_term == None:
                real_term = t
                terms.remove( t )
            elif not d:
                raise ValueError( '`%s` contains more than one non-dummy term.'
                                  '  This is not currently supported.' % item )
        terms.sort( )
        return( terms, real_term )

class mathDictMaker(mathDataStore):
    '''To provide a column for inclusion in the matrix of original columns,
    simply add it as if adding to a dict( ).

    The following describes setting a column via `mathDictMaker[key] = value`.

    Error checking is relatively thorough because this is intended for use
    while setting up a large batch of calculations that might take a long time
    to perform, and mistakes might not otherwise be caught until after-the-
    fact.

    Parameters
    ----------
    key : str
        The key is used as the column name, and must either be a valid Python
        variable name, or be a valid Python variable name immediately followed
        by brackets.  The string enclosed in the brackets is not restricted.
    value : sequence of numerical values
        The sequence represents the column cells.  The data type for the column
        is determined by the first value in the sequence, so if the first value
        is a float then all values will be treated as floats.  If the first
        value is an integer, than all values must be integers.

    Raises
    ------
    TypeError
        If the value is not a sequence, or if the value is a string.
        If the first item in the sequence is an integer but one or more
        subsequent values is not.
        If the first item in the sequence is a float but one or more subsequent
        values is neither an integer nor a float.
        If the first item in the sequence is neither a float nor an integer.
    KeyError
        If the key is not a string.
        If the key is a string but is not a valid Python variable name.
    ValueError
        If the length of the sequence does not match the length of the existing
        column(s).
    '''
    def _crossproducts(self):
        '''(for internal use): Pre-calculates the crossproducts of all columns.

        Returns
        -------
        RA_length : int
            Length of space, in bytes, needed in the shared data array to store
            the pre-calculated crossproducts.
        RA : multiprocessing.sharedctypes.RawArray of bytes
            Sequence of bytes consisting of the pre-calculated crossproduct
            columns to be stored in the shared data array.
        cp_dtypes : list of numpy dtype strings
            Data types of the pre-calculated crossproduct columns.
        '''
        cp_count = math.factorial( len( self ) ) \
                   // 2 \
                   // math.factorial( len( self ) - 2 )
        indices = [i for i in range( len( self ) )]
        indices = [i for i in itertools.combinations( indices, 2 )]
        RA_length = self.rows*cp_count*self.itemsize
        RA = sharedctypes.RawArray( 'b', RA_length )
        cp_dtypes = list( )
        for i in range( cp_count ):
            a = self.rows*i*self.itemsize
            t = self.rows*(i+1)*self.itemsize
            if isinstance( self[self.key_list[indices[i][0]]][0], float ) or \
               isinstance( self[self.key_list[indices[i][1]]][0], float ):
                dt = np.dtype( PR_NP_FLT )
            else:
                dt = np.dtype( PR_NP_INT )
            arr = np.asarray( self[self.key_list[indices[i][0]]], dtype=dt ) \
                * np.asarray( self[self.key_list[indices[i][1]]], dtype=dt )
            RA[a:t] = arr.tobytes( )
            cp_dtypes.append( dt )
        return( RA_length, RA, cp_dtypes )

    def _powers(self, powers ):
        '''(for internal use): Pre-calculates powers two through `powers` of
        all columns.

        Returns
        -------
        RA_length : int
            Length of space, in bytes, needed in the shared data array to store
            the pre-calculated column powers.
        RA : multiprocessing.sharedctypes.RawArray of bytes
            Sequence of bytes consisting of the pre-calculated column powers to
            be stored in the shared data array.
        pwr_dtypes : list of numpy dtype strings
            Data types of the columns of pre-calculated columns.
        '''
        RA_length = self.rows*len( self )*( powers - 1 )*self.itemsize
        RA = sharedctypes.RawArray( 'b', RA_length )
        pwr_dtypes = list( )
        for pwr in range( powers - 1 ):
            for i in range( len( self ) ):
                a = self.rows*pwr*len( self.key_list )*self.itemsize
                a += self.rows*i*self.itemsize
                t = a + self.rows*1*self.itemsize
                if isinstance( self[self.key_list[i]][0], float ):
                    dt = np.dtype( PR_NP_FLT )
                else:
                    dt = np.dtype( PR_NP_INT )
                arr = np.asarray( self[self.key_list[i]], dtype=dt ) \
                      ** (pwr + 2)
                RA[a:t] = arr.tobytes( )
                pwr_dtypes.append( dt )
        return( RA_length, RA, pwr_dtypes )

    def make(self, cache_crossproducts=False, cache_powers=1 ):
        '''Assembles the shared data array and mathDict( ) matrix
        representation.

        Parameters
        ----------
        cache_crossproducts : boolean, optional
            If True, then the crossproducts of all combinations of columns
            (without replacement) will be pre-calculated and stored along with
            the matrix of original columns.  To pre-calculate the product of a
            column and itself, set cache_powers to a number greater than or
            equal to two.
        cache_powers : int, optional
            If an integer greater than one, then powers of all columns from two
            to this number will be pre-calculated and stored with the matrix of
            original columns.  Numbers less than or equal to one will be
            ignored.

        Returns
        -------
        RA : multiprocessing.sharedctypes.RawArray
            The shared data array in which the matrix of original columns and
            any pre-calculated columns will be stored.
        MD : mathDict( )
            The mathDict( ) representation of a matrix initially consisting of
            the same columns as the matrix of original columns.  The
            mathDict( ) object can then be used to mask some columns and/or
            append calculated columns to the matrix represented by the
            mathDict( ).  Local columns can then be appended to copies of the
            mathDict( ) object in other processes.
        '''
        RA_length = self.rows*len( self )*self.itemsize
        if cache_crossproducts:
            cp_ra_len, CP, cp_dtypes = self._crossproducts( )
            RA_length += cp_ra_len
        else:
            cp_ra_len = 0
        if cache_powers > 1:
            pwr_ra_len, PWR, pwr_dtypes = self._powers( cache_powers )
            RA_length += pwr_ra_len
        else:
            pwr_ra_len = 0
        RA = sharedctypes.RawArray( 'b', RA_length )
        dtypes = list( )
        for i in range( len( self ) ):
            a = self.rows*i*self.itemsize
            t = self.rows*(i+1)*self.itemsize
            RA[a:t], dt_np = self._toBytes( self.key_list[i] )
            dtypes.append( dt_np )
        if cache_crossproducts:
            a = self.rows*len( self )*self.itemsize
            t = a + cp_ra_len
            RA[a:t] = CP[:]
            dtypes.extend( cp_dtypes )
        if cache_powers > 1:
            a = self.rows*len( self )*self.itemsize + cp_ra_len
            t = a + pwr_ra_len
            RA[a:t] = PWR[:]
            dtypes.extend( pwr_dtypes )
        MD = mathDict( RA, self.rows*len( self ), self.key_list,
                       dtypes=dtypes,
                       cache_crossproducts=cache_crossproducts,
                       cache_powers=cache_powers,
                       padding=self.padding_list )
        if 'terms' in self.__dir__( ):
            MD.terms = self.terms
        return( RA, MD )

    @staticmethod
    def fromMatrix( matrix, integer=False ):
        '''Creates a SharedDataArray and mathDict( ) representation from an
        existing two-dimensional matrix.

        Parameters
        ----------
        matrix : 2-dimensional array
            An existing Numpy matrix or 2-dimensional Numpy array.
        integer : bool
            If True, the matrix in the SharedDataArray will consist of
            integers.

        Returns
        -------
        SharedDataArray : multiprocessing.sharedctypes.RawArray
            The shared data array in which the matrix will be stored.
        mathDict( ) : mathDict( )
            The mathDict( ) representation of a matrix.
        '''
        if integer:
            np_datatype = PR_NP_INT
        else:
            np_datatype = PR_NP_FLT
        r, c = matrix.shape
        items = r * c
        RA_length = items * mathDictMaker( ).itemsize
        column_names = ['x%d' % i for i in range( c )]
        SharedDataArray = sharedctypes.RawArray( 'b', RA_length )
        SharedDataArray[:] = np.asarray( matrix, dtype=np_datatype
                                        ).tobytes( order='F' )
        return( SharedDataArray, mathDict( SharedDataArray=SharedDataArray,
                                           items=items,
                                           column_names=column_names,
                                           mask=[True],
                                           dtypes=np_datatype) )

class mathDictHypothesis(object):
    '''Generates testable hypotheses about a mathDict( ) matrix in the form of
    linear constraints (for use in regression analysis).

    An 'X' matrix representing the regressors (RHS, or 'independent' variables)
    for a linear model for testing the hypothesis is generated.  The columns in
    the matrix will be the union of all columns in the matrix represented by
    the mathDict( ) object and all columns in the hypothesis, including
    calculated columns.

    An 'R' matrix with the same number of columns as the 'X' matrix and one row
    for each column in the hypothesis, as well as an 'r' column vector/vertical
    array with one row/cell for each column in the hypothesis will also be
    generated.  These can then be used for either an F or a Wald test.

    '''
    def __init__(self, mathDict ):
        self.mathDict = mathDict
        self.hypothesis = OrderedDict( )

    def add(self, column, hypothesis=0 ):
        '''Adds a column to the hypothesis, and to the resulting X matrix if
        not included in the matrix represented by the mathDict( ).

        Raises
        ------
        RankError
            If an attempt to square a dummy variable is made.
        '''
        def checkRank( termRep=None, column=None, all_term_keys=None ):
            '''Determines that the resulting X matrix will not be of sufficient
            rank if adding termRep to the hypothesis would result in duplicate
            column values due to the over-use of a dummy term.

            Parameters
            ----------
            termRep : string
                The representation of an individual column from the matrix of
                original columns used in the column being considered for the
                hypothesis.
            column : string
                The column for which an attempt is being made to add it to the
                hypothesis.  This can be a column that mathDict can calculate
                from one or more columns in the matrix of original columns.
            all_term_keys : Sequence of strings
                If this is specified, then each string in this sequence is
                checked to see if it is a dummy term.  The function returns
                true only if all strings in the sequence are dummy terms, and
                false otherwise.

            Raises
            ------
            RankError
                If all_term_keys is not specified and `termRep` is both a dummy
                term and already used in a column in the matrix.  (If only one
                column attempts to use the term, then it makes no difference
                whether or not the term is squared.)

            Note
            ----
            Either `termRep` and `column` should be specified, or
            `all_term_keys`, but not all three.  If the latter is specified,
            then the first two will be ignored.
            '''
            if self.mathDict.terms:
                if all_term_keys:
                    term_keys = all_term_keys
                else:
                    term_keys = _vars_in_factor( termRep )
                for key in term_keys:
                    if self.mathDict.terms.is_a( 'dummy',
                                                 key=key,
                                                 value=termRep ):
                        if all_term_keys:
                            continue
                        for c in self.mathDict.columns:
                            if has_term( c, key ):
                                raise RankError( 'Cannot square dummy '
                                      'variables in hypotheses.  The column '
                                      '`%s` contains a squared dummy variable.'
                                      % column )
                    elif all_term_keys:
                        return( False )
                if all_term_keys:
                    return( True )
        #######################################################################
        ## Body of method starts here:
        if column in self.mathDict.columns:
            self.hypothesis[column] = hypothesis
            return( )
        mc = mask_brackets( column )              # Only calculated columns are
        mobj = cross_Aor_power.fullmatch( mc )    # evaluated for rejection.
        if mobj != None:
            mobjDict = masked_dict( column, mobj )
            if mobjDict['column_a'] == mobjDict['column_b']: # If the string is
                if mobjDict['power_a'].isdigit( ):           # 1 column multi-
                    powers = int( mobjDict['power_a'] )      # plied by itself,
                else:                             # then it's rewritten as a
                    powers = 1                    # single instance of the
                if mobjDict['power_b'].isdigit( ):# column, raised to a power
                    powers += int( mobjDict['power_b'] )
                else:
                    powers += 1
                col_name = '%s**%d' % (mobjDict['column_a'], powers)
                col_name, boolchk = _soft_in(     # The rewritten column is
                    col_name,                     # checked against existing
                    self.mathDict.columns )       # columns in the matrix
                if not boolchk:                   # represented by the
                    checkRank(                    # mathDict( ) so that dupli-
                        mobjDict['column_a'],     # cates aren't added.
                        column )                  # checkRank( ) is performed
                self.hypothesis[col_name] = hypothesis # only if this would be
                return( )                         # an additional column.
            else:
                fznCl   = frozenset(              # If column_a and column_b
                    _vars_in_factor(              # both consist of the same
                        mobjDict['column_a'] ) )  # terms, regardless of order,
                fznCl_b = frozenset(              # then checkRank is used to
                    _vars_in_factor(              # check if all of the terms
                        mobjDict['column_b'] ) )  # are dummy terms, and if so,
                if fznCl == fznCl_b:              # if there already exists a
                    if checkRank( all_term_keys=fznCl ): # column with the same
                        for col in self.mathDict.columns:# set of terms.
                            if fznCl == frozenset( _vars_in_factor( col ) ):
                                if mobjDict['power_a'].isdigit( ):
                                    pa = '**' + mobjDict['power_a']
                                else:
                                    pa = ''
                                if mobjDict['power_b'].isdigit( ):
                                    pb = '**' + mobjDict['power_b']
                                else:
                                    pb = ''
                                raise RankError( '`%s%s` and `%s%s` consist of'
                                      ' the same set of dummy terms.  The '
                                      'mathDict( ) matrix already has `%s`, '
                                      'which also consists of the same dummy '
                                      'terms.'
                                      % (mobjDict['column_a'], pa,
                                         mobjDict['column_b'], pb,
                                         col) )
            for l in ['a', 'b']:
                if mobjDict['power_'+l].isdigit( ) and \
                   int( mobjDict['power_'+l] ) >= 2:
                    checkRank( mobjDict['column_'+l], column )
        self.hypothesis[column] = hypothesis

    def make(self):
        '''Returns a tuple consisting of the X matrix, R matrix, and r column
        vector/vertical array, each in the form of a two-dimensional numpy
        array.
        '''
        X, R, r = self._make( )
        return( X[:], R, r )

    def _get_X(self):
        X, R, r = self._make( )
        return( X )

    def _make(self):
        superset_only = list( )
        rt_orig = list( )
        rt_superset_only = list( )
        for key, val in self.hypothesis.items( ):
            if key in self.mathDict.columns:
                rt_orig.append( (self.mathDict.columns.index( key ), val) )
            else:
                rt_superset_only.append( (len( superset_only ), val) )
                if key in self.mathDict._column_names:    #
                    ###########################################################
                    ## `**1` so that it extends the matrix to the right with  #
                    ##  the calculated columns, instead of getting inserted   #
                    superset_only.append( '%s**1' % key ) ## in the middle.   #
                else: #########################################################
                    superset_only.append( key )
        # Build X
        X = mathDict( self.mathDict.buffer,
                      self.mathDict.items,
                      self.mathDict._column_names,
                      self.mathDict._mask,
                      self.mathDict.dtypes,
                      self.mathDict._calculated_columns.copy( ),
                      self.mathDict.cache_crossproducts,
                      self.mathDict.cache_powers,
                      padding=self.mathDict._orig_padding )
        X._local_mask = self.mathDict._local_mask
        X.local = self.mathDict.local
        X.local.mathDict = X
        X.calculated_columns( 'extend', superset_only )
        # Build R, r
        Z = np.zeros( (1, len(self.mathDict.columns) + len( superset_only )),
                       dtype=np.dtype( PR_NP_INT ) )
        R = np.zeros_like( Z )
        r = np.zeros( (len( rt_orig ) + len( rt_superset_only )),
                      dtype=np.dtype( PR_NP_INT ) )
        for j in range( len( rt_orig ) ):
            RX = np.zeros_like( Z )
            i, v = rt_orig[j]
            RX[0,i] = 1
            R = np.vstack( (R, RX) )
            r[j] = v
        for j in range( len( rt_superset_only ) ):
            RX = np.zeros_like( Z )
            i, v = rt_superset_only[j]
            RX[0,self.mathDict.shape[1] + i] = 1
            R = np.vstack( (R, RX) )
            r[len( rt_orig ) + j] = v
        return( X, R[1:,:], r )

class mathDictConfig(dict):
    def rebuild(self, SharedDataArray, terms=None ):
        '''Recreates the mathDict( ) object whose configuration is stored in
        this dict( ).  Requires that the shared data array is provided as a
        parameter.
        '''
        if terms == None and 'terms' in self:
            terms = self['terms']
        MD = mathDict( SharedDataArray=SharedDataArray,
                       items=self['items'],
                       column_names=self['_column_names'],
                       mask=deepcopy( self['mask'] ),
                       dtypes=self['dtypes'],
                       calculated_columns=deepcopy(
                                                  self['calculated_columns'] ),
                       cache_crossproducts=self['cache_crossproducts'],
                       cache_powers=self['cache_powers'],
                       padding=self['_orig_padding'],
                       terms=terms )
        MD.local_calculated = self['local_calculated']
        for calculated in self['local_calculated']:
            MD.add( calculated )
        MD._local_mask = self['local_mask'].intersection( set( MD.local.keys( )
                                                                ) )
        self._last_columns = None
        return( MD )

class mathDict(object):
    def __init__(self, SharedDataArray, items, column_names,
                 mask=None,
                 dtypes=None,
                 calculated_columns=None,
                 cache_crossproducts=False,
                 cache_powers=1,
                 terms=None,
                 padding=None):
        '''dict( )-like interface to a shared-memory, two-dimensional array of
        heterogenous numeric data-typed columns that builds linear constraints/
        hypotheses and pre-calculates powers and cross-products of columns.

        Parameters
        ----------
        SharedDataArray : multiprocessing.sharedctypes.RawArray or comparable
            The shared-memory byte array in which the columns and pre-
            calculated manipulations thereof are stored.  See the note
            regarding array length.
        items : int
            The number of cells in the matrix of original columns, not
            including the intercept column or calculated columns.  All columns
            must be supplied to mathDict( ) with the same number of rows.
        column_names : sequence of strings
            The names of the original columns, not including the intercept
            column or calculated columns.  There must be exactly one string in
            column_name for each column in the matrix of original columns.
        mask : sequence of booleans, optional
            Columns with an associated mask boolean of True are hidden.  The
            first boolean in the sequence is associated with the intercept
            column, resulting in a column of ones at index 0 if set to False.
            After that, the value at mask[1] corresponds to column_name[0], and
            so on.  If the mask sequence is shorter than the sequence of
            column_names, than columns without an associated mask value are
            unmasked.
        dtypes : string or Sequence, optional
            If a sequence, there must be one value for each column,
            representing the numpy data type of the cells in that column.  If a
            single string, then all columns must consist of cells of that data
            type.  Currently only 8-byte-per-item data types ('i8', 'f8') are
            suported.
        calculated_columns : sequence of strings
            Each string represents a column that extends the matrix represented
            by the mathDict( ) beyond the matrix of original columns with a
            column calculated therefrom using operations supported by
            mathDict( ).  Currently limited to crossproducts, powers, and lags.
        cache_crossproducts : boolean, optional
            If True, then the crossproducts of each combination (without
            replacement) of columns in the matrix of original columns has been
            pre-calculated and appended to the shared data array after the
            original columns.  Use mathDictMaker( ) to pre-calculate these
            values.
        cache_powers : int, optional
            If set to an integer greater than 1, then the powers of each
            original column ranging from 2 through this value (inclusive) have
            been pre-calculated and appended to the shared data array after the
            original columns and cached crossproducts (if present).  Use
            mathDictMaker( ) to pre-calculate these values.
        terms : termSet( ), optional
            The termSet( ) to be used by mathDict.hypothesis.add( ) in
            determining whether or not adding a column string to the hypothesis
            should result in a RankError.
        padding : list of ints, optional
            A list in which each entry corresponds to the column at the same
            index in column_names, indicating the number of rows of padding at
            the beginning of column.  This is the list produced by
            mathDictMaker.padding_list.  If provided, this must be the same
            length as column_names, with zeros for each column that has no
            padding.

        Notes
        -----
        Size of `SharedDataArray`: The shared-memory array identified by the
        shared data array parameter stores each column in the matrix of
        original columns, in column-major order, followed by each pre-
        calculated crossproduct column (if present) and pre-calculated power
        column (if present).  With 8-byte-per-item data types, the matrix of
        original columns alone requires 8*[row count]*[column count].
        '''
        self.buffer = SharedDataArray
        self.items = items
        self._column_names = column_names
        self.cache_crossproducts = cache_crossproducts
        self.cache_powers = cache_powers
        self.itemsize = PR_BYTESIZE
        self.hypothesis = mathDictHypothesis( self )
        #######################################################################
        ## Setting self.local to dict( ) before setting it to mathDataStore( )#
        ## avoids an error that would otherwise result from a chicken-or-the- #
        self.local = dict( ) ## egg problem.                                  #
        self.local = mathDataStore( mathDict=self )                           #
        #######################################################################
        self._local_mask = setList( )
        self.local_calculated = list( )
        self.terms = terms
        self.strings_checked = setList( )
        if calculated_columns == None:            # ._calculated_columns,
            calculated_columns = []               # ._mask, and ._local_mask
        if mask == None:                          # should never be accessed
            mask = [False]                        # directly.  Use the corre-
        self._mask = mask                         # sponding methods.
        self._calculated_columns = calculated_columns
        if dtypes == None:
            dtypes = PR_NP_FLT
        if isinstance( dtypes, str ):
            dtypes = [dtypes for x in column_names]
        self.dtypes = dtypes
        if padding == None:
            padding = [0 for i in range( len( column_names ) )]
        elif len( padding ) != len( column_names ):
            raise ValueError( 'Length of padding must equal length of '
                              'column_names.' )
        self._orig_padding = padding
        self._last_columns = None
        self._last_max_lag = None

    def mask(self, action=None, *args, **kwargs ):
        '''Protected object: _mask2.
        For read access, calling .method( ) with no parameters returns the
        object, which can then be read/subscripted as usual.

        To call a method of the protected object, the first parameter should be
        the name of the method to be called.  All further parameters are passed
        on to the method.  To replace the protected object, `'replace`' should
        be the first parameter, followed by the new object.
        '''
        protected_name = '_mask'
        protected = getattr( self, protected_name )
        if action == 'replace':
            return( setattr( self, protected_name, *args ) )
        if action == None:
            return( protected )
        self._recal_max_lag( )
        return( getattr( protected, action )( *args, **kwargs ) )

    def calculated_columns(self, action=None, *args, **kwargs ):
        '''Protected object: _calculated_columns.
        For read access, calling .method( ) with no parameters returns the
        object, which can then be read/subscripted as usual.

        To call a method of the protected object, the first parameter should be
        the name of the method to be called.  All further parameters are passed
        on to the method.  To replace the protected object, `'replace`' should
        be the first parameter, followed by the new object.
        '''
        protected_name = '_calculated_columns'
        protected = getattr( self, protected_name )
        if action == 'replace':
            return( setattr( self, protected_name, *args ) )
        if action == None:
            return( protected )
        self._recal_max_lag( )
        return( getattr( protected, action )( *args, **kwargs ) )

    def local_mask(self, action=None, *args, **kwargs ):
        '''Protected object: _local_mask.
        For read access, calling .method( ) with no parameters returns the
        object, which can then be read/subscripted as usual.

        To call a method of the protected object, the first parameter should be
        the name of the method to be called.  All further parameters are passed
        on to the method.  To replace the protected object, `'replace`' should
        be the first parameter, followed by the new object.
        '''
        protected_name = '_local_mask'
        protected = getattr( self, protected_name )
        if action == 'replace':
            return( setattr( self, protected_name, *args ) )
        if action == None:
            return( protected )
        self._recal_max_lag( )
        return( getattr( protected, action )( *args, **kwargs ) )

    def _recal_max_lag(self, obj=None, attr=None ):
        '''(for internal use): clears the cached .max_lag and .columns values
        so that next time either one is accessed, it is recalculated/
        regenerated.

        Originally implemented because it was necessary for .max_lag to perform
        well, and then extended to cover .columns as well.
        self.local.__setitem__( ) also clears self._last_columns, but not
        self._last_max_lag.
        '''
        self._last_max_lag = None
        self._last_columns = None

    @property
    def columns(self):
        '''list of strings: Lists the name of each column in the matrix
        represented by the mathDict( ), starting with the Intercept column
        unless masked, followed by columns in the matrix of original columns
        that are not masked, by local columns, and finally by calculated
        columns.
        '''
        if self._last_columns != None:
            return( self._last_columns )
        ret = list( )
        if self.mask( )[0] == False:
            ret.append( 'Intercept' )
        len_mask = len( self.mask( ) )
        for i in range( len( self._column_names ) ):
            if (i+1 < len_mask and self.mask( )[i+1] == False) or \
               i+1 >= len_mask:
                ret.append( self._column_names[i] )
        for key in self.local.key_list:
            if key not in self.local_mask( ):
                ret.append( key )
        ret.extend( self.calculated_columns( ) )
        if len( ret ) > 0:
            self._last_columns = ret
        return( ret )

    @property
    def rows(self):
        '''The number of rows in the matrix of original columns.'''
        return( self.items // len( self._column_names ) )

    @property
    def max_lag(self):
        '''int: Returns the maximum lag currently used in any column in the
        matrix represented in the mathDict( ) object.

        This value is recalculated each time there is a change in one of the
        columns in the matrix, based on the column identifier strings.
        '''
        if self._last_max_lag != None:
            return( self._last_max_lag )
        def column_lag( column ):
            if column in self._column_names:
                return( self._orig_padding[self._column_names.index(
                                                                    column )] )
            return( (self.local.padding[column] if column in \
                     self.local.padding.keys( ) else 0) if column in \
                     self.local.keys( ) else None )
        max_lag = 0
        for column in self.columns:
            columns_lag = column_lag( column )
            if columns_lag != None:
                max_lag = max( max_lag, columns_lag )
            elif column == 'Intercept':
                pass
            else:
                action, mobj_dict = self._interpret_str( column )
                for kp in {'', '_a', '_b'}:
                    if 'lag' + kp in mobj_dict.keys( ):
                        lag = mobj_dict['lag'+kp]
                        if lag in NoneSet:
                            lag = 0
                        else:
                            lag = int( lag )
                        if kp == '':
                            lag += column_lag( mobj_dict['column_name'] )
                        else:
                            lag += column_lag( mobj_dict['column'+kp] )
                        max_lag = max( max_lag, lag )
        self._last_max_lag = max_lag
        return( max_lag )

    @property
    def shape(self):
        '''tuple(effective row count, column count): The effective row count
        will be listed even if all columns are masked, resulting in an (n, 0)
        tuple.

        Effective row count: The number of rows in the matrix of original
        columns, less the maximum lag that the mathDict( ) has been configured
        to support.
        '''
        r  = self.rows - self.max_lag
        c  = len( self._column_names ) + 1
        c -= sum( self.mask( ) )
        c += len( self.calculated_columns( ) )
        c += len( self.local )
        c -= len( self.local_mask( ) )
        return( r, c )

    @staticmethod
    def _interpret_str( index, vector='column', lag=None ):
        '''(for internal use): contains all logic for processing column
        identifier strings that is not dependent on the particulars of the
        specific mathDict( ) instance.

        Parameters
        ----------
        index : string
            Any string that could be passed into mathDict[`index`]
            (.__getitem__( index )) to identify result in the form of a single
            column.
        vector : {'column', 'row'}, optional
            The orientation in which the result is to be returned.
        lag : int, optional
            Lag should only be specified as a separate parameter when it is not
            specified within the column identifier string.

        Returns
        -------
        action : string
            The action to be taken to retrieve or calculate the column (or in
            the case of an interactions matrix, the matrix) to be returned.
            This will be the name of a particular mathDict( ) method.
        dict
            The keyword arguments to be passed into the method identified in
            `action`.
        '''
        action = None
        masked_index = mask_brackets( index )
        mobj = crosspattern.fullmatch( masked_index )
        if mobj != None:
            mobj_dict = masked_dict( index, mobj )
            action = 'crossproduct'
        if not action:
            mobj = crosspower.fullmatch( masked_index )
            if mobj != None:
                mobj_dict = masked_dict( index, mobj )
                action = 'crosspower'
        if not action:
            mobj = powerpattern.fullmatch( masked_index )
            if mobj != None:
                mobj_dict = masked_dict( index, mobj )
                if mobj_dict['lag'] not in {None, ''}:
                    if lag:
                        raise Exception( '_handle_str received lag in multiple'
                                         ' forms.' )
                    mobj_dict['lag'] = int( mobj_dict['lag'] )
                elif lag:
                    mobj_dict['lag'] = lag
                else:
                    mobj_dict['lag'] = 0
                if mobj_dict['power'] in {None, ''}:
                    action = 'get_column'
                    del mobj_dict['power']
                else:
                    action = 'power'
        elif lag:
            if mobj_dict['lag_a'] not in NoneSet or \
               mobj_dict['lag_b'] not in NoneSet:
                raise Exception(
                                '_handle_str received lag in multiple forms.' )
            mobj_dict['lag_a'] = lag
            mobj_dict['lag_b'] = lag
        if action:
            mobj_dict['vector'] = vector
        else:
            if masked_index.count( ':' ) > 0:
                mobj_dict = {'index': index, 'masked_index': masked_index}
                action = '_interaction_columns'
        if not action:
            raise mathDictKeyError(  '%s is not a valid key.' % index  )
        return( action, mobj_dict )

    def _handle_str(self, index,
                          vector='column',
                          lag=None,
                          expanded_columns=False ):
        '''(for internal use): contains logic for processing column identifier
        strings that is dependent on the particulars of the specific
        mathDict( ) instance, calling ._interpret_str( ) for non-instance-
        dependent logic.

        Parameters
        ----------
        index : string
            Any string that could be passed into mathDict[`index`]
            (.__getitem__( index )) to identify result in the form of a single
            column.
        vector : {'column', 'row'}, optional
            The orientation in which the result is to be returned.
        lag : int, optional
            Lag should only be specified as a separate parameter when it is not
            specified within the column identifier string.
        expanded_columns : bool
            If true, the list of column identifier strings identifying
            individual columns in an interactions matrix is returned after the
            first two return values when and only when the action identified in
            `index` is an interactions matrix.

        Returns
        -------
        action : string
            The action to be taken to retrieve or calculate the column (or in
            the case of an interactions matrix, the matrix) to be returned.
            This will be the name of a particular mathDict( ) method.
        dict
            The keyword arguments to be passed into the method identified in
            `action`.
        columns : list of strings
            The column identifier strings for the individual columns in the
            interactions matrix, if applicable.
        '''
        action, mobj_dict = self._interpret_str( index,
                                                 vector=vector,
                                                 lag=lag )
        if action == '_interaction_columns':
            columns = self._interaction_columns( **mobj_dict )
            action = '_mat'
            mobj_dict = {'column_indexes': columns}
            if expanded_columns:
                return( action, mobj_dict, columns )
        return( action, mobj_dict )

    def __getitem__(self, index, vector='column', lag=None ):
        '''Returns the matrix represented by the mathDict( ) or a portion
        thereof.

        **Supported Notation**

        ------------------
        str -> returns one vertical array
            A string can be the name of one column or the formula for one
            column that can be calculated from the original columns.  No check
            is performed to ensure that referenced columns are unmasked.
        int -> returns one vertical array
            A single integer will return the corresponding column in the matrix
            represented by the mathDict( ) object.
        int(m):int(n) slice -> returns a matrix with `n` - `m` columns.
            A slice with start and or stop specified will return the
            corresponding column in the matrix represented by the mathDict( ).
            A slice with neither specified will return the entire matrix
            represented by the mathDict( ).  This includes local columns and
            calculated_columns listed in the calculated_columns attribute, but
            does not include masked columns.

        Examples
        --------
        str : 'a'
            Returns column 'a'.
        str : 'a * b'
            Returns a column in which each row i is 'a'[i] * 'b'[i].
        str : 'L2@a'
            Returns the second lag of column 'a'.  (Case sensitive.)
        int : 0
            If the first value in the mask sequence is False, this will return
            a column of ones.  The data type of cells in the Intercept column
            will match the first column in the matrix of original columns.
        slice : [:]
            This will return the Intercept column of ones unless masked,
            unmasked columns in the matrix of original columns, unmasked local
            columns, and finally calculated columns.
        '''
        INTERNAL_NOTES  = '''
        Parameters
        ----------
        index : int, slice, or str
            See above.
        vector : {'column', 'row'}, optional
            Passed on to column retrieval methods when used by ._mat( ) to get
            calculated columns.

        Notes
        -----
        All logic translating column numbers in the matrix represented by the
        mathDict( ) to index numbers/keys of stored columns is contained here.
        '''
        if isinstance( index, str ):
            indexTpl = self._handle_str( index,
                                      vector=vector,
                                      lag=lag )
            action = indexTpl[0]
            mobj_dict = indexTpl[1]
            for key in {'column_name', 'column_a', 'column_b'}:
                if key in mobj_dict.keys( ):
                    if mobj_dict[key] not in self._column_names and \
                       mobj_dict[key] not in self.local.key_list:
                        raise mathDictKeyError( '%s is neither a column in the'
                              ' shared data matrix nor a column in the local '
                              'data store.' % key )
            return( getattr( self, action )( **mobj_dict ) )
        elif isinstance( index, int ):
            if index == 0 and self.mask( )[0] == False:
                return( self.power( self._column_names[0],
                                    0,
                                    vector=vector,
                                    lag=lag ) )
            index = slice( index, index+1 )
        if isinstance( index, slice ):
            if index.step != None:
                raise KeyError( 'Steps are not supported' )
            else:
                xes = list( ) #.................... xes = list of all colums to
                                                  # return as part of the slice
                ind = 0 #.......................... ind = index of the column
                                                  # in the matrix represented
                                                  # by the mathDict( ) object
                                                  # that is currently being
                                                  # identified
                ## val_if_present( ) replaces None with the default, whereas
                ## getattr( ) does not.
                start = val_if_present( index, 'start', 0 )
                stop = val_if_present( index, 'stop', self.shape[1] )
                if stop > self.shape[1]:
                    raise IndexError( 'Outer-bound of slice, %d, exceeds the '
                                      'number of columns, %d.'
                                      % (stop, self.shape[1]) )
                ###############################################################
                ## Except here, ind is incremented when a column is determined#
                ## to be part of the matrix.  ind = n after original matrix   #
                ## columns 0, 1, ..., n-1 if all are unmasked.  If there are  #
                ## n+1 columns including the Intercept, then ind = n is the   #
                ## correct stopping point, and the Intercept column still has #
                ## to get included.                                           #
                if self.mask( )[0] == False:                                 ##
                    stop -= 1                                                ##
                    if start == 0:                                           ##
                        xes.append( -1 )                                     ##
                    else:                                                    ##
                        start -= 1                                           ##
                ###############################################################
                for i in range( len( self._column_names ) ):
                    # i = index of the original matrix column currently being
                    # considered
                    if i+1 >= len( self.mask( ) ) or \
                       self.mask( )[i+1] == False:
                        if ind >= start and ind < stop:
                            xes.append( i ) # Adding original column index int
                        ind += 1
                        if ind == stop:
                            break
                for i in range( len( self.local.keys( ) ) ):
                    # i = index of the local column currently being considered
                    if ind == stop:
                        break
                    key = self.local.key_list[i]
                    if key not in self.local_mask( ):
                        if ind >= start and ind < stop:
                            xes.append( key ) # Adding local column key string
                        ind += 1
                for i in range( len( self.calculated_columns( ) ) ):
                    # i = index of the calculated column currently being
                    # considered
                    if ind == stop:
                        break
                    if ind >= start and ind < stop:
                        xes.append( self.calculated_columns( )[i] )
                               # Adding calculated column string
                    ind += 1
                if not lag:
                    lag = 0
                return( self._mat( xes, lag=lag ) )

    def _interaction_columns( self, index, masked_index ):
        '''(for internal use): takes a column identifier string identifying an
        interactions matrix in its original and masked form, and returns the
        list of column identifier strings for the individual columns that are
        actually stored in the mathDict( ), whether as local or shared columns.

        If no such columns exist, then .local.interactions_matrix( ) is called
        in order to calculate them.
        '''
        terms, real_term = self.local._inter_col_terms( index, masked_index )
        column_name_pattern = list( )
        for t in terms:
            column_name_pattern.append( re.escape( t ) + '=[0-1]' )
        column_name_pattern = '\(' + ':'.join( column_name_pattern )
        if real_term != None:
            column_name_pattern += ':' + re.escape( real_term )
        column_name_pattern += '\)'
        repatt = re.compile( column_name_pattern )
        ret = list( )
        for column in self._column_names:
            if repatt.fullmatch( column ) != None:
                ret.append( column )
        for column in self.local.key_list:
            if repatt.fullmatch( column ) != None:
                ret.append( column )
        if len( ret ) > 0:
            return( ret )
        else:
            self.local.interactions_matrix( terms, real_term )
            ret = self._interaction_columns( index, masked_index )
            for column in ret:
                self.set_mask( column )
            return( ret )

    def __setitem__(self, key, value ):
        if len( value ) == self.rows:
            self.local[key] = value
        else:
            self.local.savePaddedColumn( key, value )

    @property
    def _ofs_start(self):
        numcols = len( self._column_names )
        cp_len  = math.factorial( numcols ) \
                  // 2 \
                  // math.factorial( numcols - 2 )
        pwr_len = numcols * (self.cache_powers - 1)
        cp_start, pwr_start = numcols, numcols
        if self.cache_crossproducts:
            pwr_start += cp_len
        return( cp_start, pwr_start )

    def _ofs(self, internal_column_index, lag=0 ):
        '''(for internal use).

        Parameters
        ----------
        interior_column_index : int
            the index number of the column in the shared data array, whether an
            original column or cached, calculated column.
        lag : int, optional
            This many rows towards the top of the matrix will be unhidden,
            without changing the total number of rows returned.

        Returns
        -------
        the offset in the shared data array at which the column with the
        specified index number begins, adjusted to hide rows as necessary to
        reserve sufficient data for lags.  Error checking for appropriate lag
        values occurs in .vec( ) in order to have the same set of code apply to
        lagged local columns as well as lagged shared columns.
        '''
        ofs = self.rows * internal_column_index * self.itemsize
        if not lag == 'cheat':
            ofs += (self.max_lag - lag) * self.itemsize
        return( ofs )

    def _vec(self, internal_column, lag=0 ):
        '''(for internal use): returns the column of the shared data array at
        the requested index, in row vector/horizontal array form.

        Parameters
        ----------
        internal_column : int or str
            Either an integer index for a column in the shared data array, or a
            string identifying either a column in the matrix of original
            columns or a local column by name.  Strings are accepted in order
            to allow calculated column retrieval methods to retrieve base
            columns via .get_column( ) without distinguishing between types.
        lag : int, optional
            (See ._ofs( ) docstring.)

        Special Values
        --------------
        - 1
            Returns `self[0]` for use representing the Intercept column in a
            list of column indexes.

        Raises
        ------
        ValueError
            If the requested lag exceeds self.max_lag.  The .max_lag attribute
            must be set prior to adding and/or retrieving columns involving
            lagged values.
        '''
        if isinstance( internal_column, str ):
            if internal_column in self._column_names:
                internal_column = self._column_names.index( internal_column )
            elif internal_column in self.local.keys( ):
                arr, dt = self.local._toBytes( internal_column, lag=lag )
                ret = np.fromstring( arr, dt )
        if internal_column == -1:
            return( self.power( self._column_names[0], 0, vector='row' ) )
        elif isinstance( internal_column, int ):
            ofs = self._ofs( internal_column, lag=lag )
            dt = self.dtypes[internal_column]
            if not lag == 'cheat':
                count = self.shape[0]
            else:
                count = self.rows
            ret = np.frombuffer( self.buffer,
                                 dtype=dt,
                                 count=count,
                                 offset=ofs
                                 )
        if 'ret' not in dir( ):
            raise mathDictKeyError( '%s is not a valid column identifier.'
                                    % internal_column )
        return( ret )

    def _mat(self, column_indexes, lag=0 ):
        '''(for internal use): returns a matrix consisting of columns specified
        by their internal column identifier (index or local column name) in the
        column_indexes list.  Strings in the column_indexes list other than
        internal column identifiers are assumed to be calculated columns.  They
        are retrieved by through the .__getitem__( ) method that checks for
        pre-calculated columns by way of .power( ) and .crossproduct( ) before
        calculating a column.
        '''
        ret = list( )
        for i in range( len( column_indexes ) ):
            if isinstance( column_indexes[i], str ) and \
               column_indexes[i] not in self.local.keys( ) and \
               column_indexes[i] not in self._column_names:
                ret.append( self.__getitem__( column_indexes[i],
                                              vector='row' ) )
            else:
                ret.append( self._vec( column_indexes[i], lag=lag ) )
        ret = np.stack( ret, 1 )
        return( ret )

    def get_column(self, column_name, lag=0, vector='column' ):
        '''Returns the column with the specified name from either the matrix of
        original columns or the local column store.

        No check is performed to ensure that the requested column is not
        masked, and calculated columns are not returned.
        '''
        INTERNAL_NOTE = '''
        All columns are stored as 1-dimensional arrays, which numpy perceives
        as row vectors unless they are transposed.  Except for when a slice or
        integer subscript causes .__getitem__( ) to call ._mat( ) directly,
        other methods pass `vector`(='column' by default) to .get_column( ) so
        that .get_column( ) can transpose the vector.  Stored vectors are
        transposed via .get_column( ) calling ._mat( ) prior to manipulation or
        return by single-column-return methods such as .power( ) and
        .crossproduct( ).
        '''
        if not lag:
            lag = 0
        if vector == 'column':
            ret = self._mat( [column_name], lag=lag )
        else:
            ret = self._vec( column_name, lag=lag )
        return( ret )

    def power(self, column_name, power, lag=0, vector='column' ):
        '''Returns the requested column with each cell raised to the requested
        power.  Checks to see if the power has been pre-calculated and returns
        the pre-calculated column if present.
        '''
        if isinstance( power, str ):
            if power.isdigit( ):
                power = int( power )
            else:
                raise TypeError( '%s is not a valid power.' % power )
        if self.cache_powers >= power and \
           power > 1 and \
           column_name in self._column_names:
            index_base, index = self._ofs_start
            index += len( self._column_names ) * (power - 2)
            index += self._column_names.index( column_name )
            ret = self.get_column( index, lag=lag, vector=vector )
        else:
            ret = self.get_column( column_name, lag=lag, vector=vector )
            ret = np.power( ret, power )
        return( ret )

    def crosspower(self, column_a, power_a, column_b, power_b,
                   lag_a=None, lag_b=None, vector='column' ):
        '''Returns the product of two columns, each raised to the specified
        power.  Makes use of precalculated columns if applicable.
        '''
        NoneSet = {None, ''}
        # Lags
        if lag_a in NoneSet:
            lag_a = 0
        else:
            lag_a = int( lag_a )
        if lag_b in NoneSet:
            lag_b = 0
        else:
            lag_b = int( lag_b )
        # Powers
        if power_a in NoneSet:
            power_a = 1
        else:
            power_a = int( power_a )
        if power_b in NoneSet:
            power_b = 1
        else:
            power_b = int( power_b )
        # Math
        ret = self.crossproduct( column_a,
                                 column_b,
                                 lag_a=lag_a,
                                 lag_b=lag_b,
                                 vector=vector )
        if power_a > 1:
            ret = ret * self.power( column_a,
                                    power_a - 1,
                                    lag=lag_a,
                                    vector=vector )
        if power_b > 1:
            ret = ret * self.power( column_b,
                                    power_b - 1,
                                    lag=lag_b,
                                    vector=vector )
        return( ret )

    def crossproduct(self, column_a, column_b,
                     lag_a=0, lag_b=0, vector='column' ):
        '''Returns a column in which each row `i` is `column_a[i]` *
        `column_b[i]`.  Checks to see if crossproducts have been pre-calculated
        and returns the pre-calculated column if present.  If `column_a` and
        `column_b` identify the same column, the requested is transfered to the
        .power( ) method.
        '''
        if column_a == column_b:
            return( self.power( column_a, 2, vector=vector ) )
        orig_columns = column_a in self._column_names and \
                       column_b in self._column_names
        if self.cache_crossproducts == True and \
           lag_a == lag_b and \
           orig_columns:
            indices = [i for i in range( len( self._column_names ) )]
            indices = [i for i in itertools.combinations( indices, 2 )]
            cp_index = (self._column_names.index( column_a ),
                        self._column_names.index( column_b ))
            if cp_index[1] < cp_index[0]:
                cp_index_one, cp_index_zero = cp_index
                cp_index = (cp_index_zero, cp_index_one)
            cp_index = self._ofs_start[0] + indices.index( cp_index )
            return( self.get_column( cp_index, lag=lag_a, vector=vector ) )
        else:
            ret = self.get_column( column_a, lag=lag_a, vector=vector )
            ret = ret * self.get_column( column_b, lag=lag_b, vector=vector )
        return( ret )

    def mask_all(self, except_intercept=False, clear_calculated=True ):
        '''Masks the Intercept column (by default), every column in the matrix
        of original columns, and every local column, leaving calculated columns
        unaffected (by default).

        Parameters
        ----------
        except_intercept : boolean, optional
            If True, then the Intercept will be unmasked regardless of its
            state prior to this method call.
        clear_calculated : boolean, optional
            If True, then the list of calculated columns will be deleted.
        '''
        self.mask( 'replace', [True for i in range( len(
                                                  self._column_names ) + 1 )] )
        for column in self.local.keys( ):
            self.local_mask( 'add', column )
        if except_intercept:
            self.mask( '__setitem__', 0, False )
        if clear_calculated:
            self.calculated_columns( 'replace', [] )

    def unmask_all(self):
        '''Unmasks the Intercept column and every column in the matrix of
        original columns.
        '''
        self.mask( 'replace', [False] )
        self.local_mask( 'clear' )

    def set_mask(self, column_name, mask=True ):
        '''Sets the mask of the specified column to the specified value,
        extending the length of the mask sequence if necessary.  If
        `column_name` == 'Intercept', then it sets the mask of the Intercept
        column instead.
        '''
        if column_name == 'Intercept':
            self.mask( '__setitem__', 0, mask )
            return( )
        elif column_name in self._column_names:
            index = self._column_names.index( column_name ) + 1
            if index >= len( self.mask( ) ):
                self.mask( 'extend', [False for i in \
                                    range( len( self.mask( ) ), index + 1 )] )
            self.mask( '__setitem__', index, mask )
        elif column_name in self.local.keys( ) and mask == True:
            self.local_mask( 'add', column_name )
        elif column_name in self.local.keys( ) and mask == False:
            self.local_mask( 'discard', column_name )
        else:
            raise KeyError( '%s is not a valid column name.' % column_name )

    def add(self, column_string ):
        '''Adds column_string to the matrix represented by the mathDict( ), and
        returns the mathDict( ) if successful.

        Raises
        ------
        UnsupportedColumn(Warning)
            If `column_string` is neither the name of a shared or local column,
            nor something that mathDict( ) can calculate therefrom.  `.args[1]`
            == `.columns` is a list of the unsupported column(s).
        '''
        if column_string in self._column_names or \
           column_string in self.local or \
           column_string == 'Intercept':
            self.set_mask( column_name=column_string, mask=False )
            return( self )
        elif column_string in self.calculated_columns( ):
            return( self )
        else:
            if column_string not in self.strings_checked:
                try:
                    cs_append = column_string.count( ':' ) == 0
                    if cs_append:
                        self.calculated_columns( 'append', column_string )
                    attempt = self._handle_str( column_string,
                                                expanded_columns=True )
                    getattr( self, attempt[0] )( **attempt[1] )
                    if not cs_append:
                    #if len( attempt ) == 3:
                        #self.calculated_columns.remove( column_string )
                        for column in attempt[2]:
                            self.set_mask( column_name=column, mask=False )
                        return( self )
                    self.strings_checked.add( column_string )
                except mathDictKeyError as e:
                    if cs_append:
                        self.calculated_columns( 'remove', column_string )
                    raise UnsupportedColumn( "mathDict( ) neither contains a "
                         "column named, nor supports the calculation of, '%s'."
                         % column_string, [column_string] )
            elif column_string not in self.calculated_columns( ):
                self.calculated_columns( 'append', column_string )
            return( self )

    def add_from_RHS(self, formula, defer_calculations=False ):
        '''Adds columns to the matrix represented by the mathDict( ) based on
        terms in the right hand side of a string representation of a formula.

        If there is one or more '~' characters in the formula string,
        everything to the left of the first '~' character will be ignored.
        Then, the string will be divided into `column_string`s by splitting on
        '+' and '-' characters  that are not enclosed within brackets, and
        .add( column_string ) will be attempted for each column string.

        Note: Subtracting in lieu of adding negatives is not currently
        supported, but there is no error checking for this.  Formula strings
        split on the minus sign in anticipation of a subsequent release
        supporting subtraction directly.

        Returns
        -------
        self : mathDict
            If there were zero '~' characters in the original string and every
            .add( ) attempt was successful.
        string
            If there were one or more '~' characters in the original string,
            then the substring to the left of the first '~' character, stripped
            of padding whitespaces, is returned if/when every .add( ) attempt
            is successful.

        Raises
        ------
        UnsupportedColumn(Warning)
            If one or more column strings is unsupported.  `.args[1]` ==
            `.columns` is a list of the unsupported column(s).  `.LHS` contains
            the string to the left of the first `~`, if any, stripped of
            padding whitespaces.  All supported columns are still added even
            when an unsupported column occurds mid-formula.
        '''
        LHS = None
        unsupported = list( )
        if formula.count( '~' ) > 0:
            LHS, formula = formula.split( '~', 1 )
        for term in terms_in( formula ):
            try:
                self.add( despace( term ) )
            except UnsupportedColumn as w:
                unsupported.extend( w.columns )
        if len( unsupported ) == 0 and LHS == None:
            return( self )
        if LHS != None:
            LHS = LHS.replace( ' ', '' )
        if len( unsupported ) > 0:
            raise UnsupportedColumn( "One or more RHS terms could not be "
                                     "added.  RHS: '%s'."
                                     % formula, unsupported, LHS=LHS )
        return( LHS )

    def config_to_dict(self, save_terms=True):
        '''Returns a dict( ) subclass object containing the configuration of
        this mathDict( ) matrix that has a .rebuild( SharedDataArray=REQUIRED )
        method to recreate the mathDict( ) in a different process.

        *NOTE: Hypothesis information is not stored.*
        '''
        INTERNAL_NOTE = '''Where a call to deepcopy( ) is needed, it is
        performed upon rebuilding the mathDict( ) from the configuration
        dictionary, as opposed to on creating the configuration dictionary.'''
        config                        = mathDictConfig( )
        config['items']               = self.items
        config['_column_names']       = self._column_names
        config['mask']                = self._mask
        config['dtypes']              = self.dtypes
        config['calculated_columns']  = self._calculated_columns
        config['cache_crossproducts'] = self.cache_crossproducts
        config['cache_powers']        = self.cache_powers
        config['_orig_padding']       = self._orig_padding
        if save_terms:
            config['terms']           = self.terms
        config['local_calculated']    = self.local_calculated
        config['local_mask']          = self._local_mask
        return( config )

    def iter_map(self, arg_iterable,
                       func,
                       placement=0,
                       process_count=None,
                       use_kwargs=False,
                       number_results=False ):
        '''Comparable to .map( ) except that it returns an unsorted iterable
        instead of a list( ).

        Parameters
        ----------
        arg_iterable : iterable of tuples
            tuple of positional arguments to be passed into `func`.  The final
            value in the tuple can be a dict( ) of keyword arguments if
            `use_kwargs` is set to True.
        func : function
            The function to be called.  It must be pickleable.
        placement : int or string, optional
            If an integer, then the matrix will be inserted as a positional
            argument at this location.  If a string, then the matrix will be
            passed in as a keyword argument using this keyword.
        process_count : int, optional
            The number of child processes to launch.  If this is not set, then
            Python will try to figure it out using a minimum of two processes,
            but Python isn't good at figuring it out so it is always better to
            provide this argument.
        use_kwargs : bool
            If True, then the final value in each tuple of arguments will be
            treated as a dict( ) of keyword arguments for `func`.
        number_results : bool
            If True, each result will be provided in the form of a tuple in
            which the first value is the position of the argument tuple in
            `arg_iterable` from which the result was computed, and the second
            value is the result itself.

        Returns
        -------
        Iterable
            Results in unsorted, iterable form.
        '''
        ProcessQueue = Queue( )
        ReturnQueue = Queue( )
        procList = list( )
        if process_count == None:
            process_count = max( cpu_count( ), 2 )
        for i in range( process_count ):
            p = Process( target=_mapper,
                         args=(ProcessQueue,
                               ReturnQueue,
                               self.buffer,
                               self.config_to_dict( ),
                               func,
                               placement )
                         )
            p.start( )
            procList.append( p )
        kwargs = dict( )
        if number_results:
            rid = 0
            for args in arg_iterable:
                if use_kwargs:
                    kwargs = args[len( args ) - 1]
                    pargs = args[:len( args ) - 1]
                else:
                    pargs = args
                ProcessQueue.put( (rid, pargs, kwargs) )
                rid += 1
        elif use_kwargs:
            for args in arg_iterable:
                kwargs = args[len( args ) - 1]
                pargs = args[:len( args ) - 1]
                ProcessQueue.put( (0, pargs, kwargs) )
        else:
            for args in arg_iterable:
                ## Yes, this involves redundant code.  It also minimizes
                ## branching in an algorithm that could be looped through tens
                ## of thousands or in some scenarios millions of times.
                ProcessQueue.put( (0, args, kwargs) )
        for i in range( len( procList ) ):
            ProcessQueue.put( 'Terminate.' )
        termination_count = 0
        while termination_count < len( procList ):
            QueueObject = ReturnQueue.get( )
            if QueueObject == 'Terminated.':
                termination_count += 1
            elif number_results:
                yield( QueueObject )
            else:
                yield( QueueObject[1] )

    def map(self, arg_iterable,
                  func,
                  placement=0,
                  process_count=None,
                  use_kwargs=False,
                  ordered=False ):
        '''Uses parallel processes and shared memory to call `func` with each
        tuple of arguments, also passing in the matrix as an argument.

        Parameters
        ----------
        arg_iterable : iterable of tuples
            tuple of positional arguments to be passed into `func`.  The final
            value in the tuple can be a dict( ) of keyword arguments if
            `use_kwargs` is set to True.
        func : function
            The function to be called.  It must be pickleable.
        placement : int or string, optional
            If an integer, then the matrix will be inserted as a positional
            argument at this location.  If a string, then the matrix will be
            passed in as a keyword argument using this keyword.
        process_count : int, optional
            The number of child processes to launch.  If this is not set, then
            Python will try to figure it out using a minimum of two processes,
            but Python isn't good at figuring it out so it is always better to
            provide this argument.
        use_kwargs : bool
            If True, then the final value in each tuple of arguments will be
            treated as a dict( ) of keyword arguments for `func`.
        ordered : bool
            If True, then the results will be listed in the order of the
            argument tuples.  Otherwise, results may be in any order.  Note:
            argument tuples are processed asynchronously (out-of-sequence)
            either way. This option sorts the results after they have been
            computed.

        Returns
        -------
        list
            Results in list( ) form.

        Example
        -------
        >>> def sum_row( matrix, row ):
        >>>     # Put this in a_file.py and import it if you receive pickle-
        >>>     # related errors.
        >>>     return( sum( matrix[row,:] ) )
        >>> matrix = np.array( [i for i in range( 24 )] ).reshape( (6, 4) )
        >>> RA, MD = mathDictMaker.fromMatrix( matrix, integer=True )
        >>> res = MD.map( [(i,) for i in range( 6 )], sum_row, ordered=True )
        >>> print( res )
        [6, 22, 38, 54, 70, 86]
        '''
        retList = list( )
        for ret in self.iter_map( arg_iterable=arg_iterable,
                                  func=func,
                                  placement=placement,
                                  process_count=process_count,
                                  use_kwargs=use_kwargs,
                                  number_results=ordered ):
            retList.append( ret )
        if ordered:
            retList.sort( )
            retList = [tpl[1] for tpl in retList]
        return( retList )

class TestCase(unittest.TestCase):
    '''Included solely for internal use.
    '''
    _dirname   = 'test_files/'
    _rfilename = _dirname + 'exp_'
    _wfilename = _dirname + 'act_'
    def assertDictUnsortedEqual(self, dictA, dictB ):
        '''Compares two dict( )s in which each entry is expected to have an
        iterable for its value.
        '''
        self.assertSetEqual( set( dictA.keys( ) ), set( dictB.keys( ) ) )
        for key in dictA.keys( ):
            self.assertSetEqual( set( dictA[key] ), set( dictB[key] ) )

    @contextmanager
    def assertRaisesWithMessage(self, e, msg, index=0 ):
        with self.assertRaises( e ) as cm:
            yield
        self.assertEqual( cm.exception.args[index], msg )

    def assertFileLineSetEqual(self, f1, f2, msg=None ):
        f1 = TestCase._dirname + f1
        f2 = TestCase._dirname + f2
        lines1 = open( f1, 'r' ).readlines( )
        lines2 = open( f2, 'r' ).readlines( )
        self.assertEqual( len( lines1 ), len( lines2 ) )
        self.assertSetEqual( set( lines1 ), set( lines2 ), msg )

    @contextmanager
    def assertStreamFileEqual(self, filename, encoding=None ):
        _rfilename = TestCase._rfilename + filename
        _wfilename = TestCase._wfilename + filename
        stream = StringIO( )
        err = None
        try:
            yield( stream )
        except Exception as e:
            err = e
        finally:
            expected = ''
            with suppress(FileNotFoundError):
                if encoding != None:
                    expected = open( _rfilename,
                                    'r',
                                    encoding=encoding ).read( )
                else:
                    expected = open( _rfilename, 'r' ).read( )
            try:
                self.assertMultiLineEqual( stream.getvalue( ), expected )
            except AssertionError as e:
                stream.seek( 0, 0 )
                if encoding != None:
                    wf = open( _wfilename,
                               'w',
                               encoding=encoding
                             ).writelines( stream.readlines( ) )
                else:
                    wf = open( _wfilename,
                               'w' ).writelines( stream.readlines( ) )
                if err == None:
                    err = e
            if err != None:
                raise err

    @contextmanager
    def assertStdoutStringEqual(self, string ):
        _old_target = getattr(sys, 'stdout')
        _new_stream = StringIO( )
        err = None
        setattr( sys, 'stdout', _new_stream )
        try:
            yield
        except Exception as e:
            err = e
        finally:
            setattr( sys, 'stdout', _old_target )
            if err != None:
                raise err
        try:
            self.assertEqual( _new_stream.getvalue( ).rstrip( ), string )
        except AssertionError as e:
            _new_stream.seek( 0, 0 )
            raise AssertionError( 'Output was `%s`.  Expected `%s`.'
                                  % (_new_stream.getvalue( ), string) )

    class assertStdoutFileEqual(ContextDecorator):
        def __init__(self, filename, parent=None, encoding=None ):
            self._old_target = None
            self._rfilename = TestCase._rfilename + filename
            self._wfilename = TestCase._wfilename + filename

            if encoding != None:
                self._encoding = encoding
            if parent != None:
                self._parent = parent
            else:
                self._parent = unittest.TestCase( )

        def __enter__(self):
            self._old_target = getattr(sys, 'stdout')
            self._new_stream = StringIO( )
            setattr( sys, 'stdout', self._new_stream )
            return( self._new_stream )

        def __exit__(self, exc_type, exc, exc_tb):
            setattr( sys, 'stdout', self._old_target )
            if not exc_type == exc == exc_tb == None:
                return( False )
            expected = ''
            with suppress(FileNotFoundError):
                if getattr( self, '_encoding', None ) != None:
                    with open( self._rfilename,
                               'r',
                               encoding=self._encoding ) as rf:
                        expected = rf.read( )
                else:
                    with open( self._rfilename, 'r' ) as rf:
                        expected = rf.read( )
            try:
                unittest.TestCase.assertMultiLineEqual( self._parent,
                                       self._new_stream.getvalue( ), expected )
            except AssertionError as e:
                self._new_stream.seek( 0, 0 )
                if getattr( self, '_encoding', None ) != None:
                    wf = open( self._wfilename,
                               'w',
                               encoding=self._encoding
                             ).writelines( self._new_stream.readlines( ) )
                else:
                    wf = open( self._wfilename,
                               'w'
                             ).writelines( self._new_stream.readlines( ) )
                raise e

if __name__ == "__main__":
    pass