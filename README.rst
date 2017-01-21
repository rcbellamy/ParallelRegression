ParallelRegression
==================

ParallelRegression is a set of Python tools for using parallel processes to analyze a data set in shared memory.  mathDict is a set of tools for assembling a matrix in a single block of shared memory and then creating different matrix views that combine columns in shared memory with process-local columns for analysis.  termSet( ) and the more generic categorizedSetDict( ) and setList( ) are classes that facilitate tracking of metadata regarding the data set being analyzed.  For example, regression in floating point mathematics can be sensitive to the ordering of terms.  Using tools built on the ordered set class setList( ) to track term metadata facilitates the reproducability of results.  ParallelRegression also includes functions that simplify working with strings that contain bracketed substrings, such as formulas.

mathDict Overview
-----------------

mathDict is a set of tools in the Parallel Regression package.  The primary function of mathDict is to allow multiple processes to easily share read-only access to a data array in shared memory.  This shared memory array contains shared columns that individual processes can combine with local columns to efficiently build matrices that can, e.g., test hypotheses.

Data manipulation that needs to be shared is done prior to launching multiple processes.  A mathDictMaker( ) object is used to assemble disparate columns into a column-major data matrix stored in a single block of shared memory, and optionally to precompute common manipulations prior to creating the mathDict( ) object.  At that point, mathDict facilitates different processes using different views of the same shared read-only data matrix, which can include integrating columns consisting of process-local data.

As different processess handle specific analysis tasks, they can use the mathDict( ) to tailor and re-tailor their view of the data matrix as needed for the particular task.  Using a series of simple, easily-readable lines of code, the analyst or developer can instruct mathDict to hide columns, add columns that are lags, crossproducts, or powers of other columns, and keep track of which columns are part of a linear hypothesis regarding the matrix to be tested by an F or Wald test.  The mathDict( ) will track all of this, while delaying all data processing until the data is retrieved from the mathDict( ) in order to minimize memory-allocating operations and compute the desired output as efficiently as possible.

Making a mathDict( ) object using mathDictMaker( )
--------------------------------------------------

mathDict( ) objects are created by adding columns to a mathDictMaker( ) object as if it were a dictionary.  Each column must be a sequence such as a list or an array that consists of numbers, and all must have the same length.  In the following example, `A`, `B`, `C`, and `D` are all lists of floats, and `d1` is a list of integer values 0 and 1::

    mdMaker = mathDictMaker( )
    mdMaker['A']  = A
    mdMaker['B']  = B
    mdMaker['C']  = C
    mdMaker['D']  = D
    mdMaker['d1'] = d1
    SharedDataArray, mDict = mdMaker.make( cache_crossproducts=True, cache_powers=2 )

The mathDictMaker simply stores pointers to these lists until the .make( ) method is called.  At that point, mathDictMaker( ) determines how much memory to allocate in order to store all of the original columns (of which there are five in this example) and the requested manipulations thereof in a single block of memory.  mathDictMaker( ) assembles the columns into the multiprocessing.sharedctypes.RawArray( ) object (named SharedDataArray in this example) and stores the metadata in the mathDict( ) object (here named mDict).

In this example, the first keyword argument tells mathDictMaker( ) to compute the crossproduct of each of the five columns with each of the other four columns and store these manipulations in the SharedDataArray.  The second keyword argument tells mathDictMaker( ) to compute the product of each of the five columns with itself.  Although mathDictMaker( ) will compute `d1 ** 2`, the mathDict toolset will not allow both `d1` and `d1 ** 2` to be included in a matrix built for hypothesis testing *if* it is told that `d1` is a dummy term.

Sharing a mathDict data matrix across processes
-----------------------------------------------

Separation of the data from the metadata makes sharing the data array across processes simple.  The SharedDataArray object returned from mathDictMaker( ).make( ) is a shared RawArray from the standard library's multiprocessing module (specifically, the sharedctypes submodule).  It must be created by the parent process before the child processes are launched, and should be passed into the child processes as an argument upon launching each child process::

    from worker import QueueWorker, QueueWorkerHypothesis
	from multiprocessing import Queue, Process
    ProcessQueue = Queue( )
    ReturnQueue = Queue( )
    procList = list( )
    for i in range( 2 ):
        p = Process( target=QueueWorker,
                     args=(ProcessQueue,
                           SharedDataArray,
                           ReturnQueue)
                     )
        p.start( )
        procList.append( p )

The result is that the collection of columns assembled using mathDictMaker( ) is stored in a single block of memory to which all processes have raw byte-array access.  The mathDict( ) object contains the metadata and methods used to read data from the shared memory.  Its .config_to_dict( ) method collapses the lightweight mathDict( ) into a super-lightweight format that is merely a dict( ) containing the metadata that has a single .rebuild( ) method added for reconstructing the mathDict( ).  Pointers to the original metadata are replaced with deepcopy( )s upon calling .rebuild( ), allowing the single configuration dictionary to be used to create many different mathDict( ) objects individually tailored for different analysis tasks::

    mDictCfg = mDict.config_to_dict( )
    for let in {'A', 'B', 'C', 'D'}:
        mDictNew = mDictCfg.rebuild( SharedDataArray )
        mDictNew.set_mask( let )
        tpl = (mDictNew.config_to_dict( ), let)
        ProcessQueue.put( tpl )

The configuration dictionary can also be attached as an attribute to other objects and then be used multiple times for different analyses relating to the object occuring in the parent or any child process.  The only argument that needs to be passed into the .rebuild( ) method is the SharedDataArray.

This example illustrates how to accomplish a result similar to using multiprocessing.Pool's .map( ) method to parallelize analyses in a manner that allows all instances of analysis being performed to efficiently share the same SharedDataArray::

    for i in range( len( procList ) ):
        ProcessQueue.put( 'Terminate.' )
        
    terminationCount = 0
    while terminationCount < len( procList ):
        QueueObject = ReturnQueue.get( )
        if QueueObject == 'Terminated.':
            terminationCount += 1
        print( QueueObject )

Analyzing a combination of shared and local data
------------------------------------------------

After the SharedDataArray has been assembled, individual analysis tasks can be performed that involve a matrix consisting of a combination of columns in the SharedDataArray and process-local columns that are specific to that analysis task.  As before when adding shared columns to the mathDictMaker( ), local columns are added to mathDict( ) objects by adding a sequence of cell values such as a list or array as if adding a single new entry to a dict( ) object, where the dict( ) key is the column name.  In this example, `d2` is another list of integer values 0 and 1::

    import statsmodels.api
    def QueueWorker( ProcessQueue, SharedDataArray, ReturnQueue ):
        QueueObject = ProcessQueue.get( )
        while QueueObject != 'Terminate.':
            mDictCfg, let = QueueObject
            mDict = mDictCfg.rebuild( SharedDataArray )
            mDict['d2'] = d2

Because this mathDict( ) was created in the local process using the configuration dictionary's .rebuild( ) method, it can be further customized by, e.g., adding calculated columns or hiding columns using .set_mask( ) without affecting any other analysis based on the SharedDataArray.

Columns that are hidden using .set_mask( ) or .mask_all( ) are excluded from the matrix represented by the mathDict( ) object, but can still be retrieved separately using the mathDict( ) by specifying the hidden column's name.  The matrix represented by the mathDict( ) is retrieved using a slice identifying the whole object (i.e. [:]).  This allows both the left-hand-side (LHS) and right-hand-side (RHS) of a regression to be retrieved from the same mathDict( ) object in the same line of code.

In this example, the Queue of objects to be processed consists of tuples where the first item is a mathDict( ) with one column hidden, and the second item is the name of the hidden column.  The hidden column is used as the LHS::

    # Continuation of def QueueWorker
            model = statsmodels.api.OLS( mDict[let], mDict[:]
                                        ).fit( cov_type='HC0' )
            ret = ' + '.join( mDict.columns ) + ' => ' + \
                  ' + '.join( [str( p ) for p in model.params] )
            ReturnQueue.put( ret )
            QueueObject = ProcessQueue.get( )
        ReturnQueue.put( 'Terminated.' )

This example simply computes some coefficients and prints them in a not-very-readable format because the point is to demonstrate the manner in which the mathDict( ) is used in order to perform the analysis.  See the source code for, or the API documentation on, the syncText( ) function for a better way to format output of this nature.

Testing a linear hypothesis using mathDict
------------------------------------------

mathDict( ) objects contain a .hypothesis( ) attribute that is of a specialized class for testing linear hypotheses about the matrix represented by the mathDict( ) object.  The hypothesis can involve columns already in that matrix, other columns in the SharedDataArray, process-local columns, and certain manipulations thereof::

    def QueueWorkerHypothesis( ProcessQueue, SharedDataArray, ReturnQueue ):
        mapLHS_RHS = {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'A'}
        QueueObject = ProcessQueue.get( )
        while QueueObject != 'Terminate.':
            mDictCfg, let = QueueObject
            mDict = mDictCfg.rebuild( SharedDataArray )
            mDict['d2'] = d2
            mDict.set_mask( 'd2' )
            mDict.hypothesis.add( 'd2' )
            mDict.hypothesis.add( 'd2 * %s' % mapLHS_RHS[let] )
            mDict.hypothesis.add( '%s ** 2' % mapLHS_RHS[let] )

mathDict( ) simply tracks this metadata until the mathDict( ).hypothesis.make( ) method is called to create the RHS (i.e. X) matrix as well as the "R" and "r" matrices used for testing a linear hypothesis.  Statsmodels and Numpy both offer core multiple linear regression functionality::

    # Continuation of def QueueWorkerHypothesis
            X, R, r = mDict.hypothesis.make( )
            model = statsmodels.api.OLS( mDict[let], X ).fit( cov_type='HC0' )

mathDict contains a function for using the results from the linear regression along with the R and r matrices from mathDict( ).hypothesis.make( ) to compute a heteroskedasticity-robust F statistic::

    # Continuation of def QueueWorkerHypothesis
            u = model.resid
            coefs = model.params
            F_stat = FStatistic( X, u, coefs, R, r )
            ret = 'Hypothesis that in modeling %s, columns: `d2`, `d2 * %s`' \
                ', and `%s ** 2` are all 0 has an F statistic of %.3f.' \
                % (let, mapLHS_RHS[let], mapLHS_RHS[let], F_stat)
            ReturnQueue.put( ret )
            QueueObject = ProcessQueue.get( )
        ReturnQueue.put( 'Terminated.' )

See the source code for, or the API documentation on, FStatistic( ) for mathematical details.