from ParallelRegression import *

d2 = [0,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,
      1,0,0,1,0,1,0,1,1,1,1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,0,
      1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,
      0,0,0,1,0,1,0]

import statsmodels.api
def QueueWorker( ProcessQueue, SharedDataArray, ReturnQueue ):
    QueueObject = ProcessQueue.get( )
    while QueueObject != 'Terminate.':
        mDictCfg, let = QueueObject
        mDict = mDictCfg.rebuild( SharedDataArray )
        mDict['d2'] = d2
        model = statsmodels.api.OLS( mDict[let], mDict[:] ).fit( cov_type='HC0' )
        ret = ' + '.join( mDict.columns ) + ' => ' + \
              ' + '.join( [str( p ) for p in model.params] )
        ReturnQueue.put( ret )
        QueueObject = ProcessQueue.get( )
    ReturnQueue.put( 'Terminated.' )

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
        X, R, r = mDict.hypothesis.make( )
        model = statsmodels.api.OLS( mDict[let], X ).fit( cov_type='HC0' )
        u = model.resid
        beta = model.params
        F_stat = FStatistic( X, u, beta, R, r )
        ret = 'Hypothesis that in modeling %s, columns: `d2`, `d2 * %s`' \
            ', and `%s ** 2` are all 0 has an F statistic of %.3f.' \
            % (let, mapLHS_RHS[let], mapLHS_RHS[let], F_stat)
        ReturnQueue.put( ret )
        QueueObject = ProcessQueue.get( )
    ReturnQueue.put( 'Terminated.' )