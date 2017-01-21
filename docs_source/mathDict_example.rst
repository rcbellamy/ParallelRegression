.. module:: ParallelRegression

.. _mathDict_example:

mathDict Example
================

.. _mathDict_example.py:

mathDict_example.py Source:
---------------------------

::

    from ParallelRegression import *

    A = [457.641, 269.473, 666.114, 738.578, 384.412, 436.901, 616.734,
         800.865, 799.396, 338.704, 647.796, 530.703, 464.395, 580.856,
         740.477, 262.685, 902.082, 455.818, 260.115, 736.013, 506.098,
         752.046, 789.867, 643.288, 637.688, 735.963, 366.387, 341.807,
         302.911, 498.582, 813.982, 814.821, 420.031, 448.213, 307.052,
         430.88,  416.088, 845.837, 410.257, 386.846, 910.287, 824.109,
         391.376, 632.234, 620.801, 395.589, 777.192, 613.002, 442.574,
         534.007, 845.773, 365.875, 357.484, 465.222, 598.116, 546.189,
         387.921, 796.199, 660.886, 504.647, 334.781, 426.913, 460.404,
         835.071, 704.785, 799.473, 362.1,   493.072, 457.163, 819.701,
         619.863, 691.317, 385.449, 451.347, 336.794, 372.412, 453.459,
         419.277, 416.966, 772.945, 943.82,  520.426, 345.733, 411.217,
         740.681, 682.992, 719.678, 727.342, 597.919, 512.705, 593.008,
         588.302, 690.788, 908.377, 704.167, 698.847, 433.792, 453.611,
         526.038, 382.376]

    B = [119.301,  66.349, 161.152, 145.175,  69.001, 219.116,  94.441,
         213.002,  92.834, 100.679, 164.569,  80.688,  92.686, 219.442,
         221.164,  43.919, 194.728, 213.813,  52.219, 154.47,  135.639,
         186.396, 222.365, 137.722,  82.308, 126.674, 163.315,  52.624,
         100.181, 213.997, 181.496, 164.274, 124.197, 195.611,  26.147,
         164.143, 155.979, 126.052, 169.598, 190.142, 234.881, 138.879,
         141.201, 111.1,   157.582, 134.78,  162.129, 165.268, 233.181,
         110.312, 197.514,  54.899, 133.201, 235.463,  37.009, 237.319,
         136.888, 174.981, 143.82,  130.934, 110.247, 189.622, 203.057,
         265.556, 209.429, 127.223, 126.889, 180.801, 201.507, 199.916,
         111.492, 117.431, 108.005, 171.611, 117.93,  179.91,  222.877,
         197.046, 256.175, 153.089, 220.919, 219.835, 101.607, 156.517,
         186.866, 142.049, 189.638, 180.149, 108.815, 217.281, 111.797,
         176.722, 194.262, 227.524,  71.405, 168.751, 180.718, 159.217,
         214.491, 160.46]

    C = [199.38,  111.518, 171.771, 195.467, 149.69,  129.294, 199.866,
         162.954, 250.934, 144.716,  75.313, 113.178, 173.147, 176.945,
          70.004, 164.992, 240.852, 193.629, 175.825, 164.11,  209.412,
          87.323,  78.069, 120.363, 189.156, 252.551, 120.92,  216.863,
         130.409, 244.084, 169.927, 134.425, 109.229, 126.777, 100.834,
          92.531, 183.025, 274.818, 199.981, 169.116, 208.509, 249.877,
         146.664, 203.326, 131.02,  119.461, 144.568, 182.734, 219.753,
         154.387, 123.408, 167.475, 145.907,  94.822, 235.145,  62.2,
         157.767, 234.263, 115.903,  70.69,  145.207, 166.503, 179.489,
         129.545, 70.273,  192.28,  188.573, 191.433, 217.216, 186.478,
         172.187, 124.913, 100.57,  261.621, 122.669, 218.585,  90.753,
         163.686, 144.649, 205.134, 234.896, 261.634, 222.179, 215.568,
          83.608, 183.83,  103.642,  89.629, 205.168, 208.448, 209.43,
         149.044,  81.287, 231.098, 147.611,  31.635, 143.063, 234.67,
          87.556, 166.715]

    D = [195.67,  129.138, 156.052, 109.087, 153.751, 127.445, 120.588,
         131.854, 173.856, 133.698, 158.244, 150.328, 176.716, 137.254,
         172.858, 122.604, 172.383, 161.954, 130.858, 175.53,  166.751,
         113.442, 124.344, 130.758, 149.935, 132.201, 113.225, 128.479,
         128.796, 173.129, 154.809, 192.431, 174.448, 223.659, 173.727,
         143.076, 170.036, 120.376, 125.871, 115.621, 185.208, 154.787,
         158.684, 150.494, 112.307, 145.116, 148.797, 100.41,  133.531,
         109.915, 183.138, 167.564, 160.118, 146.176, 166.684, 168.459,
         163.29,  135.122, 133.735, 127.535, 111.317, 134.029, 190.334,
         116.149, 118.005, 147.545, 107.594, 183.479, 127.041, 161.095,
         131.097, 178.309, 164.199, 123.923, 124.991, 121.905, 130.822,
         100.141, 103.182, 138.563, 145.071, 168.989, 143.085, 141.522,
         149.141, 138.302, 151.767, 162.769, 137.202, 140.127, 142.806,
          78.273, 128.285, 157.013, 141.791, 142.812, 139.937, 174.723,
         146.152, 133.782]

    d1 = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,0,0,1,
          1,0,0,0,0,0,1,0,0,1,1,0,1,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,1,0,0,
          0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,1,1,
          1,1,1,0,0,0,0]

    if __name__ == "__main__":
        mdMaker = mathDictMaker( )
        mdMaker['A']  = A
        mdMaker['B']  = B
        mdMaker['C']  = C
        mdMaker['D']  = D
        mdMaker['d1'] = d1
        SharedDataArray, mDict = mdMaker.make( cache_crossproducts=True,
                                               cache_powers=2 )

`QueueWorker` and `QueueWorkerHypothesis` can be used interchangeably in the line starting `'p = Process'` to try the different examples::
		
        from worker import QueueWorker, QueueWorkerHypothesis
        from multiprocessing import Queue, Process
        ProcessQueue = Queue( )
        ReturnQueue = Queue( )
        procList = list( )
        for i in range( 2 ):
            p = Process( target=QueueWorkerHypothesis,
                         args=(ProcessQueue,
                               SharedDataArray,
                               ReturnQueue)
                         )
            p.start( )
            procList.append( p )
        
        mDictCfg = mDict.config_to_dict( )
        for let in {'A', 'B', 'C', 'D'}:
            mDictNew = mDictCfg.rebuild( SharedDataArray )
            mDictNew.set_mask( let )
            tpl = (mDictNew.config_to_dict( ), let)
            ProcessQueue.put( tpl )
        
        for i in range( len( procList ) ):
            ProcessQueue.put( 'Terminate.' )
        
        terminationCount = 0
        while terminationCount < len( procList ):
            QueueObject = ReturnQueue.get( )
            if QueueObject == 'Terminated.':
                terminationCount += 1
            print( QueueObject )

.. _worker.py:
			
worker.py Source:
-----------------

::

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
            model = statsmodels.api.OLS( mDict[let], mDict[:]
                                        ).fit( cov_type='HC0' )
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

QueueWorker Output:
-------------------

::

	Intercept + A + B + C + d1 + d2 => 112.685540322 + 0.223817620334 + -0.249513977434 + -0.106076169993 + -71.4369309379 + -7.38778376659
	Intercept + B + C + D + d1 + d2 => -54.5647897751 + 1.12424317191 + 0.530330297026 + 1.36579120342 + 311.067932614 + 18.1086372194
	Intercept + A + C + D + d1 + d2 => 95.7100429924 + 0.508495046485 + -0.307037597337 + -0.688670348879 + -158.160160309 + -0.492666217004
	Terminated.
	Intercept + A + B + D + d1 + d2 => 153.25155005 + 0.449976021676 + -0.575980823121 + -0.54922563694 + -142.118879576 + -14.5960961384
	Terminated.

QueueWorkerHypothesis Output:
-----------------------------

::

	Hypothesis that in modeling D, columns: `d2`, `d2 * A`, and `A ** 2` are all 0 has an F statistic of 3.450.
	Hypothesis that in modeling A, columns: `d2`, `d2 * B`, and `B ** 2` are all 0 has an F statistic of 2.998.
	Hypothesis that in modeling C, columns: `d2`, `d2 * D`, and `D ** 2` are all 0 has an F statistic of 2.292.
	Hypothesis that in modeling B, columns: `d2`, `d2 * C`, and `C ** 2` are all 0 has an F statistic of 0.097.
	Terminated.
	Terminated.