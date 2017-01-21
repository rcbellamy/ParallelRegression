#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

meta = dict( )
with open( 'ParallelRegression/metadata.py' ) as fp:
    exec( fp.read( ), meta )

setup( name             = meta['name'],
       version          = meta['release'],
       author           = meta['author'],
       author_email     = meta['email'],
       url              = meta['url'],
       packages         = ['ParallelRegression'],
       license          = 'AGPLv3',
       classifiers      = ['Development Status :: 4 - Beta',
                           'Intended Audience :: Developers',
                           'Intended Audience :: Science/Research',
                           'License :: OSI Approved :: GNU Affero General Public License v3',
                           'Natural Language :: English',
                           'Programming Language :: Python :: 3',
                           'Programming Language :: Python :: 3 :: Only',
                           'Topic :: Scientific/Engineering',
                           'Topic :: Scientific/Engineering :: Mathematics',
                           'Topic :: Software Development :: Libraries :: Python Modules'],
       keywords         = 'RawArray regression parallel numpy shared array',
       description      = meta['short_desc'],
       long_description = meta['long_desc'],
       install_requires = ['numpy'] )