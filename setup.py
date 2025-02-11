#!/usr/bin/env python3

from setuptools import setup

ext_modules = [ ]
cmdclass = { }


if __name__ == '__main__':
    setup(name='ddp',
          use_scm_version=dict(
              root='.',
              relative_to=__file__,
              version_scheme='no-guess-dev'
          ),
          ext_modules=ext_modules,
          cmdclass=cmdclass,
          scripts=[]
          )