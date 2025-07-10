#!/usr/bin/env python

from distutils.core import setup

setup(name='skysat_stereo_snow',
      version='0.1',
      description='library for DEM generation workflows from Planet SkySat-C imagery ',
      author='Rainey Aberle and Ellyn Enderlin',
      author_email='raineyaberle@u.boisestate.edu',
      license='MIT',
      long_description=open('README.md').read(),
      url='https://github.com/uw-cryo/skysat_stereo.git',
      packages=['skysat_stereo_snow'],
      install_requires=['requests']
      )
