#!/usr/bin/env python

from distutils.core import setup

setup(name='skysat_stereo_snow',
      description='library for DEM generation and snow depth estimation workflows from Planet SkySat-C imagery',
      author='Rainey Aberle and Ellyn Enderlin',
      author_email='raineyaberle@u.boisestate.edu',
      license='MIT',
      long_description=open('README.md').read(),
      url='https://github.com/RaineyAbe/skysat_stereo_snow.git',
      packages=['skysat_stereo_snow'],
      install_requires=['requests']
      )
