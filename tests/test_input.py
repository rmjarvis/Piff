# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

from __future__ import print_function
import piff
import os
import galsim
import numpy as np

from piff_test_helper import get_script_name, timer


def setup():
    """Make sure the images and catalogs that we'll use throughout this module are done first.
    """
    gs_config = galsim.config.ReadConfig(os.path.join('input','make_input.yaml'))[0]
    galsim.config.Process(gs_config)

@timer
def test_basic():
    """Test the (usual) basic kind of input field without too many bells and whistles.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    dir = 'input'
    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # Simple with one image, cat
    config = {
                'dir' : dir,
                'image_file_name' : image_file,
                'cat_file_name': cat_file
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 100

    # Can omit the dir and just inlcude it in the file names
    config = {
                'image_file_name' : os.path.join(dir,image_file),
                'cat_file_name': os.path.join(dir,cat_file)
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 100

    # 3 images in a list
    image_files = [ 'test_input_image_%02d.fits'%k for k in range(3) ]
    cat_files = [ 'test_input_cat_%02d.fits'%k for k in range(3) ]
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 3
    np.testing.assert_array_equal([len(p) for p in input.image_pos], 100)

    # 3 images using glob
    image_files = 'test_input_image_*.fits'
    cat_files = 'test_input_cat_*.fits'
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 3
    np.testing.assert_array_equal([len(p) for p in input.image_pos], 100)


@timer
def test_invalid():
    """Test invalid configuration elements
    """

    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # Leaving off either image_file_name or cat_file_name is an error
    config = { 'image_file_name' : os.path.join('input',image_file) }
    np.testing.assert_raises(AttributeError, piff.InputFiles, config)
    config = { 'cat_file_name': os.path.join('input',cat_file) }
    np.testing.assert_raises(AttributeError, piff.InputFiles, config)

    # Invalid values for image or cat name
    config = { 'image_file_name' : os.path.join('input',image_file), 'cat_file_name' : 17 }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'cat_file_name': os.path.join('input',cat_file), 'image_file_name' : 17 }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)

    # Using the wrong name (either as a glob or not) should raise an appropriate exception
    config = { 'dir' : 'input', 'image_file_name' : 'x'+image_file, 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : 'x*'+image_file, 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : image_file, 'cat_file_name' : 'x'+cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : image_file, 'cat_file_name' : 'x*'+cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)

    # nimages given but wrong
    image_files = 'input/test_input_image_*.fits'
    cat_files = 'input/test_input_cat_*.fits'
    image_files_1 = { 'type': 'Eval', 'str': '"input/test_input_image_01.fits"' }
    cat_files_1 = { 'type': 'Eval', 'str': '"input/test_input_cat_01.fits"' }
    config = { 'nimages' : 2, 'image_file_name' : image_files, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : 2, 'image_file_name' : image_files, 'cat_file_name': cat_files_1 }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : 2, 'image_file_name' : image_files_1, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : 0, 'image_file_name' : image_files, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : -1, 'image_file_name' : image_files, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)

    # No images in list
    config = { 'dir' : 'input', 'image_file_name' : [], 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : image_file, 'cat_file_name' : [] }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)

@timer
def test_chipnum():
    """Test the ability to renumber the chipnums
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    # First, the default is to just use the index in the image list
    dir = 'input'
    image_files = 'test_input_image_*.fits'
    cat_files = 'test_input_cat_*.fits'
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.chipnums == range(3)

    # Now make the chipnums something else
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files,
                'chipnum' : [ 5, 6, 7 ]
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.chipnums == [ i+5 for i in range(3) ]

    # Use the GalSim Eval capability to get the index + 1
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files,
                'chipnum' : '$image_num + 1'
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.chipnums == [ i+1 for i in range(3) ]



if __name__ == '__main__':
    setup()
    test_basic()
    test_invalid()
    test_chipnum()
