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
import os
import shutil
import numpy as np
import piff

from piff_test_helper import timer

@timer
def test_ensure_dir():
    """Test the ensure_dir utiltity
    """
    d = os.path.join('output', 'test_dir')
    if os.path.exists(d):
        shutil.rmtree(d)

    assert not os.path.exists(d)

    f = os.path.join(d, 'test_file')

    # ensure that the directory needed to write f exisits
    piff.util.ensure_dir(f)

    assert os.path.exists(d)
    assert os.path.isdir(d)

    # write something to f
    with open(f,'w') as fout:
        fout.write('test')

    # doing it again doesn't destroy f
    piff.util.ensure_dir(f)

    assert os.path.exists(d)
    assert os.path.isdir(d)

    with open(f) as fin:
        s = fin.read()
    print(s)
    assert s == 'test'


@timer
def test_base_output():
    """Test the base Output class.
    """
    # A bit gratuitous, since no one should ever call these, but just check that the
    # trivial implementation (or NotImplementedErrors) in the base class work as expected.
    config = { 'file_name' : 'dummy_file' }

    out = piff.Output()

    kwargs = out.parseKwargs(config)
    assert kwargs == config

    np.testing.assert_raises(NotImplementedError, out.write, None)
    np.testing.assert_raises(NotImplementedError, out.read)

@timer
def test_invalid():
    # Invalid output type
    config = { 'type': 'invalid' }
    with np.testing.assert_raises(ValueError):
        piff.Output.process(config)

    # Also check errors when registering other type names
    class NoOutput1(piff.Output):
        pass
    assert NoOutput1 not in piff.Output.valid_output_types.values()
    class NoOutput2(piff.Output):
        _type_name = None
    assert NoOutput2 not in piff.Output.valid_output_types.values()
    class ValidOutput1(piff.Output):
        _type_name = 'valid'
    assert ValidOutput1 in piff.Output.valid_output_types.values()
    assert ValidOutput1 == piff.Output.valid_output_types['valid']
    with np.testing.assert_raises(ValueError):
        class ValidOutput2(piff.Output):
            _type_name = 'valid'
    with np.testing.assert_raises(ValueError):
        class ValidOutput3(ValidOutput1):
            pass

if __name__ == '__main__':
    test_ensure_dir()
    test_base_output()
    test_invalid()
