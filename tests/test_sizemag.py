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

import numpy as np
import piff
import os

from piff_test_helper import timer

@timer
def test_sizemag_plot():
    """Check a size-magnitude plot.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_sizemag_plot.log')

    config = piff.config.read_config('sizemag.yaml')
    file_name = os.path.join('output', config['output']['stats'][0]['file_name'])

    # Some modifications to speed it up a bit.
    config['select'] = {
        'type': 'Properties',
        'where': '(CLASS_STAR > 0.9) & (MAG_AUTO < 13)',
        'hsm_size_reject': 4,
        'min_snr': 50,
    }
    config['psf']['interp'] = {'type': 'Mean'}
    config['psf']['outliers']['nsigma'] = 10
    del config['output']['stats'][1:]

    # Run via piffify
    piff.piffify(config, logger)
    assert os.path.isfile(file_name)

    # repeat with plotify function
    os.remove(file_name)
    piff.plotify(config, logger)
    assert os.path.isfile(file_name)


if __name__ == '__main__':
    test_sizemag_plot()
