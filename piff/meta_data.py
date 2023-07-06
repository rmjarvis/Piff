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

import os

piff_dir = os.path.split(os.path.realpath(__file__))[0]

if 'PIFF_DATA_DIR' in os.environ:
    data_dir = os.environ['PIFF_DATA_DIR']
else:
    data_dir = os.path.join(piff_dir, 'share')
