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

"""
.. module:: aimodels

Neural network architectures and training tools for the AIPSF model.

Everything in this package requires PyTorch, which is an optional dependency
of Piff.  To keep `import piff` working without torch, the submodules are
imported lazily on first attribute access (PEP 562).
"""

_model_attrs = ('Conv2dAutoEncoder', 'ResidualBlockConv', 'SpatialSoftmax',
                'save_checkpoint', 'load_autoencoder')
_train_attrs = ('PSFDataset', 'load_training_data', 'create_dataloaders',
                'train_autoencoder', 'train')

def __getattr__(name):
    if name in _model_attrs:
        submodule = 'models'
    elif name in _train_attrs:
        submodule = 'train'
    else:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))

    import importlib
    try:
        mod = importlib.import_module('.' + submodule, __name__)
    except ImportError as e:
        raise ImportError(
            "The piff.aimodels module (used by the AIPSF model) requires PyTorch, "
            "which could not be imported.  Install it with e.g. `pip install torch`.  "
            "Original error: %s" % e) from e
    return getattr(mod, name)

def __dir__():
    return sorted(list(globals().keys()) + list(_model_attrs) + list(_train_attrs))
