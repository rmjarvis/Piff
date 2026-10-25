Selecting Good PSF Stars
========================

It is not uncommon to want to select only a subset of the objects in an input catalog
to use for PSF estimation.

* Even if your input catalog is nominally all real stars, you may want to exclude stars that have nearby neighbors or are too fain.
* More realistically, some of the input "stars" may not in fact be stars.  There may be a few quasars or AGN, which are not precisely point-like, in the catalog.
* And finally, you may only have an input catalog of all detections, including both stars and extended objects, and you want Piff to pick out the stars for you.

All of these types of selection can be effected by the ``select`` field in the config file.

The class used for selection is specified by a ``type`` item as usual.  Options currently
include:

* "Flag" selects stars according some kind of flags column in the input catalog. See `FlagSelect`.
* "Properties" selects stars according to any columns that are identified as "properties" to read in for each obect.  See `PropertiesSelect`.
* "SizeMag" selects stars by looking for a stellar locus in a size-magnitude diagram.  See `SizeMagSelect`.
* "SmallBright" selects stars that are small and bright.  This isn't typically very good on its own, but it can be a decent seed selection for the "SizeMag" selection.  See `SmallBrightSelect`.

Finally, all of the selection types allow additional rejection steps after the initial selection.
These are described in `Select`.

.. autoclass:: piff.Select
   :members:

.. autoclass:: piff.FlagSelect
   :members:

.. autoclass:: piff.PropertiesSelect
   :members:

.. autoclass:: piff.SizeMagSelect
   :members:

.. autoclass:: piff.SmallBrightSelect
   :members:
