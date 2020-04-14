Writing the output file
=======================

Output handlers govern how the final solution is handled.  The typical thing to do is to
write to a FITS file, which is handled by the class :class:`~piff.OutputFile`.  This is 
the default, so if you are using that, you can omit the :type: specification.

The Output class
----------------

.. autoclass:: piff.Output
   :members:

.. autoclass:: piff.OutputFile
   :members:

