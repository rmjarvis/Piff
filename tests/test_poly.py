import numpy
import piff
from nose.tools import assert_raises

PolynomialsTypes = piff.polynomial_types.keys()



def test_poly_indexing():
    # Some indexing tests for a polynomial up to order 3
    N = 3
    interp = piff.Polynomial([N])

    # We expect there to be these coefficients:
    # x^0 y^0   1

    # x^0 y^1   2
    # x^1 y^0   3

    # x^0 y^2   4
    # x^1 y^1   5 
    # x^2 y^0   6

    # x^0 y^3   7
    # x^1 y^2   8
    # x^2 y^1   9
    # x^3 y^0   10 

    # Check that we have the indices we expect
    assert interp.indices[0] == [
        (0,0),
        (0,1), (1,0),
        (0,2), (1,1), (2,0),
        (0,3), (1,2), (2,1), (3,0)
    ]
    assert interp.nvariables[0]==10

    # check the packing then unpacking a 
    packed = numpy.random.uniform(size=interp.nvariables[0])
    unpacked = interp._unpack_coefficients(0,packed)
    packed_test = interp._pack_coefficients(0,unpacked)

    # Check that the shape is 4*4 in the unpacked (because we
    # want space for all the terms), and that we can unpack and
    # repack successfully.
    numpy.testing.assert_array_equal(packed,packed_test)
    assert unpacked.shape == (N+1,N+1)

    unpacked_test = numpy.zeros_like(unpacked)

    # check that we have zeros for the terms that should be zero in the matrix.
    # We don't want any terms with total exponent > N.
    # The variabled "unpacked" was created above by unpacking a random vector.
    # it should be zero where i+j>3 and have the random valus below that.
    for i in xrange(N+1):
        for j in xrange(N+1):
            if i+j>N:
                #Note we have two arrays, unpacked and unpacked_test
                assert unpacked[i,j]==0.0
                unpacked_test[i,j]=0.0

    # Now do the test the other way around, checking that 
    # we can pack and then unpack
    packed_test_2 = interp._pack_coefficients(0,unpacked_test)
    unpacked_test_2 = interp._unpack_coefficients(0,packed_test_2)
    numpy.testing.assert_array_equal(unpacked_test_2,unpacked_test)

def test_poly_mean():
    # Zero'th order polynomial fitting should be pretty trivial, just
    # the same as the mean fitting. So much of this code is just taken from
    # the mean testing in test_simple
    N = 0
    nparam = 5
    orders = [N for i in xrange(nparam)]
    interp = piff.Polynomial(orders)
    nstars = 100

    # Choose some random values of star parameters
    vectors = [ numpy.random.random(size=nparam) for i in range(nstars) ]

    # take the mean of them. Our curve fit should be able to reproduce this.
    mean = numpy.mean(vectors, axis=0)

    # Choose some random positions in the field.
    # NB think more about bounds and 
    pos = [ (numpy.random.random()*10, numpy.random.random()*10)
            for i in range(nstars) ]

    # Run our solver.
    interp.solve(pos, vectors)

    # we expect one set of coefficients per object
    assert len(interp.coeffs)==5

    # We should have very close values (not necessarily identical) since
    # we calculate these in numerically different ways.
    for mu, val in zip(mean, interp.coeffs):
        assert numpy.isclose(mu, val[0,0])

    # We also expect that if we interpolate to any point we just
    # get the mean as well
    for i in xrange(30):
        p=(numpy.random.random()*10, numpy.random.random()*10)
        v = interp.interpolate(p)
        numpy.testing.assert_almost_equal(v, mean)


def sub_poly_linear(type1):
    # Now lets do something more interesting - test a linear model.
    # with no noise this should fit really well, though again not
    # numerically perfectly.
    numpy.random.seed(12834)
    nparam = 3
    N = 1
    nstars=50   
    orders = [N for i in xrange(nparam)]
    interp = piff.Polynomial(orders, poly_type=type1)
    X = 10.0 # size of the field
    Y = 10.0

    pos = [ (numpy.random.random()*X, numpy.random.random()*Y)
            for i in range(nstars) ]


    # Let's make a function that is linear just as a function of one parameter
    # These are the linear fit parameters for each parameter in turn
    m1 = numpy.random.uniform(size=nparam)
    m2 = numpy.random.uniform(size=nparam)
    c = numpy.random.uniform(size=nparam)
    def f(pos):
        u = pos[0]
        v = pos[1]
        r = m1*u+m2*v+c
        return r

    # Simulate the vectors under this model
    vectors = [f(p) for p in pos]

    # Fit them. Linear fitting is quite easy so this should 
    # be okay
    interp.solve(pos, vectors)

    # Check that the interpolation recovers the desired function
    for i in xrange(30):
        p=(numpy.random.random()*X, numpy.random.random()*Y)
        numpy.testing.assert_almost_equal(f(p), interp.interpolate(p))

def test_poly_linear():
    for poly_type in PolynomialsTypes:
        sub_poly_linear(poly_type)

def sub_poly_quadratic(type1):
    # This is basically the same as linear but with
    # quadratic variation
    numpy.random.seed(1234)
    nparam = 3
    N = 2
    nstars=50
    orders = [N for i in xrange(nparam)]
    interp = piff.Polynomial(orders, poly_type=type1)
    X = 10.0 # size of the field
    Y = 10.0

    pos = [ (numpy.random.random()*X, numpy.random.random()*Y)
            for i in range(nstars) ]


    # Let's make a function that is linear just as a function of one parameter
    # These are the linear fit parameters for each parameter in turn
    m1 = numpy.random.uniform(size=nparam)
    m2 = numpy.random.uniform(size=nparam)
    q1 = numpy.random.uniform(size=nparam)
    c = numpy.random.uniform(size=nparam)
    def f(pos):
        u = pos[0]
        v = pos[1]
        r = q1*u*v+ m1*u+m2*v+c
        return r

    # Simulate the vectors under this model
    vectors = [f(p) for p in pos]

    # Fit them. Linear fitting is quite easy so this should 
    # be okay
    interp.solve(pos, vectors)

    # Check that the interpolation recovers the desired function
    for i in xrange(30):
        p=(numpy.random.random()*X, numpy.random.random()*Y)
        numpy.testing.assert_almost_equal(f(p), interp.interpolate(p))

def test_poly_quadratic():
    for poly_type in PolynomialsTypes:
        sub_poly_quadratic(poly_type)


def test_poly_guess():
    # test that our initial guess gives us a flat function given
    # by the mean
    numpy.random.seed(12434)
    N = 2
    X = 10.0
    Y = 10.0
    nstars=50
    nparam = 10
    orders = [N for i in xrange(nparam)]
    interp = piff.Polynomial(orders)
    pos = [ (numpy.random.random()*X, numpy.random.random()*Y)
            for i in range(nstars) ]

    for i in xrange(nparam):
        param = numpy.random.random(size=nstars)
        p0 = interp._initialGuess(pos, param, i)
        mu = param.mean()
        assert numpy.isclose(p0[0,0],mu)

        numpy.testing.assert_array_equal(p0[0,1:],0.0)
        numpy.testing.assert_array_equal(p0[1,0],0.0)
        numpy.testing.assert_array_equal(p0[1:,1:],0.0)
        numpy.testing.assert_almost_equal(interp._interpolationModel(pos, p0), mu)



def poly_load_save_sub(type1, type2):
    # Test that we can serialize and deserialize a polynomial 
    # interpolator correctly.  Copying all this stuff from above:

    numpy.random.seed(12434)
    nparam = 3
    nstars=50   
    # Use three different sizes to test everything
    orders = [1,2,3]
    interp = piff.Polynomial(orders, poly_type=type1)
    X = 10.0 # size of the field
    Y = 10.0

    pos = [ (numpy.random.random()*X, numpy.random.random()*Y)
            for i in range(nstars) ]

    # Let's make a function that is linear just as a function of one parameter
    # These are the linear fit parameters for each parameter in turn
    m1 = numpy.random.uniform(size=nparam)
    m2 = numpy.random.uniform(size=nparam)
    q1 = numpy.random.uniform(size=nparam)
    c = numpy.random.uniform(size=nparam)

    def f(pos):
        u = pos[0]
        v = pos[1]
        r = q1*u*v+ m1*u+m2*v+c
        return r

    # Simulate the vectors under this model
    vectors = [f(p) for p in pos]

    # Fit them!
    interp.solve(pos, vectors)

    # We should overwrite the order parameter when we load in
    interp2 = piff.Polynomial([0], poly_type=type2)


    import tempfile
    import os
    import fitsio
    extname = "interp"
    dirname = tempfile.mkdtemp()
    filename=os.path.join(dirname,'poly_test_file.fits')
    with fitsio.FITS(filename,'rw',clobber=False) as f:
        interp.writeSolution(f, extname=extname)
    f2 = fitsio.FITS(filename, "r")
    interp2.readSolution(f2, extname=extname)
    os.remove(filename)
    os.rmdir(dirname)


    # The type and other parameters should now have been overwritten and updated
    assert interp2.poly_type == interp.poly_type
    assert interp2.orders==interp.orders
    assert interp2.nvariables==interp.nvariables
    assert interp2.indices==interp.indices

    # Check that the old and new interpolators generate the same
    # value
    for i in xrange(30):
        p=(numpy.random.random()*X, numpy.random.random()*Y)
        numpy.testing.assert_almost_equal(interp.interpolate(p),interp2.interpolate(p))

def test_poly_raise():
    # Test that we can serialize and deserialize a polynomial 
    # interpolator correctly.  Copying all this stuff from above:

    numpy.random.seed(12434)
    nparam = 3
    nstars = 50

    # Use three different sizes to test everything
    orders = [1,2,3]
    interp = piff.Polynomial(orders)
    pos = [ (numpy.random.random()*10, numpy.random.random()*10)
            for i in range(nstars) ]
    #use the wrong number of parameters here so that we raise an error
    vectors = [ numpy.random.random(size=nparam+1) for i in range(nstars) ]
    assert_raises(ValueError, interp.solve, pos, vectors)



def test_poly_load_save():
    for poly_type in PolynomialsTypes:
        poly_load_save_sub(poly_type,poly_type)

def test_poly_load_err():
    for poly_type1 in PolynomialsTypes[:]:
        for poly_type2 in PolynomialsTypes[:]:
            if poly_type1!=poly_type2:
                poly_load_save_sub(poly_type1,poly_type2)

if __name__ == '__main__':
    test_poly_indexing()
    test_poly_mean()
    test_poly_linear()
    test_poly_quadratic()
    test_poly_guess()
    test_poly_raise()
    test_poly_load_save()
    test_poly_load_err()
