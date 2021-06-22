import numpy as np
import pickle
import galsim
from .star import Star, StarData
from sklearn import neighbors
from scipy.interpolate import Rbf
from scipy.interpolate import RegularGridInterpolator
import fitsio

def convert_zernikes_des(a_fp):
    """ This method converts an array of Noll Zernike coefficients from
    Focal Plane coordinates to a set suitable for u,v Sky coordinates. Assumes
    that the WCS is of form [[0,-1],[-1,0]].  See AR's notebook calculate-Donut-to-GalsimPiff-Zernike-Coordinate-conversion
    for the calculation of this conversion.

    param: a_fp       An ndarray or list with Zernike's in Focal Plane coordinates
    return: a_sky      An ndarray or list with Zernike's in u,v Sky coordinates
    """

    shape_in = a_fp.shape[0]

    # fill tmp array with dimensions to 37
    a_fp_tmp = np.zeros(37+1)
    a_fp_tmp[0:shape_in] = a_fp

    a_sky = np.zeros(37+1)

    a_sky[1] = a_fp_tmp[1]
    a_sky[2] = -a_fp_tmp[3]
    a_sky[3] = -a_fp_tmp[2]
    a_sky[4] = a_fp_tmp[4]
    a_sky[5] = a_fp_tmp[5]
    a_sky[6] = -a_fp_tmp[6]
    a_sky[7] = -a_fp_tmp[8]
    a_sky[8] = -a_fp_tmp[7]
    a_sky[9] = a_fp_tmp[10]
    a_sky[10] = a_fp_tmp[9]
    a_sky[11] = a_fp_tmp[11]
    a_sky[12] = -a_fp_tmp[12]
    a_sky[13] = a_fp_tmp[13]
    a_sky[14] = a_fp_tmp[14]
    a_sky[15] = -a_fp_tmp[15]
    a_sky[16] = -a_fp_tmp[17]
    a_sky[17] = -a_fp_tmp[16]
    a_sky[18] = a_fp_tmp[19]
    a_sky[19] = a_fp_tmp[18]
    a_sky[20] = -a_fp_tmp[21]
    a_sky[21] = -a_fp_tmp[20]
    a_sky[22] = a_fp_tmp[22]
    a_sky[23] = a_fp_tmp[23]
    a_sky[24] = -a_fp_tmp[24]
    a_sky[25] = -a_fp_tmp[25]
    a_sky[26] = a_fp_tmp[26]
    a_sky[27] = a_fp_tmp[27]
    a_sky[28] = -a_fp_tmp[28]
    a_sky[29] = -a_fp_tmp[30]
    a_sky[30] = -a_fp_tmp[29]
    a_sky[31] = a_fp_tmp[32]
    a_sky[32] = a_fp_tmp[31]
    a_sky[33] = -a_fp_tmp[34]
    a_sky[34] = -a_fp_tmp[33]
    a_sky[35] = a_fp_tmp[36]
    a_sky[36] = a_fp_tmp[35]
    a_sky[37] = a_fp_tmp[37]

    return a_sky[0:shape_in]

class Interper(object):
    """ Class to provide a common interface to scipy or sklearn interpolation methods
    """
    def __init__(self,type,X,Z,logger=None,kwargs=None):
        """ Create and train Interpolation

        :param type:         Method, options are 'Rbf','Rgi' and 'Knn'
        :param X:            An ndarray with Locations [points,2]
        :param Z:            An ndarray with Targets [points]
        :param logger:       A logger object for logging debug info. [default: None]
        :param kwargs:       Other options passed to interpolator. [default: None]
        """

        logger = galsim.config.LoggerWrapper(logger)
        self.type = type

        if type=='Rbf':
            self.interp = Rbf(X[:,0], X[:,1], Z)
        elif type=='Rgi':
            # RegularGridInterpolator just wants individual arrays of the unique
            # grid locations in x,y.  Then it assumes that Z is ordered ala np.meshgrid(x,y,indexing='ij')
            xall = X[:,0]
            yall = X[:,1]
            x = np.unique(xall)
            y = np.unique(yall)
            nx = len(x)
            ny = len(y)

            # sort original x,y arrays to find index so that Z will have needed ordering
            index = np.lexsort((yall,xall))
            ZZ = Z[index]
            ZZZ = ZZ.reshape(nx,ny)
            self.interp = RegularGridInterpolator( (x, y) , ZZZ )
        elif type=='Knn':
            knn_kwargs = {'n_neighbors':10,'weights':'distance'}
            knn_kwargs.update(kwargs)
            self.interp = neighbors.KNeighborsRegressor(**knn_kwargs)
            self.interp.fit(X,Z)
        else:
            raise ValueError('Interper type {0} not defined'.format(type))

    def predict(self,X):
        """ Interpolate to locations X

        param: X            An ndarray with Locations [points,2]
        return: Z           An ndarray with predictions at X
        """
        if self.type=='Rbf':
            Z = self.interp(X[:,0],X[:,1])
        elif self.type=='Rgi':
            Z = self.interp((X[:,0],X[:,1]))
        elif self.type=='Knn':
            Z = self.interp.predict(X)
        else:
            raise ValueError('Interper type {0} not defined'.format(self.type))

        return Z


class Wavefront(object):
    """ This class reads in wavefront data and assigns Zernike coefficients
        to Stars by interpolation.

    """
    def __init__(self,wavefront_kwargs,logger=None):
        """ Parse the input options

        param: wavefront_kwargs    A dictionaries holding the options for each
                                   source of Zernike Coefficients.  Multiple input files are allowed,
                                   with Dictionaries keyed by 'source1','source2'...
                                   Each 'sourceN' dictionary has keys: 'file','ext','zlist','keys','type','chip'.
                                   The key 'survey' applies custom code for the desired survey.
        param: logger              A logger object for logging debug info. [default: None]
        """

        logger = galsim.config.LoggerWrapper(logger)
        self.maxnZ = 37            # hardcoded maximum Zernike index, Noll parameterization
        zformatstr = "z%d"         # hardcoded format string for Zernike coefficients
        self.interp_objects = {}   # store all interpolation objects

        # Custom code for specific surveys
        if 'survey' in wavefront_kwargs:
            self.survey = wavefront_kwargs['survey']
        else:
            self.survey = None

        # fill lists for each wavefront source
        wf_dicts = []
        for key in wavefront_kwargs.keys():
            if key[0:6] == 'source':
                wf_dicts.append(wavefront_kwargs[key])
        self.nsources = len(wf_dicts)
        self.starxykeys = []
        self.chipkeys = []
        self.zlists = []
        self.chiplists = []

        # loop over input sources
        for isource,kwargs in enumerate(wf_dicts):

            # table with data
            table = fitsio.read(kwargs['file'],kwargs['ext'])
            keylist = list(kwargs['keys'].keys())
            xkey = keylist[0]
            ykey = keylist[1]

            # store keys into Star object for this file
            self.starxykeys.append([kwargs['keys'][xkey],kwargs['keys'][ykey]])

            # store list of Zernike coefficients for this file
            self.zlists.append(kwargs['zlist'])

            # if we want to interpolate Chip by Chip, do so here
            if kwargs['chip'] != 'None':

                # column name in .fits file
                chipkey = list(kwargs['chip'].keys())[0]

                # store chip key into Star object for this file
                self.chipkeys.append(kwargs['chip'][chipkey])

                # get chipnum from .fits source file
                chipvec = table[chipkey]
                chips = np.unique(chipvec).tolist()   # get all chips present here
                self.chiplists.append(chips)

                for achip in chips:
                    ok = (table[chipkey]==achip)
                    table_onechip = table[ok]
                    xvec = table_onechip[kwargs['keys'][xkey]]
                    yvec = table_onechip[kwargs['keys'][ykey]]
                    X = np.column_stack([xvec,yvec])

                    for iZ in kwargs['zlist']:
                        zname = zformatstr % (iZ)
                        Z = table_onechip[zname]

                        # make interpolation object for each desired Zernike coefficient
                        # with locations,targets
                        self.interp_objects[(isource,achip,iZ)] = Interper(kwargs['type'],X,Z)

            else:
                xvec = table[kwargs['keys'][xkey]]
                yvec = table[kwargs['keys'][ykey]]
                X = np.column_stack([xvec,yvec])

                # store None in chipkeys
                self.chipkeys.append(None)
                self.chiplists.append([])

                # interpolate for each desired Zernike coefficient
                for iZ in kwargs['zlist']:
                    zname = zformatstr % (iZ)
                    Z = table[zname]

                    # make interpolation object for each desired Zernike coefficient
                    # with locations,targets
                    self.interp_objects[(isource,None,iZ)] = Interper(kwargs['type'],X,Z)

        return

    def fillWavefront(self,star_list,logger=None):
        """ Interpolate wavefront to each star's location, fill wavefront key of star.data.properties

        :param star_list:         A list of Star instances
        :param logger:            A logger object for logging debug info. [default: None]
        """

        nstars = len(star_list)
        wf_arr = np.zeros((nstars,self.maxnZ+1))

        # for DES fill Focal plane position keys 'x_fp' and 'y_fp'
        if self.survey == 'des':
            from .des import decaminfo
            dinfo = decaminfo.DECamInfo()
            ix_arr = np.array([aStar.x for aStar in star_list])
            iy_arr = np.array([aStar.y for aStar in star_list])
            chipnum_arr = np.array([aStar['chipnum'] for aStar in star_list])
            x_fp,y_fp = dinfo.getPosition(chipnum_arr, ix_arr, iy_arr)

        # loop over wavefront sources
        for isource in range(self.nsources):

            logger.debug("Filling Zernike coefficients for source %d" % (isource))
            xkey,ykey = self.starxykeys[isource]

            # get x,y from Star.data, or use x_fp,y_fp for DES, since we didn't remake the Stars above...
            if self.survey == 'des':
                Xs = np.column_stack([x_fp,y_fp])
            else:
                xvals = np.array([aStar.data[xkey] for aStar in star_list])
                yvals = np.array([aStar.data[ykey] for aStar in star_list])
                Xs = np.column_stack([xvals,yvals])

            # if we want to interpolate Chip by Chip, get the chipnums
            if self.chipkeys[isource]:
                chipnums = np.array([aStar.data[self.chipkeys[isource]] for aStar in star_list])

            # loop over Zernike coefficients from this source
            for iZ in self.zlists[isource]:

                # if we want to interpolate Chip by Chip, do so here
                if self.chipkeys[isource]:

                    for achip in self.chiplists[isource]:
                        ok = (chipnums == achip)   # index of Stars with this chipnum
                        Xs_chip = Xs[ok,:]
                        ys_chip = self.interp_objects[(isource,achip,iZ)].predict(Xs_chip)
                        wf_arr[ok,iZ] = ys_chip

                else:
                    ys = self.interp_objects[(isource,None,iZ)].predict(Xs)
                    wf_arr[:,iZ] = ys

        # convert Zernike coeff for DES
        logger.debug("Converting Zernike coefficients for Survey %s" % (self.survey))
        if self.survey == 'des':
            wf_arr_final = np.zeros_like(wf_arr)
            for i in range(wf_arr_final.shape[0]):
                wf_arr_final[i,:] = convert_zernikes_des(wf_arr[i,:])
        else:
            wf_arr_final = wf_arr

        # convert from 700nm standard to wavelength used in yaml which is now 782.1
        # TODO: make this not hardcoded, get the 700 from psf.wavefront yaml and the 782.1 from the psf.model
        wf_arr_final = (700.0/782.1) * wf_arr_final

        # insert wavefront data into new Star's data.properties
        new_stars = []
        for istar,aStar in enumerate(star_list):
            new_data = StarData(image=aStar.image,
                        image_pos=aStar.image_pos,
                        field_pos=aStar.field_pos,
                        weight=aStar.weight,
                        pointing=aStar.data.pointing,
                        properties=dict(aStar.data.properties, wavefront=wf_arr_final[istar,:]),_xyuv_set=True)
            new_star = Star(data=new_data,fit=aStar.fit,extrafit=aStar.extrafit)
            new_stars.append(new_star)

        return new_stars
