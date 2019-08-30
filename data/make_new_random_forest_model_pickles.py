from __future__ import print_function, division
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import seaborn as sns
sns.set()
from scipy import stats
from astropy.io import fits
import fitsio

import os
import glob
import pickle

import lmfit
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import corner
import itertools
import galsim
import piff

import copy

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import scipy.stats as st
import scipy.spatial as sp
import math
from scipy.optimize import curve_fit

import lmfit
import galsim

import piff

import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


shapes_all_stars_list = []
param_values_all_stars_list = []

core_directory = os.path.realpath(__file__)
program_name = core_directory.split("/")[-1]
core_directory = core_directory.split("/{0}".format(program_name))[0]

exposures = glob.glob("{0}/data_for_random_forest/*".format(core_directory))


print("len(exposures): {0}".format(len(exposures)))

for exposure_i, exposure in enumerate(exposures):
    print("exposure_i: {0}".format(exposure_i))
    filename = "{0}/shapes_test_psf_optatmo_const_gpvonkarman.h5".format(exposure)       
    try:
        shapes = pd.read_hdf(filename)
    except:
        print("failed to read")
        continue

    model_e0 = np.array(shapes['model_e0'].tolist())
    number_of_stars = len(model_e0)
    model_e1 = np.array(shapes['model_e1'].tolist())
    model_e2 = np.array(shapes['model_e2'].tolist())
    model_zeta1 = np.array(shapes['model_zeta1'].tolist())
    model_zeta2 = np.array(shapes['model_zeta2'].tolist())
    model_delta1 = np.array(shapes['model_delta1'].tolist())
    model_delta2 = np.array(shapes['model_delta2'].tolist())

    shapes_all_stars = np.column_stack((model_e0, model_e1, model_e2, model_zeta1, model_zeta2, model_delta1, model_delta2))

    param_values_all_stars = np.ones((number_of_stars, 11))
    optatmo_params = np.array(["size", "g1", "g2", "z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11"])
    for optatmo_param_i, optatmo_param in enumerate(optatmo_params):
        if optatmo_param=="size":
            param_values_all_stars[:,0] = np.array(shapes['optics_size'].tolist()) + np.array(shapes['atmo_size'].tolist())
        elif optatmo_param=="g1":
            param_values_all_stars[:,1] = np.array(shapes['optics_g1'].tolist()) + np.array(shapes['atmo_g1'].tolist())  
        elif optatmo_param=="g2":
            param_values_all_stars[:,2] = np.array(shapes['optics_g2'].tolist()) + np.array(shapes['atmo_g2'].tolist()) 
        else:
            param_values_all_stars[:,optatmo_param_i] = np.array(shapes[optatmo_param].tolist())

    delete_list = []
    for index in range(0,len(model_e0)):
        if np.any(np.isnan(shapes_all_stars[index])) or np.any(np.isnan(param_values_all_stars[index])):
            delete_list.append(index)     
    shapes_all_stars = np.delete(shapes_all_stars, delete_list, axis=0)
    param_values_all_stars = np.delete(param_values_all_stars, delete_list, axis=0)
    
    model_moment_dictionary = {"model_e0":shapes_all_stars[:,0], "model_e1":shapes_all_stars[:,1], "model_e2":shapes_all_stars[:,2], "model_zeta1":shapes_all_stars[:,3], "model_zeta2":shapes_all_stars[:,4], "model_delta1":shapes_all_stars[:,5], "model_delta2":shapes_all_stars[:,6]}
    moment_med_dictionary = {"e0_med":None, "e1_med":None, "e2_med":None, "zeta1_med":None, "zeta2_med":None, "delta1_med":None, "delta2_med":None}
    moment_mad_dictionary = {"e0_mad":None, "e1_mad":None, "e2_mad":None, "zeta1_mad":None, "zeta2_mad":None, "delta1_mad":None, "delta2_mad":None}
    for m, moment in enumerate(["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"]):
        model_moment = model_moment_dictionary["model_{0}".format(moment)]
        moment_med = np.median(model_moment)
        moment_mad = np.median(np.abs(model_moment-moment_med))
        moment_med_dictionary["{0}_med".format(moment)] = moment_med
        moment_mad_dictionary["{0}_mad".format(moment)] = moment_mad      
    delete_array = np.zeros(len(shapes_all_stars[:,0]))
    for moment_i, moment in enumerate(["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"]):
        for index in range(0,len(shapes_all_stars[:,0])):
            if np.abs(model_moment_dictionary["model_{0}".format(moment)][index]-moment_med_dictionary["{0}_med".format(moment)]) > 6.0 * moment_mad_dictionary["{0}_mad".format(moment)]:
                if delete_array[index] == 0.0:
                    delete_array[index] = 1.0
    delete_list = []
    for d, delete_array_element in enumerate(delete_array):
        if delete_array_element == 1.0:
            delete_list.append(d)
    shapes_all_stars = np.delete(shapes_all_stars, delete_list, axis=0)
    param_values_all_stars = np.delete(param_values_all_stars, delete_list, axis=0)

    shapes_all_stars_list.append(shapes_all_stars)
    param_values_all_stars_list.append(param_values_all_stars)
shapes_all_stars = np.concatenate(shapes_all_stars_list, axis=0)
param_values_all_stars = np.concatenate(param_values_all_stars_list, axis=0)





for m, moment in enumerate(np.array(["e0", "e1", "e2", "zeta1", "zeta2", "delta1", "delta2"])):
    print("m: {0}".format(m))
    regr = RandomForestRegressor(n_estimators=10)
    X_train, X_test, y_train, y_test = train_test_split(param_values_all_stars, shapes_all_stars[:,m], test_size=0.25)
    regr.fit(X_train, y_train)
    y_test_predict = regr.predict(X_test)
    print('Test data: R^2 score = %.2f' % regr.score(X_test, y_test))
    plt.figure()
    plt.hist2d(y_test, y_test_predict, bins=40, cmin=1)
    plt.title("regr performance", fontsize=18)
    plt.xlabel("true {0}".format(moment), fontsize=18)
    plt.ylabel("predicted {0}".format(moment), fontsize=18)
    plt.colorbar()
    plt.plot([-1.0,1.0], [-1.0,1.0], color='red')    
    y_fraction_error = (y_test - y_test_predict) / y_test
    plt.savefig("{0}/{1}_1.png".format(core_directory,moment))
    plt.figure()
    plt.hist(y_test, bins=100)
    plt.savefig("{0}/{1}_2.png".format(core_directory,moment))
    plt.figure()
    plt.hist(y_test_predict, bins=100)
    plt.savefig("{0}/{1}_3.png".format(core_directory,moment)) 
    plt.figure()
    plt.hist(y_test - y_test_predict, bins=100)
    plt.savefig("{0}/{1}_4.png".format(core_directory,moment))  
    plt.figure()
    plt.hist(y_fraction_error, bins=100)
    plt.savefig("{0}/{1}_5.png".format(core_directory,moment))
    with open('{0}/random_forest_shapes_model_{1}.pickle'.format(core_directory, moment), 'wb') as f:
        pickle.dump(regr, f) 


