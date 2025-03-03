import sys,os,glob,re
import shutil
import urllib.request as urllib2
import tarfile

from setuptools import setup, find_packages
import setuptools
print("Using setuptools version",setuptools.__version__)

from pybind11.setup_helpers import Pybind11Extension, build_ext

try:
    from sysconfig import get_config_vars
except:
    from distutils.sysconfig import get_config_vars

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

scripts = ['piffify', 'plotify', 'meanify']
scripts = [ os.path.join('scripts',f) for f in scripts ]
shared_data = glob.glob('share/*')

packages = find_packages()
print('packages = ',packages)

# Check for Eigen in some likely places
def find_eigen_dir(output=False):

    import distutils.sysconfig

    try_dirs = []

    # Start with a user-specified directory.
    if 'EIGEN_DIR' in os.environ:
        try_dirs.append(os.environ['EIGEN_DIR'])
        try_dirs.append(os.path.join(os.environ['EIGEN_DIR'], 'include'))

    # Add the python system include directory.
    try_dirs.append(distutils.sysconfig.get_config_var('INCLUDEDIR'))

    # If using Anaconda, add their lib dir in case fftw is installed there.
    # (With envs, this might be different than the sysconfig LIBDIR.)
    if 'CONDA_PREFIX' in os.environ:
        try_dirs.append(os.path.join(os.environ['CONDA_PREFIX'],'lib'))

    # Some standard install locations:
    try_dirs.extend(['/usr/local/include', '/usr/include'])
    if sys.platform == "darwin":
        try_dirs.extend(['/sw/include', '/opt/local/include'])

    # Also if there is a C_INCLUDE_PATH, check those dirs.
    for path in ['C_INCLUDE_PATH']:
        if path in os.environ:
            for dir in os.environ[path].split(':'):
                try_dirs.append(dir)

    # Finally, (last resort) check our own download of eigen.
    if os.path.isdir('downloaded_eigen'):
        try_dirs.extend(glob.glob(os.path.join('downloaded_eigen','*')))

    if output: print("Looking for Eigen:")
    for dir in try_dirs:
        if dir is None: continue
        if not os.path.isdir(dir): continue
        if output: print("  ", dir, end='')
        if os.path.isfile(os.path.join(dir, 'Eigen/Core')):
            if output: print("  (yes)")
            return dir
        if os.path.isfile(os.path.join(dir, 'eigen3', 'Eigen/Core')):
            dir = os.path.join(dir, 'eigen3')
            if output:
                # Only print this if the eigen3 addition was key to finding it.
                print("\n  ", dir, "  (yes)")
            return dir
        if output: print("  (no)")

    if output:
        print("Could not find Eigen in any of the standard locations.")
        print("Will now try to download it from gitlab.com. This requires an internet")
        print("connection, so it will fail if you are currently offline.")
        print("If Eigen is installed in a non-standard location, and you want to use that")
        print("instead, you should make sure the right directory is either in your")
        print("C_INCLUDE_PATH or specified in an EIGEN_DIR environment variable.")

    try:
        dir = 'downloaded_eigen'
        if os.path.isdir(dir):
            # If this exists, it was tried above and failed.  Something must be wrong with it.
            print("Previous attempt to download eigen found. Deleting and trying again.")
            shutil.rmtree(dir)
        os.mkdir(dir)
        url = 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2'
        if output:
            print("Downloading eigen from ",url)
        # Unfortunately, gitlab doesn't allow direct downloads. We need to spoof the request
        # so it thinks we're a web browser.
        # cf. https://stackoverflow.com/questions/42863240/how-to-get-round-the-http-error-403-forbidden-with-urllib-request-using-python
        page=urllib2.Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        data=urllib2.urlopen(page).read()
        fname = 'eigen.tar.bz2'
        with open(fname, 'wb') as f:
            f.write(data)
        if output:
            print("Downloaded %s.  Unpacking tarball."%fname)
        with tarfile.open(fname) as tar:

            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                # Avoid security vulnerability in tar.extractall function.
                # This bit of code was added by the Advanced Research Center at Trellix in PR #1188.
                # For more information about the security vulnerability, see
                # https://github.com/advisories/GHSA-gw9q-c7gh-j9vm

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, dir)
        os.remove(fname)
        # This actually extracts into a subdirectory with a name eigen-eigen-5a0156e40feb/
        # I'm not sure if that name is reliable, so use glob to get it.
        dir = glob.glob(os.path.join(dir,'*'))[0]
        if os.path.isfile(os.path.join(dir, 'Eigen/Core')):
            return dir
        elif output:
            print("Downloaded eigen, but it didn't have the expected Eigen/Core file.")
    except Exception as e:
        if output:
            print("Error encountered while downloading Eigen from the internet")
            print(e)

    raise OSError("Could not find Eigen")

def find_pybind_path():

    # Finally, add pybind11's include dir
    import pybind11
    import os
    print('PyBind11 is version ',pybind11.__version__)
    print('Looking for pybind11 header files: ')
    locations = [pybind11.get_include(user=True),
                 pybind11.get_include(user=False),
                 '/usr/include',
                 '/usr/local/include',
                 None]
    for try_dir in locations:
        if try_dir is None:
            # Last time through, raise an error.
            print("Could not find pybind11 header files.")
            print("They should have been in one of the following locations:")
            for l in locations:
                if l is not None:
                    print("   ", l)
            raise OSError("Could not find PyBind11")
        print('  ',try_dir,end='')
        if os.path.isfile(os.path.join(try_dir, 'pybind11/pybind11.h')):
            print('  (yes)')
            # builder.include_dirs.append(try_dir)
            pybind_path = try_dir # os.path.join(try_dir, 'pybind11')
            break
        else:
            raise OSError("Could not find PyBind11")
            print('  (no)')

    return pybind_path

build_dep = ['setuptools>=38', 'numpy>=1.17', 'pybind11>=2.2']
run_dep = ['galsim>=2.4', 'numpy>=1.17', 'scipy>=1.2', 'pyyaml>=5.1', 'treecorr>=4.3.1', 'fitsio>=1.0', 'matplotlib>=3.6', 'LSSTDESC.Coord>=1.0', 'treegp>=1.3', 'threadpoolctl>=3.1']

with open('README.rst') as file:
    long_description = file.read()

# Read in the piff version from piff/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('piff','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    piff_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('Piff version is %s'%(piff_version))

sources = glob.glob(os.path.join('src','*.cpp'))

ext = Pybind11Extension(
        "piff._piff",
        sources,
        include_dirs=[find_eigen_dir(output=True), find_pybind_path()]
      )

dist = setup(name="Piff",
      version=piff_version,
      author="Mike Jarvis",
      author_email="michael@jarvis.net",
      description="PSFs in the Full FOV",
      long_description=long_description,
      license = "BSD License",
      url="https://github.com/rmjarvis/Piff",
      download_url="https://github.com/rmjarvis/Piff/releases/tag/v%s.zip"%piff_version,
      packages=packages,
      package_data={'piff' : shared_data},
      setup_requires=build_dep,
      install_requires=run_dep,
      ext_modules=[ext],
      cmdclass = {'build_ext': build_ext},
      scripts=scripts
      )
