import sys,os,glob,re

try:
    from setuptools import setup, Extension, find_packages
    from setuptools.command.build_ext import build_ext
    from setuptools.command.install_scripts import install_scripts
    from setuptools.command.easy_install import easy_install
    import setuptools
    print("Using setuptools version",setuptools.__version__)
except ImportError:
    print('Unable to import setuptools.  Using distutils instead.')
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext
    from distutils.command.install_scripts import install_scripts
    easy_install = object  # Prevent error when using as base class
    import distutils
    # cf. http://stackoverflow.com/questions/1612733/including-non-python-files-with-setup-py
    from distutils.command.install import INSTALL_SCHEMES
    for scheme in INSTALL_SCHEMES.values():
        scheme['data'] = scheme['purelib']
    # cf. http://stackoverflow.com/questions/37350816/whats-distutils-equivalent-of-setuptools-find-packages-python
    from distutils.util import convert_path
    def find_packages(base_path='.'):
        base_path = convert_path(base_path)
        found = []
        for root, dirs, files in os.walk(base_path, followlinks=True):
            dirs[:] = [d for d in dirs if d[0] != '.' and d not in ('ez_setup', '__pycache__')]
            relpath = os.path.relpath(root, base_path)
            parent = relpath.replace(os.sep, '.').lstrip('.')
            if relpath != '.' and parent not in found:
                # foo.bar package but no foo package, skip
                continue
            for dir in dirs:
                if os.path.isfile(os.path.join(root, dir, '__init__.py')):
                    package = '.'.join((parent, dir)) if parent else dir
                    found.append(package)
        return found
    print("Using distutils version",distutils.__version__)

from distutils.command.install_headers import install_headers
from pybind11.setup_helpers import Pybind11Extension

try:
    from sysconfig import get_config_vars
except:
    from distutils.sysconfig import get_config_vars

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

scripts = ['piffify', 'plotify', 'meanify']
scripts = [ os.path.join('scripts',f) for f in scripts ]
shared_data = glob.glob('share/*')

undef_macros = []

packages = find_packages()
print('packages = ',packages)

# If we build with debug, also undefine NDEBUG flag
if "--debug" in sys.argv:
    undef_macros+=['NDEBUG']

copt =  {
    'gcc' : ['-fopenmp','-O3','-ffast-math'],
    'icc' : ['-openmp','-O3'],
    'clang' : ['-O3','-ffast-math'],
    'clang w/ OpenMP' : ['-fopenmp=libomp','-O3','-ffast-math'],
    'unknown' : [],
}
lopt =  {
    'gcc' : ['-fopenmp'],
    'icc' : ['-openmp'],
    'clang' : [],
    'clang w/ OpenMP' : ['-fopenmp=libomp'],
    'unknown' : [],
}

if "--debug" in sys.argv:
    copt['gcc'].append('-g')
    copt['icc'].append('-g')
    copt['clang'].append('-g')
    copt['clang w/ OpenMP'].append('-g')

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

def get_compiler(cc):
    """Try to figure out which kind of compiler this really is.
    In particular, try to distinguish between clang and gcc, either of which may
    be called cc or gcc.
    """
    cmd = [cc,'--version']
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    print('compiler version information: ')
    for line in lines:
        print(line.strip())
    line = lines[0].decode(encoding='UTF-8')
    if line.startswith('Configured'):
        line = lines[1].decode(encoding='UTF-8')

    if "clang" in line:
        # clang 3.7 is the first with openmp support. So check the version number.
        # It can show up in one of two places depending on whether this is Apple clang or
        # regular clang.
        import re
        if 'LLVM' in line:
            match = re.search(r'LLVM [0-9]+(\.[0-9]+)+', line)
            match_num = 1
        else:
            match = re.search(r'[0-9]+(\.[0-9]+)+', line)
            match_num = 0
        if match:
            version = match.group(0).split()[match_num]
            print('clang version = ',version)
            # Get the version up to the first decimal
            # e.g. for 3.4.1 we only keep 3.4
            vnum = version[0:version.find('.')+2]
            if vnum >= '3.7':
                return 'clang w/ OpenMP'
        return 'clang'
    elif 'gcc' in line:
        return 'gcc'
    elif 'GCC' in line:
        return 'gcc'
    elif 'clang' in cc:
        return 'clang'
    elif 'gcc' in cc or 'g++' in cc:
        return 'gcc'
    elif 'icc' in cc or 'icpc' in cc:
        return 'icc'
    else:
        # OK, the main thing we need to know is what openmp flag we need for this compiler,
        # so let's just try the various options and see what works.  Don't try icc, since
        # the -openmp flag there gets treated as '-o penmp' by gcc and clang, which is bad.
        # Plus, icc should be detected correctly by the above procedure anyway.
        for cc_type in ['gcc', 'clang']:
            if try_cc(cc, cc_type):
                return cc_type
        # I guess none of them worked.  Now we really do have to bail.
        return 'unknown'

def try_cc(cc, cc_type):
    """
    If cc --version is not helpful, the last resort is to try each compiler type and see
    if it works.
    """
    cpp_code = """
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include "omp.h"
#endif

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int main() {
    int n = 500;
    std::vector<double> x(n,0.);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<n; ++i) x[i] = 2*i+1;

    double sum = 0.;
    for (int i=0; i<n; ++i) sum += x[i];
    // Sum should be n^2 = 250000

    std::cout<<get_max_threads()<<"  "<<sum<<std::endl;
    return 0;
}
"""
    import tempfile
    cpp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cpp')
    cpp_file.write(cpp_code)
    cpp_file.close()

    # Just get a named temporary file to write to:
    o_file = tempfile.NamedTemporaryFile(delete=False, suffix='.os')
    o_file.close()

    # Try compiling with the given flags
    import subprocess
    cmd = [cc] + copt[cc_type] + ['-c',cpp_file.name,'-o',o_file.name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    p.communicate()
    if p.returncode != 0:
        os.remove(cpp_file.name)
        if os.path.exists(o_file.name): os.remove(o_file.name)
        return False

    # Another named temporary file for the executable
    exe_file = tempfile.NamedTemporaryFile(delete=False, suffix='.exe')
    exe_file.close()

    # Try linking
    cmd = [cc] + lopt[cc_type] + [o_file.name,'-o',exe_file.name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = p.stdout.readlines()
    p.communicate()

    if p.returncode and cc == 'cc':
        # The linker needs to be a c++ linker, which isn't 'cc'.  However, I couldn't figure
        # out how to get setup.py to tell me the actual command to use for linking.  All the
        # executables available from build_ext.compiler.executables are 'cc', not 'c++'.
        # I think this must be related to the bugs about not handling c++ correctly.
        #    http://bugs.python.org/issue9031
        #    http://bugs.python.org/issue1222585
        # So just switch it manually and see if that works.
        cmd = ['c++'] + lopt[cc_type] + [o_file.name,'-o',exe_file.name]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        p.communicate()

    # Remove the temp files
    os.remove(cpp_file.name)
    os.remove(o_file.name)
    if os.path.exists(exe_file.name): os.remove(exe_file.name)
    return p.returncode == 0


# This was supposed to remove the -Wstrict-prototypes flag
# But it doesn't work....
# Hopefully they'll fix this bug soon:
#  http://bugs.python.org/issue9031
#  http://bugs.python.org/issue1222585
#(opt,) = get_config_vars('OPT')
#os.environ['OPT'] = " ".join( flag for flag in opt.split() if flag != '-Wstrict-prototypes')

# Make a subclass of build_ext so we can do different things depending on which compiler we have.
# In particular, we may want to use different compiler options for OpenMP in each case.
# cf. http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
# We're not currently using OpenMP, but with this bit we'll be ready.
class my_builder( build_ext ):
    def build_extensions(self):
        # Figure out what compiler it will use
        cc = self.compiler.executables['compiler_cxx'][0]
        comp_type = get_compiler(cc)
        if cc == comp_type:
            print('Using compiler %s'%(cc))
        else:
            print('Using compiler %s, which is %s'%(cc,comp_type))
        # Add the appropriate extra flags for that compiler.
        eigen_path = find_eigen_dir(output=False)
        for e in self.extensions:
            e.extra_compile_args = copt[ comp_type ]
            e.extra_link_args = lopt[ comp_type ]
            e.include_dirs = ['include']
            e.include_dirs.append(eigen_path)
        # Now run the normal build function.
        build_ext.build_extensions(self)

# AFAICT, setuptools doesn't provide any easy access to the final installation location of the
# executable scripts.  This bit is just to save the value of script_dir so I can use it later.
# cf. http://stackoverflow.com/questions/12975540/correct-way-to-find-scripts-directory-from-setup-py-in-python-distutils/
class my_easy_install( easy_install ):  # For setuptools

    # Match the call signature of the easy_install version.
    def write_script(self, script_name, contents, mode="t", *ignored):
        # Run the normal version
        easy_install.write_script(self, script_name, contents, mode, *ignored)
        # Save the script install directory in the distribution object.
        # This is the same thing that is returned by the setup function.
        self.distribution.script_install_dir = self.script_dir

# For distutils, the appropriate thing is the install_scripts command class, not easy_install.
# So here is the appropriate thing in that case.
class my_install_scripts( install_scripts ):  # For distutils
    def run(self):
        install_scripts.run(self)
        self.distribution.script_install_dir = self.install_dir

dependencies = ['galsim>=2.3', 'numpy>=1.17', 'scipy>=1.2', 'pyyaml>=5.1', 'treecorr>=4.3.1', 'fitsio>=1.0', 'matplotlib>=3.3', 'LSSTDESC.Coord>=1.0', 'treegp>=0.6', 'threadpoolctl>=3.1']

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

basis_cpp_mod = Pybind11Extension("piff/basic_solver", ["src/basic_solver.cpp"])

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
      install_requires=dependencies,
      ext_modules=[basis_cpp_mod],
      cmdclass = {'build_ext': my_builder,
                  'install_scripts': my_install_scripts,
                  'easy_install': my_easy_install,
                  },
      scripts=scripts
      )

# Check that the path includes the directory where the scripts are installed.
# NB. If not running install, then script_install_dir won't be there...
real_env_path = [os.path.realpath(d) for d in os.environ['PATH'].split(':')]
if (hasattr(dist,'script_install_dir') and
    dist.script_install_dir not in os.environ['PATH'].split(':') and
    os.path.realpath(dist.script_install_dir) not in real_env_path):

    print('\nWARNING: The Piff executables were installed in a directory not in your PATH')
    print('         If you want to use the executables, you should add the directory')
    print('\n             ',dist.script_install_dir,'\n')
    print('         to your path.  The current path is')
    print('\n             ',os.environ['PATH'],'\n')
    print('         Alternatively, you can specify a different prefix with --prefix=PREFIX,')
    print('         in which case the scripts will be installed in PREFIX/bin.')
    print('         If you are installing via pip use --install-option="--prefix=PREFIX"')
