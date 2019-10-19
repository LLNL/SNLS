		SNLS -- small non-linear solver library

	 ________  ________   ___       ________      
	|\   ____\|\   ___  \|\  \     |\   ____\     
	\ \  \___|\ \  \\ \  \ \  \    \ \  \___|_    
	 \ \_____  \ \  \\ \  \ \  \    \ \_____  \   
	  \|____|\  \ \  \\ \  \ \  \____\|____|\  \  
	    ____\_\  \ \__\\ \__\ \_______\____\_\  \ 
	   |\_________\|__| \|__|\|_______|\_________\
	   \|_________|                   \|_________|
	                                              

BACKGROUND
======

SNLS (small non-linear solver) is a C++ library for solving non-linear systems of equations. The emphasis is on small systems of equations for which direct factorization of the Jacobian is appropriate. The systems of equations may be numerically ill-conditioned. Implementation is amenable to GPU usage, with the idea that class instantiation would occur within device functions.

Examples of work that has made use of SNLS or substantially equivalent algorithms:
  * [Journal publication](http://dx.doi.org/10.1063/1.4971654) on use with a porosity mechanics type model
  * [Journal publication](http://dx.doi.org/10.1088/0965-0393/17/3/035003) on use with a crystal mechanics type model

BUILDING
======

The build system is cmake-based.

Dependencies:
* blt -- required
  - https://github.com/LLNL/blt
  - in cmake invocation specify location with `-DBLT_SOURCE_DIR`

TESTING
======

Run cmake with `-DENABLE_TESTS=ON` and do `make test`

DEVELOPMENT
======

The develop branch is the main development branch for snls. Changes to develop are by pull request.

AUTHORS
======

The principal devleoper of SNLS is Nathan Barton, nrbarton@llnl.gov. Brett Wayne has also made contributions. 

CITATION
======

SNLS may be cited using the following `bibtex` entry:
```
@Misc{snls,
title = {SNLS},
author = {Wayne, Brett M. and Barton, Nathan R. and USDOE National Nuclear Security Administration},
abstractNote = {{SNLS} (small non-linear solver) is a C++ library for solving non-linear systems. The emphasis is on small systems of equations for which direct factorization of the Jacobian is appropriate. The systems of equations may be numerically ill-conditioned. Implementation is amenable to {GPU} usage, with the idea that class instantiation would occur within device functions.},
doi = {10.11578/dc.20181217.9},
year = 2018,
month = {9},
url = {https://github.com/LLNL/SNLS},
annote =    {
   https://www.osti.gov//servlets/purl/1487196
   https://www.osti.gov/biblio/1487196
}
}
```

LICENSE
======

License is under the BSD-3-Clause license. See [LICENSE](LICENSE) file for details. And see also the [NOTICE](NOTICE) file. 

`SPDX-License-Identifier: BSD-3-Clause`

``LLNL-CODE-761139``
