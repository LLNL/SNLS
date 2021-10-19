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

Solvers currently in the library:
  * `SNLSTrDlDenseG` Dogleg approximation to the trust-region sub-problem for multi-dimensional nonlinear systems of equations. This method reduces to a Newton-Raphson approach near the solution. These methods are sometimes described as "restricted step" approaches instead of trust-region approaches. See, for example, [Practical Methods of Optimization](https://doi.org/10.1002/9781118723203). As compared to general trust-region approaches in scalar minimization, the approach here for solving non-linear systems amounts to assuming that the Hessian matrix can be approximated using information from the Jacobian of the system.
  * `SNLSTrDlDenseG_Batch` a batch version of `SNLSTrDlDenseG` that is only available if the library is compiled with the `-DUSE_BATCH_SOLVERS=On` cmake option. It makes use of the RAJA Performance Suite to abstract away the complexity of memory management and execution strategies of forall loops. The `test/SNLS_batch_testdriver.cc` provides an example of how this solver, the memory manager, and forall abstraction layer can be used to solve a modified Broyden Tridiagonal Problem.
  * `NewtonBB` Simple 1D Newton solver with a fallback to bisection. If the zero of the function can be bounded then this solver can return a result even if the function is badly behaved. 

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

The principal devleoper of SNLS is Nathan Barton, nrbarton@llnl.gov. Brett Wayne and Robert Carson have also made contributions.

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
