		SNLS -- small non-linear solver library

	 ________  ________   ___       ________      
	|\   ____\|\   ___  \|\  \     |\   ____\     
	\ \  \___|\ \  \\ \  \ \  \    \ \  \___|_    
	 \ \_____  \ \  \\ \  \ \  \    \ \_____  \   
	  \|____|\  \ \  \\ \  \ \  \____\|____|\  \  
	    ____\_\  \ \__\\ \__\ \_______\____\_\  \ 
	   |\_________\|__| \|__|\|_______|\_________\
	   \|_________|                   \|_________|
	                                              

# BACKGROUND

SNLS (small non-linear solver) is a C++ library for solving non-linear systems of equations. The emphasis is on small systems of equations for which direct factorization of the Jacobian is appropriate. The systems of equations may be numerically ill-conditioned. Implementation is amenable to GPU usage, with the idea that class instantiation would occur within device functions for non-batch type solvers.

Examples of work that has made use of SNLS or substantially equivalent algorithms:
  * [Journal publication](http://dx.doi.org/10.1063/1.4971654) on use with a porosity mechanics type model
  * [Journal publication](http://dx.doi.org/10.1088/0965-0393/17/3/035003) on use with a crystal mechanics type model

# SOLVERS

  * `SNLSTrDlDenseG` Dogleg approximation to the trust-region sub-problem for multi-dimensional nonlinear systems of equations. This method reduces to a Newton-Raphson approach near the solution. These methods are sometimes described as "restricted step" approaches instead of trust-region approaches. See, for example, [Practical Methods of Optimization](https://doi.org/10.1002/9781118723203). As compared to general trust-region approaches in scalar minimization, the approach here for solving non-linear systems amounts to assuming that the Hessian matrix can be approximated using information from the Jacobian of the system.
  * `SNLSHybrdTrDLDenseG` A hybrid trust region type solver, dogleg approximation for dense general Jacobian matrix that makes use of a rank-1 update of the jacobian using QR factorization. The method is inspired by SNLS current trust region dogleg solver and Powell's original hybrid method for nonlinear equations, and MINPACK's modified version of it. Powell's original hybrid method can be found at: M. J. D. Powell, "A hybrid method for nonlinear equations", in Numerical methods for nonlinear algebraic equations. MINPACK's user guide is found at https://doi.org/10.2172/6997568 . As compared to general trust-region approaches in scalar minimization, the approach here for solving non-linear systems amounts to assuming that the Hessian matrix can be approximated using information from the Jacobian of the system.
  * `SNLSTrDlDenseG_Batch` a batch version of `SNLSTrDlDenseG` that is only available if the library is compiled with the `-DUSE_BATCH_SOLVERS=On` cmake option. It makes use of the RAJA Performance Suite to abstract away the complexity of memory management and execution strategies of forall loops. The `test/SNLS_batch_testdriver.cc` provides an example of how this solver, the memory manager, and forall abstraction layer can be used to solve a modified Broyden Tridiagonal Problem.
  * `NewtonBB` Simple 1D Newton solver with a fallback to bisection. If the zero of the function can be bounded then this solver can return a result even if the function is badly behaved.

  The non-batch solvers can now take in a lambda function instead of requiring a class variable as it's input. This design change does result in some minor breaking changes from versions of the code before v0.4.0. Mainly in that the dimension of the system must now be passed in as template parameter if using a lambda function.

  Version 0.4.0 of the code and later for the `NewtonBB` model has removed the unbounded template parameter. This parameter is now passed in as class constructor argument.

  Due to the C++17+ requirement in version 0.4.0 and later, the construction of solvers can also be greatly simplified as the compiler should be able to auto-deduce the CRJ/CFJ template parameter type.

# ABSTRACTION LAYERS

SNLS provides a number of abstraction layers built upon the RAJA Portability Suite.

## Forall abstractions
We provide a number of forall abstractions (`snls::forall()`) based upon the work done in the MFEM library that allow one to write a single forall abstraction and at runtime have the correct forall called. These abstractions can currently be found in the `src/SNLS_device_forall.h` file. In order to accomplish this runtime selection, users currently specify the execution strategy through the Device singleton object or they can make use of the `snls::forall_strat()` abstraction to specify it through the forall loop. It should be mentioned that each `snls::forall` layer has template parameters that allows one to specify the number of GPU blocks to use as well as if this should be an async call. For the purpose of async calls, we allow users to specify a RAJA resource set as obtained from the `Device::GetRAJAResource()` or `Device::GetDefaultRAJAResource()` calls. The `Device::GetRAJAResource()` will cycle through a set of available streams for the GPU kernel to launch on. This functionality is useful when you want to have multiple streams running at the same time. Additionally, each forall call will return a `RAJA::resources::Event` type which can be queried to determine if a kernel has finished. Note, we also provide a `Device::WaitFor()` function which when provided the matching RAJA resource set and RAJA resource event we can wait for a given forall call to finish.

## Memory Manager
We provide a simple memory manager (`snls::memoryManager`) which can be provided Umpire memory pool IDs to utilize when building `chai::ManagedArray<T>`. This can be found in the `SNLS_memory_manager.h` file. If we are not provided Umpire memory pools then we will just utilize whatever the default ones are for Umpire. It should be noted that we do not return raw pointers from this memory manager as we've decided to leverage some of the RAJA Portability suite CHAI abstraction layers to return safer objects that can automatically handle the shuffling of data between host and device.

## View and Subview types
We provide a couple `typedefs` for some common `RAJA::View` types that might be of interest to users working with the nonlinear solvers. Additionally, we provide a `SubView` class that allows us to take windows or sliding windows within a larger View type. This is particularly useful if we're dealing with batches of data and we want something to examine at a given point either the matrix, vector, or scalar data at that point. We allow users to also provide offsets into a view that they want their SubView to operate so that on the newest slowest operating index they're 0 index for the view is pointing to their value of interest. It should be noted that this type while powerful in that you build off of any RAJA-like View or a SubView it has a number of limitations. First, the SubView assumes your data is row-major and therefore the slowest index is the left-most index and this is where we take the index from. Next, we only allow the offset to apply to the next left-most index and not some arbitary other axis. In other words, if we'd provide a window over the data like `arr[ind, offset:-1, .. ,:]` So, we are not providing something as powerful as `std::mdspan` a C++23 feature born from the Kokkos project. Next, if you need to check if your view has data you can make use of the `SubView<T>::contains_data()` which returns true if the internal `RAJA::Layout::size() > 0`. While this isn't an absolute check that the underlying data is not null. We make the assumption that any one using this would not have a `View` with null data but a `Layout.size() > 0`.

## Linear Algebra functions
Within the `SNLS_linalg.h` we provide a number of basic linear algebra functions that we've found useful not only here but also in some of our other libraries that use SNLS. So, they are not going to be the most performant but they do reduce the number of equivalent functions that our other libraries might use as well. We also provide a basic `partial-pivoted LU` and `QR` linear solver for small system sizes. You can find those in the `SNLS_lup_solve.h` and `SNLS_qr_solve.h` headers.

## Availability of above abstractions
All of these are available if you build the code with the cmake feature `-DUSE_BATCH_SOLVERS=ON` which requires the RAJA Portability Suite to work. If you only have RAJA and you build with the cmake feature `-DUSE_RAJA_ONLY=ON` then you have access to all of the above but the memory manager abstractions. If you have none of those features turned on then you only have access to the linear algebra functions.

# BUILDING

The build system is cmake-based.

Dependencies:
* blt -- required
  - https://github.com/LLNL/blt
  - in cmake invocation specify location with `-DBLT_SOURCE_DIR`

# TESTING

Run cmake with `-DENABLE_TESTS=ON` and do `make test`

# DEVELOPMENT

The develop branch is the main development branch for snls. Changes to develop are by pull request.

# AUTHORS

The current principal developer of SNLS is Robert Carson. The original principal developer of SNLS was Nathan Barton, nrbarton@llnl.gov. Brett Wayne and Guy Bergel have also made contributions.

# CITATION

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

# LICENSE

License is under the BSD-3-Clause license. See [LICENSE](LICENSE) file for details. And see also the [NOTICE](NOTICE) file. 

`SPDX-License-Identifier: BSD-3-Clause`

``LLNL-CODE-761139``
