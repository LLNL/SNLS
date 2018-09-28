		SNLS -- small non-linear solver library

	 ________  ________   ___       ________      
	|\   ____\|\   ___  \|\  \     |\   ____\     
	\ \  \___|\ \  \\ \  \ \  \    \ \  \___|_    
	 \ \_____  \ \  \\ \  \ \  \    \ \_____  \   
	  \|____|\  \ \  \\ \  \ \  \____\|____|\  \  
	    ____\_\  \ \__\\ \__\ \_______\____\_\  \ 
	   |\_________\|__| \|__|\|_______|\_________\
	   \|_________|                   \|_________|
	                                              

COPYRIGHT
======

Copyright 2017-2018, Lawrence Livermore National Security, LLC. All
rights reserved.  This work was produced at the Lawrence Livermore
National Laboratory (LLNL) under contract no. DE-AC52-07NA27344
(Contract 44) between the U.S. Department of Energy (DOE) and Lawrence
Livermore National Security, LLC (LLNS) for the operation of
LLNL. Copyright is reserved to Lawrence Livermore National Security,
LLC for purposes of controlled dissemination, commercialization
through formal licensing, or other disposition under terms of Contract
44; DOE policies, regulations and orders; and U.S. statutes. The
rights of the Federal Government are reserved under Contract 44.

DISCLAIMER
======

This work was prepared as an account of work sponsored by an agency of
the United States Government. Neither the United States Government nor
Lawrence Livermore National Security, LLC nor any of their employees,
makes any warranty, express or implied, or assumes any liability or
responsibility for the accuracy, completeness, or usefulness of any
information, apparatus, product, or process disclosed, or represents
that its use would not infringe privately-owned rights. Reference
herein to any specific commercial products, process, or service by
trade name, trademark, manufacturer or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by
the United States Government or Lawrence Livermore National Security,
LLC. The views and opinions of authors expressed herein do not
necessarily state or reflect those of the United States Government or
Lawrence Livermore National Security, LLC, and shall not be used for
advertising or product endorsement purposes.

LICENSING REQUIREMENTS
======

Any use, reproduction, modification, or distribution of this software
or documentation for commercial purposes requires a license from
Lawrence Livermore National Security, LLC. Contact: Lawrence Livermore
National Laboratory, Industrial Partnerships Office, P.O. Box 808,
L-795, Livermore, CA 94551.

The Government is granted for itself and others acting on its behalf a
paid-up, nonexclusive, irrevocable worldwide license in this data to
reproduce, prepare derivative works, and perform publicly and display
publicly.

ACCESS
======

... Release for SNLS is in process -- but it is NOT yet released. 

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
