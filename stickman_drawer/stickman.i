%module stickman

%{
    #define SWIG_FILE_WITH_INIT
	#include "stickman.hpp"
%}

%include "numpy.i"

%init %{
	import_array();
%}

%apply (float* INPLACE_ARRAY1, int DIM1) {(float * output, int sz1), (float * input, int sz2)}


%include "stickman.hpp"

class StickmanData_C;
