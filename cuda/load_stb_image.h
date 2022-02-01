#ifndef LOAD_STB_IMAGE_H
#define LOAD_STB_IMAGE_H

// Disable pedantic warnings for this external library.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#pragma diag_suppress = set_but_not_used
#include "stb_image.h"
#pragma diag_default = set_but_not_used

// Restore warning levels.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
    #pragma warning (pop)
#endif


#endif
