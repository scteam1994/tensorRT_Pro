#pragma once

#ifndef BUILD_STATIC
# if defined(TRT_LIB)
#  define TRT_EXPORT __declspec(dllexport)
# else
#  define TRT_EXPORT __declspec(dllimport)
# endif
#else
# define TRT_EXPORT
#endif