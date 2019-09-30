#pragma once
#include "fp16_emu.h"
#include "fp16_dev.h"

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T>
struct ScaleFactorTypeMap { typedef T Type; };
template <> struct ScaleFactorTypeMap<half1> { typedef float Type; };

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
	template <class T>
	value_type operator()(T x) { return value_type(x); } //float<=>double
	value_type operator()(half1 x) { return value_type(cpu_half2float(x)); }//half1=half1
};

template <>
class Convert<half1>
{
public:
	template <class T>
	half1 operator()(T x) { return cpu_float2half_rn(T(x)); }
	half1 operator()(half1 x) { return x; }
};

