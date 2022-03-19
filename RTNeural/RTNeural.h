#pragma once

// global include file for the RTNeural library!

// C++ STL includes
#include <limits>

// Handle default RTNeural defines
#ifndef RTNEURAL_DEFAULT_ALIGNMENT
#warning "RTNEURAL_DEFAULT_ALIGNMENT was not defined! Using default alginment = 16."
#define RTNEURAL_DEFAULT_ALIGNMENT 16
#endif

#if !defined(RTNEURAL_USE_EIGEN) && !defined(RTNEURAL_USE_XSIMD) && !defined(RTNEURAL_USE_STL)
#warning "No backend was defined for RTNeural! Using STL backend as a fallback, even though it might will be slower."
#define RTNEURAL_USE_STL 1
#endif

// RTNeural includes:
#include "Model.h"
#include "ModelT.h"
#include "model_loader.h"
