#pragma once

// global include file for the RTNeural library!

// C++ STL includes
#include <limits>

// Handle default RTNeural defines
#ifndef RTNEURAL_DEFAULT_ALIGNMENT
#if _MSC_VER
#pragma message("RTNEURAL_DEFAULT_ALIGNMENT was not defined! Using default alignment = 16.")
#else
#warning "RTNEURAL_DEFAULT_ALIGNMENT was not defined! Using default alignment = 16."
#endif
#define RTNEURAL_DEFAULT_ALIGNMENT 16
#endif

// RTNeural includes:
#include "Model.h"
#include "ModelT.h"
#include "model_loader.h"
#include "torch_helpers.h"
