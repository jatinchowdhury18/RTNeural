#pragma once

#ifndef RTNEURAL_NAMESPACE
#define RTNEURAL_NAMESPACE RTNeural
#endif

#ifndef RTNEURAL_DEFAULT_ALIGNMENT
#if _MSC_VER
#pragma message("RTNEURAL_DEFAULT_ALIGNMENT was not defined! Using default alignment = 16.")
#else
#warning "RTNEURAL_DEFAULT_ALIGNMENT was not defined! Using default alignment = 16."
#endif
#define RTNEURAL_DEFAULT_ALIGNMENT 16
#endif

#ifdef RTNEURAL_RADSAN_ENABLED
  #define RTNEURAL_REALTIME [[clang::realtime]]
#else
  #define RTNEURAL_REALTIME
#endif
