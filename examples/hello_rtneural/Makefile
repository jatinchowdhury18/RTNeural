CXXFLAGS = -O3 -I../../ -std=c++17 -DRTNEURAL_DEFAULT_ALIGNMENT=16

# Extra flags for compiling with Eigen backend
# CXXFLAGS += -DRTNEURAL_USE_EIGEN=1 -I../../modules/Eigen

# Extra flags for compiling with XSIMD backend
# CXXFLAGS += -DRTNEURAL_USE_XSIMD=1 -I../../modules/xsimd/include

SRCS=hello_rtneural.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

hello_rtneural.exe: $(OBJS)
	$(CXX) -o hello_rtneural.exe $(OBJS)

hello_rtneural.o: hello_rtneural.cpp
	$(CXX) $(CXXFLAGS) -c hello_rtneural.cpp
