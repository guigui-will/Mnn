CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
SRC       = src/backend.cpp src/cpu_backend.cpp
BUILD     = build

# CUDA (opcional):  make CUDA=1   (requer nvcc)
ifeq ($(CUDA),1)
  NVCC     ?= nvcc
  CXXFLAGS += -DMNN_WITH_CUDA
  CUOBJ     = $(BUILD)/cuda_backend.o
  LDLIBS   += -lcudart
endif

all: $(BUILD)/test_all $(BUILD)/spiral

$(BUILD)/cuda_backend.o: src/cuda_backend.cu | $(BUILD)
	$(NVCC) -std=c++17 -O2 -Iinclude -c $< -o $@

$(BUILD)/test_all: tests/test_all.cpp $(SRC) $(CUOBJ) $(wildcard include/mnn/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_all.cpp $(SRC) $(CUOBJ) -o $@ $(LDLIBS)

$(BUILD)/spiral: examples/spiral.cpp $(SRC) $(CUOBJ) $(wildcard include/mnn/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Iexamples examples/spiral.cpp $(SRC) $(CUOBJ) -o $@ $(LDLIBS)

$(BUILD): ; mkdir -p $(BUILD)
test: all ; $(BUILD)/test_all cpu
clean: ; rm -rf $(BUILD)
.PHONY: all test clean
