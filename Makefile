CXX = clang++
CXXFLAGS = -std=c++17 -fobjc-arc -I src
LDFLAGS = -framework Metal -framework Foundation

TARGET = orchard_test
SRC = src/main.cpp src/platform/metal_backend.mm src/runtime/tensor.cpp src/runtime/model.cpp src/runtime/kv_cache.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
