DEEPSTREAM_DIR ?= /opt/nvidia/deepstream/deepstream
CUDA_VER ?= 12.8
CC := g++
CFLAGS := -Wall -std=c++14 -shared -fPIC
CFLAGS += -I$(DEEPSTREAM_DIR)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
LIBS := -lnvinfer
LFLAGS := -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES := nvdsparsebbox_yolo26_ensemble.cpp
TARGET_LIB := libnvds_yolo26_ensemble_parser.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
