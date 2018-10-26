### Variables for this project ###
GPUOBJ = image_blurring.o
CPUOBJ = cpu_image_blurring.o

# The executable programs to be created
CPU = cpu_image_blurring.cpp
GPU = image_blurring.cu

CC = nvcc
GCC = g++

CFLAGS = -std=c++11
CVFLAGS = `pkg-config --cflags --libs opencv`
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

GPUOUT = GPU_blurring.exe
CPUOUT = CPU_blurring.exe

IMAGES = GPU_Altered_Image.jpg CPU_Altered_Image.jpg OMP_Altered_Image.jpg

# Default rule
all: $(CPUOUT) $(GPUOUT)

# Rule to make the CPU program
$(CPUOUT): $(CPUOBJ)
	$(GCC) $^ -o $(CPUOUT) $(CFLAGS) $(CVFLAGS)

# Rule to make the GPU program
$(GPUOUT): $(GPUOBJ)
	$(CC) $(CFLAGS) -o $(GPUOUT) $< $(LDFLAGS)

# Rules to make the object files
$(CPUOBJ): $(CPU)
	$(GCC) $(CFLAGS) $(CVFLAGS) -c $<

$(GPUOBJ): $(GPU)
	$(CC) $(CFLAGS) -c $<

# Clear the compiled files
clean:
	rm -rf *.o $(CPUOUT) $(GPUOUT) $(IMAGES)
