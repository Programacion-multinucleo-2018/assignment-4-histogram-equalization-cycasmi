### Variables for this project ###
GPUOBJ = gpu_equalization.o
CPUOBJ = cpu_equalization.o

# The executable programs to be created
CPU = cpu_equalization.cpp
GPU = gpu_equalization.cu

CC = nvcc
GCC = g++

CFLAGS = -std=c++11
OMP = -fopenmp
CVFLAGS = `pkg-config --cflags --libs opencv`
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

GPUOUT = GPU_equalization.exe
CPUOUT = CPU_equalization.exe

IMAGES = GPU_Altered_Image.jpg CPU_Altered_Image.jpg

# Default rule
all: $(CPUOUT) $(GPUOUT)

# Rule to make the CPU program
$(CPUOUT): $(CPUOBJ)
	$(GCC) $^ -o $(CPUOUT) $(CFLAGS) $(OMP) $(CVFLAGS)

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
	rm -rf *.o $(CPUOUT) $(GPUOUT) $(GPUOBJ) $(CPUOBJ)  $(IMAGES)
