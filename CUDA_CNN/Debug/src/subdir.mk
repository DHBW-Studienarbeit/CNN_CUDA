################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/CUDA_CNN.cu \
../src/ConvLayer.cu \
../src/DropoutLayer.cu \
../src/FullyConnectedLayer.cu \
../src/InputLayer.cu \
../src/Layer.cu \
../src/MaxPoolingLayer.cu \
../src/Network.cu \
../src/Picture.cu \
../src/PictureContainer.cu \
../src/cuda_kernels.cu \
../src/mathematics.cu 

CU_DEPS += \
./src/CUDA_CNN.d \
./src/ConvLayer.d \
./src/DropoutLayer.d \
./src/FullyConnectedLayer.d \
./src/InputLayer.d \
./src/Layer.d \
./src/MaxPoolingLayer.d \
./src/Network.d \
./src/Picture.d \
./src/PictureContainer.d \
./src/cuda_kernels.d \
./src/mathematics.d 

OBJS += \
./src/CUDA_CNN.o \
./src/ConvLayer.o \
./src/DropoutLayer.o \
./src/FullyConnectedLayer.o \
./src/InputLayer.o \
./src/Layer.o \
./src/MaxPoolingLayer.o \
./src/Network.o \
./src/Picture.o \
./src/PictureContainer.o \
./src/cuda_kernels.o \
./src/mathematics.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


