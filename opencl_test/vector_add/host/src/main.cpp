// Copyright (C) 2013-2017 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

///////////////////////////////////////////////////////////////////////////////////
// This host program executes a vector addition kernel to perform:
//  C = A + B
// where A, B and C are vectors with N elements.
//
// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M points. The host program
// assumes that all devices are of the same type (that is, the same binary can
// be used), but the code can be generalized to support different device types
// easily.
//
// Verification is performed against the same computation on the host CPU.
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements
#if USE_SVM_API == 0
scoped_array<cl_mem> cons_buf; // num_devices elements
scoped_array<cl_mem> prim_buf; // num_devices elements
#endif /* USE_SVM_API == 0 */

// Problem data.
unsigned ni = 256,nj = 256,nk = 256;
unsigned size = ni*nj*nk;// problem size
unsigned NHYDRO = 5;
#if USE_SVM_API == 0
scoped_array<scoped_aligned_ptr<float> > cons; // num_devices elements
scoped_array<scoped_aligned_ptr<float> > prim; // num_devices elements
#else
scoped_array<scoped_SVM_aligned_ptr<float> > cons; // num_devices elements
scoped_array<scoped_SVM_aligned_ptr<float> > prim; // num_devices elements
#endif /* USE_SVM_API == 0 */
scoped_array<scoped_array<float> > ref_prim; // num_devices elements
scoped_array<unsigned> n_per_device; // num_devices elements
scoped_array<unsigned> me_per_device; // num_devices elements


//Problem specific number
const unsigned IDN = 0;
const unsigned IM1 = 1;
const unsigned IM2 = 2;
const unsigned IM3 = 3;
const unsigned IEN = 4;

// const unsigned IDN = 0;
const unsigned IVX = 1;
const unsigned IVY = 2;
const unsigned IVZ = 3;
const unsigned IPR = 4;

const float density_floor_;
const float pressure_floor_;
const float gm1_;


// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  // Requires the number of devices to be known.
  init_problem();

  // Run the kernel.
  run();

  // Free the resources allocated
  cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("vector_add", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  n_per_device.reset(num_devices);
  mem_per_device.reset(num_devices);
#if USE_SVM_API == 0
  cons_buf.reset(num_devices);
  input_b_buf.reset(num_devices);
  prim_buf.reset(num_devices);
#endif /* USE_SVM_API == 0 */

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "vector_add";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    n_per_device[i] = size / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if(i < (size % num_devices)) {
      n_per_device[i]++;
    }

    mem_per_device[i] = n_per_device[i]*NHYDRO;

#if USE_SVM_API == 0
    // Input buffers.
    cons_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        mem_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for cons");

    // Output buffer.
    prim_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        mem_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for prim");
#else
    cl_device_svm_capabilities caps = 0;

    status = clGetDeviceInfo(
      device[i],
      CL_DEVICE_SVM_CAPABILITIES,
      sizeof(cl_device_svm_capabilities),
      &caps,
      0
    );
    checkError(status, "Failed to get device info");

    if (!(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
      printf("The host was compiled with USE_SVM_API, however the device currently being targeted does not support SVM.\n");
      // Free the resources allocated
      cleanup();
      return false;
    }
#endif /* USE_SVM_API == 0 */
  }

  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  cons.reset(num_devices);
  prim.reset(num_devices);
  ref_prim.reset(num_devices);

  // Generate input vectors A and B and the reference prim consisting
  // of a total of N elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer.
  for(unsigned i = 0; i < num_devices; ++i) {
#if USE_SVM_API == 0
    cons[i].reset(mem_per_device[i]);
    prim[i].reset(mem_per_device[i]);
    ref_prim[i].reset(mem_per_device[i]);

    for(unsigned j = 0; j < n_per_device[i]; ++j) {
      //Initialize cons
      float c = (j + 1.0)/size_;
      cons[i][IDN*size + j] = c;
      cons[i][IM1*size + j] = sin(c);
      cons[i][IM2*size + j] = cos(c);
      cons[i][IM3*size + j] = tan(c);
      cons[i][IEN*size + j] = c*c+4.0;

			//Find prim on the host
			float& u_d  = cons[i][IDN*size + j];
			float& u_m1 = cons[i][IM1*size + j];
			float& u_m2 = cons[i][IM2*size + j];
			float& u_m3 = cons[i][IM3*size + j];
			float& u_e  = cons[i][IEN*size + j];

			float& w_d  = ref_prim[i][IDN*size + j];
			float& w_vx = ref_prim[i][IVX*size + j];
			float& w_vy = ref_prim[i][IVY*size + j];
			float& w_vz = ref_prim[i][IVZ*size + j];
			float& w_p  = ref_prim[i][IPR*size + j];

			// apply density floor, without changing momentum or energy
			u_d = (u_d > density_floor_) ?  u_d : density_floor_;
			w_d = u_d;

			float di = 1.0/u_d;
			w_vx = u_m1*di;
			w_vy = u_m2*di;
			w_vz = u_m3*di;

			float ke = 0.5*di*(u_m1*u_m1 + u_m2*u_m2 + u_m3*u_m3);
			w_p = gm1_*(u_e - ke);

			// apply pressure floor, correct total energy
			u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
			w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

    }
#else
    cons[i].reset(context, mem_per_device[i]);
    prim[i].reset(context, mem_per_device[i]);
    ref_prim[i].reset(mem_per_device[i]);

    cl_int status;

    status = clEnqueueSVMMap(queue[i], CL_TRUE, CL_MAP_WRITE,
        (void *)cons[i], mem_per_device[i] * sizeof(float), 0, NULL, NULL);
    checkError(status, "Failed to map cons");

    for(unsigned j = 0; j < n_per_device[i]; ++j) {
      //Initialize cons
      float c = (j + 1.0)/size_;
      cons[i][IDN*size + j] = c;
      cons[i][IM1*size + j] = sin(c);
      cons[i][IM2*size + j] = cos(c);
      cons[i][IM3*size + j] = tan(c);
      cons[i][IEN*size + j] = c*c+4.0;

			//Find prim on the host
			float& u_d  = cons[i][IDN*size + j];
			float& u_m1 = cons[i][IM1*size + j];
			float& u_m2 = cons[i][IM2*size + j];
			float& u_m3 = cons[i][IM3*size + j];
			float& u_e  = cons[i][IEN*size + j];

			float& w_d  = ref_prim[i][IDN*size + j];
			float& w_vx = ref_prim[i][IVX*size + j];
			float& w_vy = ref_prim[i][IVY*size + j];
			float& w_vz = ref_prim[i][IVZ*size + j];
			float& w_p  = ref_prim[i][IPR*size + j];

			// apply density floor, without changing momentum or energy
			u_d = (u_d > density_floor_) ?  u_d : density_floor_;
			w_d = u_d;

			float di = 1.0/u_d;
			w_vx = u_m1*di;
			w_vy = u_m2*di;
			w_vz = u_m3*di;

			float ke = 0.5*di*(u_m1*u_m1 + u_m2*u_m2 + u_m3*u_m3);
			w_p = gm1_*(u_e - ke);

			// apply pressure floor, correct total energy
			u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1_) + ke);
			w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
    }

    status = clEnqueueSVMUnmap(queue[i], (void *)cons[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap cons");
#endif /* USE_SVM_API == 0 */
  }
}

void run() {
  cl_int status;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {

#if USE_SVM_API == 0
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue[i], cons_buf[i], CL_FALSE,
        0, mem_per_device[i] * sizeof(float), cons[i], 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer cons");

#endif /* USE_SVM_API == 0 */

    // Set kernel arguments.
    unsigned argi = 0;

#if USE_SVM_API == 0
    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &cons_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &prim_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);
#else
    status = clSetKernelArgSVMPointer(kernel[i], argi++, (void*)cons[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArgSVMPointer(kernel[i], argi++, (void*)prim[i]);
    checkError(status, "Failed to set argument %d", argi - 1);
#endif /* USE_SVM_API == 0 */

    status = clSetKernelArg(kernel[i], argi++, sizeof(unsigned), &size);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(float), &density_floor_);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(float), &pressure_floor_);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(float), &gm1_);
    checkError(status, "Failed to set argument %d", argi - 1);

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    //
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = n_per_device[i];
    printf("Launching for device %d (%zd elements)\n", i, global_work_size);

#if USE_SVM_API == 0
    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event[i]);
#else
    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
        &global_work_size, NULL, 0, NULL, &kernel_event[i]);
#endif /* USE_SVM_API == 0 */
    checkError(status, "Failed to launch kernel");

#if USE_SVM_API == 0
    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], prim_buf[i], CL_FALSE,
        0, mem_per_device[i] * sizeof(float), prim[i], 1, &kernel_event[i], &finish_event[i]);

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
#else
    status = clEnqueueSVMMap(queue[i], CL_TRUE, CL_MAP_READ,
        (void *)prim[i], mem_per_device[i] * sizeof(float), 0, NULL, NULL);
    checkError(status, "Failed to map prim");
	clFinish(queue[i]);
#endif /* USE_SVM_API == 0 */
  }

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }

  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }

  // Verify results.
  bool pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < mem_per_device[i] && pass; ++j) {
      if(fabsf(prim[i][j] - ref_prim[i][j]) > 1.0e-5f) {
        printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
            i, j, prim[i][j], ref_prim[i][j]);
        pass = false;
      }
    }
  }

#if USE_SVM_API == 1
  for (unsigned i = 0; i < num_devices; ++i) {
    status = clEnqueueSVMUnmap(queue[i], (void *)prim[i], 0, NULL, NULL);
    checkError(status, "Failed to unmap prim");
  }
#endif /* USE_SVM_API == 1 */
  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
#if USE_SVM_API == 0
    if(cons_buf && cons_buf[i]) {
      clReleaseMemObject(cons_buf[i]);
    }
    if(prim_buf && prim_buf[i]) {
      clReleaseMemObject(prim_buf[i]);
    }
#else
    if(cons[i].get())
      cons[i].reset();
    if(prim[i].get())
      prim[i].reset();
#endif /* USE_SVM_API == 0 */
  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

