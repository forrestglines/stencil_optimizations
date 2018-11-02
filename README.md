
The goal of my project is to determine the best/easiest way to obtain maximum
performance for stenciled functions simultaneously for CPUs and GPUs using
Kokkos. Explicit methods such as finited difference and finite volume for
MHD/hydrodynamics on structured meshes can typically be decomposed into a
series of stenciled functions applied to the mesh. Rather than write an entire
MHD code using several different approaches, I want to focus on a simple
function, such as a centered finite derivative. Depending on time constraints,
I will implement this hypothetical dieal approach in the hydro code Kathena and
compare to a naive Kokkos implementation and a CPU optimized Athena version.

----------------------------------------
Basic types of functions I want to test:

Stenciless operations

Centered 1D stencil operations, in X, Y, and Z.

Small and Large Stencils

High and Low arthmetic intensities

High and Low register use (simple and complex functions)

----------------------------------------
The main questions I want answered are:

What is the fastest CPU implementation?

What is the fastest Kokkos implementation for CPUs?

What is the fastest Kokkos implementation for GPUs?

What is the fastest Kokkos implementation for both CPUs and GPUs?

What is the fastest CUDA implementation for GPUs?

What is the easiest+fast Kokkos implementation for both CPUs and GPUs?

How do all of these implementations compare?

----------------------------------------
Specific approaches I want to try/Questions/Random Notes

Varying problem size
Varying register use?

How to get SIMD computations on the CPU?

Can Kokkos give the same performance as CUDA?

Does UVM incur a performance penalty?
Even if the data stays on the GPU?
In CUDA? In Kokkos?
How big a performance penalty is there if data is transferred on and off the GPU every time step?

How do you best utilize the explicit caches?
On GPUs?
Can you do this with Kokkos?
On CPUs with SIMD?

Do aligned data accesses matter?
On the GPU?
On the CPU?

What if boundaries are computed in a separate function?
    -On the GPU?
    -On the CPU? (w/, w/o UVM)

What if boundaries are transferred via MPI?
    -Between CUDA addresses? 
    -UVM?
    -Same node? Separate nodes?
    -cudaAdvise - in Kokkos too?

How does high register use change these answers? (EG. in reconstruction?)

Can OpenACC give the same performance as Kokkos? As CUDA?

Do these same considerations apply for the FPGAs?

