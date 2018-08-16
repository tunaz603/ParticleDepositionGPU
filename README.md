# ParticleDepositionGPU
Particle deposition in grid according to Particle-In-Cell (PIC) Scheme using Graphical Processing Unit (GPU)

I developed parallel algorithm for particle deposition in grid according to Particle-In-Cell (PIC) scheme using Graphical Processing Unit (GPU). PIC simulations require intense computation. In this project, both sequential code and parallel code (using atomic operation) were implemented for particle deposition. I measured execution time for both CPU and GPU as well as the speed up. The main bottleneck of the code was to use of atomic. Using atomicAdd() instructions to access the global memory was particularly slow, both because global memory access is a few orders of magnitude slower than that of on-chip memories and because atomicity prevents parallel execution of the code by stalling other threads in the code segment where atomic instructions are located. To achieve more speed up, I implemented the parallel PIC simulation code using shared memory as this was much faster than local and global memory. I presented this work as a poster under the title “Particle deposition in grid according to Particle-In-Cell (PIC) Scheme using Graphical Processing Unit (GPU)” in both Grace Hopper Celebration of Women in Computing 2016 (GHC) and Capital Region Celebration of Women in Computing 2016 (CAPWIC). 

Here is the link of PDF: https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnx0dW5hemlzbGFtfGd4OjNhOThiNGVlOGY2NjIzMDU

Here is te link of the Poster PDF: https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnx0dW5hemlzbGFtfGd4OjdmZjAzMGM5ZjFiYmEzZTM
