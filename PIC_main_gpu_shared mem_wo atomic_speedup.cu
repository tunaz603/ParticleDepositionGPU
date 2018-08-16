#include <iostream>
#include<cmath>
#include<iomanip>
#include<stdlib.h>
#include<ctime>
#include <sys/time.h>


#include<fstream>
//for tokenizer
#include <cstring>
#include <sstream>//split


#include <curand.h>
#include <curand_kernel.h>



#define N 256
#define Gx N
#define Gy N
#define block_size 128

using namespace std;

struct particle{
    double xcoord;
    double ycoord;
};

struct freqInfo{
    int freqIndex;
    int freqStart;
};


void CalculateCoordinate(particle* h_point, long int n);
double* ParticleDeposition_CPU(particle* point, freqInfo * freqCell, long int n);
double* ParticleDeposition_GPU(particle* h_point, freqInfo * h_freqCell, long int n);
void checkCUDAError(const char* msg);


double xn_max = 0.5, yn_max = 0.5;

int sort_xy(void const *a, void const *b)
{
    double dXn= (2.0 * xn_max)/ (Gx - 1);
    double dYn= (2.0 * yn_max)/ (Gy - 1);

    particle *pa, *pb;
    pa = (particle *) a;
    pb = (particle *) b;

    int itx =0;
    int ity =0;
    int ix =0;
    int iy =0;

    itx = (pa->xcoord + xn_max)/dXn;
    ity = (pa->ycoord + yn_max)/dYn;
    ix  = itx+1;
    iy  = ity+1;

    int pax=ix;
    int pay=iy;

    //second point's grid
    itx = (pb->xcoord + xn_max)/dXn;
    ity = (pb->ycoord + yn_max)/dYn;
    ix  = itx+1;
    iy  = ity+1;
    int pbx=ix;
    int pby=iy;

    if( pax < pbx) return -1;
    if( pax > pbx) return 1;

    if( pay < pby) return -1;
    if( pay > pby) return 1;

    return 0;
}





__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void kernel(particle* d_point, double *d_rho,freqInfo* d_freqCell,long int n)
{
    double xn_max = 0.5, yn_max = 0.5;    
    double dXn= (2 * xn_max)/ (Gx - 1);
    double dYn= (2 * yn_max)/ (Gy - 1);
   // double s1[4];
    __shared__ double s[block_size][4];
    int actIndexCell = blockIdx.x*N + threadIdx.x;
    if(actIndexCell<(N*N))
    {
      
            /*s1[0]=0.0;
            s1[1]=0.0;
            s1[2]=0.0;
            s1[3]=0.0;*/
            s[threadIdx.x][0]=0.0;
            s[threadIdx.x][1]=0.0;
            s[threadIdx.x][2]=0.0;
            s[threadIdx.x][3]=0.0;
            //cout<<s[0]<<", "<<s[1]<<", "<<s[2]<<", "<<s[3]<<endl;
            int pointIndex=d_freqCell[actIndexCell].freqStart;
            for(int jk=0;jk<d_freqCell[actIndexCell].freqIndex ;jk++)
            {

                double tx  = (d_point[pointIndex].xcoord + xn_max)/dXn;
                int itx = (d_point[pointIndex].xcoord + xn_max)/dXn;
                double wx2 = tx - itx;
                double wx1 = 1.0 - wx2;


                double ty  = (d_point[pointIndex].ycoord + yn_max)/dYn;
                int ity = (d_point[pointIndex].ycoord + yn_max)/dYn;
                double wy2 = ty - ity;
                double wy1 = 1.0 - wy2;

                /*s[0]+=  wx1 * wy1;
                s[1]+= wx2 * wy1;
                s[2]+= wx1 * wy2;
                s[3]+= wx2 * wy2;*/

                s[threadIdx.x][0]+=  wx1 * wy1;
                s[threadIdx.x][1]+= wx2 * wy1;
                s[threadIdx.x][2]+= wx1 * wy2;
                s[threadIdx.x][3]+= wx2 * wy2;

               //loop count
                pointIndex++;//increase point index

            }
            int i = blockIdx.x;
            int j= threadIdx.x;
          
          // __syncthreads();
          /* atomicAdd(&d_rho[(j*N)+i]   ,  s[threadIdx.x][0]) ;
            atomicAdd(&d_rho[(j*N)+(i+1)]   ,s[threadIdx.x][1]);
           atomicAdd( &d_rho[((j+1)*N)+i]    , s[threadIdx.x][2]);
            atomicAdd(&d_rho[((j+1)*N)+(i+1)]  , s[threadIdx.x][3]);*/

            d_rho[(j*N)+i]      += s[threadIdx.x][0];
            d_rho[(j*N)+(i+1)]    += s[threadIdx.x][1];
            d_rho[((j+1)*N)+i]    += s[threadIdx.x][2];
            d_rho[((j+1)*N)+(i+1)]  += s[threadIdx.x][3];
        }


}


int main()
{
    srand(1); //seed value set as 1
    // srand(time(NULL));
    long int n=100000000; //number of particle
    //long int n;
    cout<<"The number of particles "<<n<<endl;
    cout<<"The grid size is "<<Gx<<"*"<<Gy<<endl;
    //cin>>n;

    //structure pointer to host
   
    particle* h_point= 0;
    h_point= (particle*)malloc(n*sizeof(particle));

    freqInfo * freqCell = new freqInfo[N*N];
    for(int i=0;i<N*N;i++){
        freqCell[i].freqIndex=0;
        freqCell[i].freqStart=0;
    }

    CalculateCoordinate(h_point,n);

    qsort(h_point, n, sizeof(particle), sort_xy);
    //calculate freq array
    double dXn= (2.0 * xn_max)/ (Gx - 1);
    double dYn= (2.0 * yn_max)/ (Gy - 1);
    int itx =0;
    int ity =0;
    int ix =0;
    int iy =0;
    int freqIndex=0;
    int prevFreqIndex=0;
    for(int m=0;m<n;m++){
        itx =0;
        ity =0;
        ix =0;
        iy =0;
        itx = (h_point[m].xcoord + xn_max)/dXn;
        ity = (h_point[m].ycoord + yn_max)/dYn;
        ix  = itx+1;
        iy  = ity+1;
        freqIndex=ix*N+iy;
        if(freqIndex!=prevFreqIndex){//when change the index then update start of previous index
            freqCell[prevFreqIndex].freqStart=m-freqCell[prevFreqIndex].freqIndex;
        }
        prevFreqIndex=freqIndex;
        freqCell[freqIndex].freqIndex++;//increament count
    }


    double *rho_cpu;
    rho_cpu =(double*)malloc((N+1)*(N+1)*sizeof(double));
    rho_cpu=ParticleDeposition_CPU(h_point,freqCell, n);
    double *rho_gpu;
    rho_gpu =(double*)malloc((N+1)*(N+1)*sizeof(double));
    rho_gpu=ParticleDeposition_GPU(h_point, freqCell, n);
    double *diff_rho;
    diff_rho =(double*)malloc((N+1)*(N+1)*sizeof(double));
    int flag=0;
    for ( int k = 1; k <= Gx; k++)
    {
        for(int l = 1; l <= Gx; l++ )
        {
            diff_rho[(k*Gx)+l]=fabs(rho_cpu[(k*Gx)+l]-rho_gpu[(k*Gx)+l]);
            if(diff_rho[(k*Gx)+l] < (1.0e-10) )
            {
                flag=1;
            }

        }
    }

    if(flag==1)
    {
        cout<<"Result matches"<<endl;
    }

    else
    {
        cout<<"Result does not match"<<endl;
    }


    free(h_point);
    free(rho_cpu);
    free(rho_gpu);
    free(diff_rho);
    return 0;
    
}


void CalculateCoordinate(particle* h_point, long int n)
{
    
     for (int i= 0; i< n; i++)
    {
        h_point[i].xcoord= (double)rand()/RAND_MAX- .5;
        h_point[i].ycoord= (double)rand()/RAND_MAX- .5;
    }
    // print the particles with coordinate
    for ( int j = 0; j < n; j++) {
    //  cout<<fixed<< setprecision(16) << h_point[j].xcoord << " , " << setprecision(16)<< h_point[j].ycoord << endl;
    }
    cout << endl;

}

double* ParticleDeposition_CPU(particle* point, freqInfo * freqCell, long int n)
{
    
     
     double dXn= (2.0 * xn_max)/ (Gx - 1);
     double dYn= (2.0 * yn_max)/ (Gy - 1);
     ofstream weightfile_cpu,rhofile_cpu;
     weightfile_cpu.open ("weightfile(double precision)_cpu.txt");
     rhofile_cpu.open ("rhofile(double precision)_cpu.txt");
     double *rho;
     rho =(double*)malloc((N+1)*(N+1)*sizeof(double));

    double *s;
    s=(double*)malloc(4*sizeof(double));
     
 
     cudaEvent_t start_cpu, stop_cpu;
     cudaEventCreate(&start_cpu);
     cudaEventCreate(&stop_cpu);
     cudaEventRecord(start_cpu, 0);
    for (int i=0; i< N; i++)
    {

        for(int j=0;j<N;j++)
        {
            int actIndexCell=i*N+j;

            s[0]=0.0;
            s[1]=0.0;
            s[2]=0.0;
            s[3]=0.0;
            //cout<<s[0]<<", "<<s[1]<<", "<<s[2]<<", "<<s[3]<<endl;
            int pointIndex=freqCell[actIndexCell].freqStart;
            for(int jk=0;jk<freqCell[actIndexCell].freqIndex ;jk++)
            {

                double tx  = (point[pointIndex].xcoord + xn_max)/dXn;
                int itx = (point[pointIndex].xcoord + xn_max)/dXn;
                double wx2 = tx - itx;
                double wx1 = 1.0 - wx2;


                double ty  = (point[pointIndex].ycoord + yn_max)/dYn;
                int ity = (point[pointIndex].ycoord + yn_max)/dYn;
                double wy2 = ty - ity;
                double wy1 = 1.0 - wy2;

                s[0]+=  wx1 * wy1;
                s[1]+= wx2 * wy1;
                s[2]+= wx1 * wy2;
                s[3]+= wx2 * wy2;

               //loop count
                pointIndex++;//increase point index

            }

            rho[(j*N)+i]      +=  s[0];
            rho[(j*N)+(i+1)]    += s[1];
            rho[((j+1)*N)+i]    += s[2];
            rho[((j+1)*N)+(i+1)]  += s[3];

         /*   int i= k/N;
            int j= k%N;

            rho[(j*N)+i]      +=  s[0];
            rho[(j*N)+(i+1)]    += s[1];
            rho[((j+1)*N)+i]    += s[2];
            rho[((j+1)*N)+(i+1)]  += s[3];*/
        }


    }
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    float elapsedTime_cpu;
    cudaEventElapsedTime(&elapsedTime_cpu, start_cpu, stop_cpu); 
    cout<<"Time required for execution in CPU:: "<<elapsedTime_cpu<<" milliseconds"<<endl;

    // print the weights of each particle
    for ( int j = 0; j < n; j++)
    {
        weightfile_cpu<<fixed<< setprecision(16) << w1[j] << " , " << setprecision(16)<< w2[j] << " , " << setprecision(16)<< w3[j] <<" , " << setprecision(16)<< w4[j] <<endl;
    }


     for ( int k = 1; k <= Gx; k++)
     {
  
        for(int l = 1; l <= Gx; l++ )
        {
            rhofile_cpu<< fixed<<setprecision(16) << "The rho value at CPU: ( "<<k<<" , "<<l<<" ) =" <<rho[(k*Gx)+l] << endl;
           
        }

    }

    //free(rho);
    weightfile_cpu.close();
    rhofile_cpu.close();
    return rho;
 }


double* ParticleDeposition_GPU(particle* h_point,freqInfo * h_freqCell, long int n)
{
  ofstream weightfile_GPU,rhofile_GPU;
  weightfile_GPU.open ("weightfile(double precision)_GPU.txt");
  rhofile_GPU.open ("rhofile(double precision)_GPU.txt");
  double *h_rho;
  h_rho= (double*)malloc((N+1)*(N+1)*sizeof(double));
  for (int i=0; i<(N+1)*(N+1); i++)
  {
    h_rho[i]= 0;
  }


    //structure pointer to device
    particle* d_point=0;
    // cudaMalloc for device structure
    cudaMalloc((void**)&d_point,n*sizeof(particle));
    cudaMemcpy(d_point,h_point, n*sizeof(particle), cudaMemcpyHostToDevice);

    freqInfo* d_freqCell=0;
    // cudaMalloc for device structure
    cudaMalloc((void**)&d_freqCell,N*N*sizeof(freqInfo));
    cudaMemcpy(d_freqCell,h_freqCell, N*N*sizeof(freqInfo), cudaMemcpyHostToDevice);



     double *d_rho;
     cudaMalloc((void**)&d_rho,(N+1)*(N+1)*sizeof(double));
     cudaMemset(d_rho,0,(N+1)*(N+1)*sizeof(double));
     cudaMemcpy(d_rho,h_rho, (N+1)*(N+1)*sizeof(double), cudaMemcpyHostToDevice);

     

     // if either memory allocation failed, report an error message
     if( h_rho==0 || d_rho==0 )
     {
        printf("couldn't allocate memory\n");
        //return 1;
     }

     // number of threads per block
     //int block_size = 128;

     // number of blocks per grid
    // int grid_size = (N / block_size)+1;
     int grid_size = block_size;

     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start, 0);
    
     // This is called "configuring" the launch.
     kernel<<<grid_size,block_size>>>(d_point,d_rho,d_freqCell,n);

     cudaEventRecord(stop, 0);
     cudaEventSynchronize(stop);
     float elapsedTime;
     cudaEventElapsedTime(&elapsedTime, start, stop); 
     cout<<"Time required for execution in GPU:: "<<elapsedTime<<" milliseconds"<<endl;

     // Clean up:
     cudaEventDestroy(start);
     cudaEventDestroy(stop);
   // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation");

     cudaMemcpy(h_rho,d_rho, (N+1)*(N+1)*sizeof(double), cudaMemcpyDeviceToHost);

      // Check for any CUDA errors
    checkCUDAError("memcpy");

     //print weight
     for ( int j = 0; j < n; j++)
    {
        weightfile_GPU<<fixed<< setprecision(16)<< h_w1[j] << " , " << setprecision(16)<< h_w2[j] << " , " << setprecision(16)<< h_w3[j] <<" , " << setprecision(16)<< h_w4[j] <<endl;

    }

     // print out the rho

    for ( int k =1; k <= Gx; k++)
     {
        //cout << "The rho:::  " << setw(3);
        for(int l = 1; l <= Gx; l++ )
        {
            rhofile_GPU<<fixed<< setprecision(16) <<"The rho value at GPU: ( "<<k<<" , "<<l<<" ) =" <<h_rho[(k*Gx)+l] << endl;
           
        }

    }
          
     
    

    //deallocate memory
    

    cudaFree(d_point);
    cudaFree(d_rho);
    cudaFree(d_freqCell);
    weightfile_GPU.close();
    rhofile_GPU.close();  
    return h_rho; 
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}