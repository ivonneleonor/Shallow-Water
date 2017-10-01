#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse_v2.h"
#include "cublas_v2.h"
#include "hello.cuh"
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do {if((x)!=cudaSuccess){\
   printf("Error at %s:%d\n",__FILE__,__LINE__);\
   }}while(0)

#define CLEANUP(s)      \
do{                     \
   printf("%s\n",s);    \
   if(I)     free(I);   \
   if(J)     free(J);   \
   if(val)   free(val); \
   if(r)     free(r);   \
   if(d_col)  cudaFree(d_col);   \
   if(d_row)  cudaFree(d_row);   \
   if(d_val)  cudaFree(d_val);   \
   if(d_x)    cudaFree(d_x);     \
   if(d_r)    cudaFree(d_r);     \
   if(d_p)    cudaFree(d_p);     \
   if(d_Ax)   cudaFree(d_Ax);    \
   if(d_r_c)  cudaFree(d_r_c);   \
   if(d_p_c)  cudaFree(d_p_c);   \
   if(descr)          cusparseDestroyMatDescr(descr);  \
   if(cublasHandle)   cublasDestroy(cublasHandle);   \
   if(cusparseHandle) cusparseDestroy(cusparseHandle);   \
   cudaDeviceReset(); \
   fflush(stdout); \
} while(0)
/*
__global__ void setup_kernel(curandState *state)
{
  int id=threadIdx.x+blockIdx.x*64;
  curand_init(1234,id,0,&state[id]);
}

__global__ void generate_kernel(curandState *state, unsigned int *result,unsigned int N)
{
  int id=threadIdx.x+blockIdx.x*64;
  int count=0;
  unsigned int x;
  curandState localState=state[id];
  for(int n=0;n<N;n++)
    { x=curand(&localState);}
  if((x&1))
    {count++;}

  state[id]=localState;
  result[id]+=count;
}



__global__ void set(double *dx,int N)
{
 int tid=threadIdx.x+blockIdx.x*64;
 if (tid<N)
i dx[tid]=1.0;
}
*/
extern "C"

{
 double *solverbicg(int* J,double* val,int* I,double* x,double* r, int N,int nz)
{

FILE *p1;
FILE *p2;
p1=fopen("Ax.dat","w+");
p2=fopen("x.dat","w+");
int i, k, kmax=4500;
const float tol = 1e-3f;
double alpha=1.0, alpham1 = -1.0, beta=0.0, r0=0.0, r1=0.0, dot=0.0, a=0.0, na=0.0, b=0.0;
double *d_Ax=0, *test=0,*test1=0, *d_val=0,*d_x=0, *d_r=0, *d_p=0, *d_r_c=0, *d_p_c=0;
int *d_col=0, *d_row=0;
//unsigned int total=0;
 unsigned int  *hostResults;
cudaError_t cudaStat1,cudaStat2,cudaStat3, cudaStat4,cudaStat5,cudaStat6,cudaStat7;
cublasStatus_t cublasStatus1;
cublasHandle_t cublasHandle=0;
cusparseStatus_t cusparseStatus1;
cusparseHandle_t cusparseHandle;
cusparseMatDescr_t descr;

   for(i=0;i<10;i++){
       printf("i=%d,J=%d,val=%7.3f,I=%d,x=%7.3f ,r=%e \n",i,J[i],val[i],I[i],x[i],r[i]);}
   printf("N=%d  nz=%d\n",N,nz);


hostResults=(unsigned int *)calloc(N,sizeof(int));
test = (double *)malloc(sizeof(double)*N);
test1 = (double *)malloc(sizeof(double)*N);
if((!test)||(!test1)||(!hostResults)){CLEANUP("Memory on host failed,test\n");}
 
    cudaStat1=cudaMalloc((void **)&d_col, nz*sizeof(int));
    cudaStat2=cudaMalloc((void **)&d_row, (N+1)*sizeof(int));
    cudaStat3=cudaMalloc((void **)&d_val, nz*sizeof(double));
    cudaStat4=cudaMalloc((void **)&d_x, N*sizeof(double));
    cudaStat5=cudaMalloc((void **)&d_r, N*sizeof(double));
    cudaStat6=cudaMalloc((void **)&d_p, N*sizeof(double));
    cudaStat7=cudaMalloc((void **)&d_Ax, N*sizeof(double));
   if((cudaStat1!=cudaSuccess)||(cudaStat2!=cudaSuccess)||(cudaStat3!=cudaSuccess)||(cudaStat4!=cudaSuccess)||(cudaStat5!=cudaSuccess)||(cudaStat6!=cudaSuccess)||(cudaStat7!=cudaSuccess)){printf("Memcpy from Host to Device failed"); }

// CUDA_CALL(cudaMalloc((void **)&devResults,N*sizeof(unsigned int)));

// CUDA_CALL(cudaMemset(d_x,1,N*sizeof(unsigned int)));
/*
 CUDA_CALL(cudaMalloc((void **)&devStates,N*sizeof(curandState)));
 generate_kernel<<<64,64>>>(devStates,devResults,N);
 CUDA_CALL(cudaMemcpy(hostResults,devResults,N*sizeof(unsigned int),cudaMemcpyDeviceToHost));

 for(i=0;i<N;i++)
 {
    total+=hostResults[i];
    printf("%d",hostResults[i]);
 }
 printf("Total de unos=%d\n",total);

for(i=0;i<nz;i++){
    printf("i=%d,J=%d,val=%7.3f,I=%d r=%7.3f \n",i,J[i],val[i],I[i],r[i]);}
 printf("N=%d  nz=%d\n",N,nz);
*/

 /* set<<<64,64>>>(d_x,N);

CUDA_CALL(cudaMemcpy(test,d_x,N*sizeof(double),cudaMemcpyDeviceToHost));
printf("d_x de la copia \n");

 for(i=0;i<N;i++)
 {
  //  total+=hostResults[i];
    printf("%f\n",test[i]);
 }

*/
/*
printf("desde el solver");
for(i=0;i<10;i++)
printf("J[%d]=%d\n",i,J[i]);
*/
    cudaStat1=cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaStat2=cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaStat3=cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaStat4=cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaStat5=cudaMemcpy(d_r, r, N*sizeof(double), cudaMemcpyHostToDevice);
    if((cudaStat1!=cudaSuccess)||(cudaStat2!=cudaSuccess)||(cudaStat3!=cudaSuccess)||(cudaStat4!=cudaSuccess)||(cudaStat5!=cudaSuccess)){printf("Memcpy from Host to Device failed\n");}

cusparseStatus1=cusparseCreate(&cusparseHandle);
 if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){
   CLEANUP("Cusparse create handle failed\n");}

cublasStatus1=cublasCreate(&cublasHandle);
  if(cublasStatus1!=CUBLAS_STATUS_SUCCESS){
     CLEANUP("Cublas create handle failed \n");}


cusparseStatus1=cusparseCreateMatDescr(&descr);
  if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){
        printf("Descriptor creation failed\n");}//Set matrix type and index base

cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

//printf("Start solver =)");

 cusparseStatus1=cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
    if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){
    CLEANUP("Ax0 performing failed\n");}

  cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    if(cublasStatus1!=CUBLAS_STATUS_SUCCESS){
     CLEANUP("b-Ax0 performing failed\n"); }


cublasStatus1 = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);


k=1;

 while (r1 > tol*tol && k <= kmax)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus1 = cublasDscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus1 = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus1 = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus1 = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus1 = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus1 = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus1 = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
//        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

//while(r1!=0.0 && k<kmax);

/*  
cudaStat1=cudaMemcpy(test,  d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaStat1!=cudaSuccess){
    CLEANUP("Memcpy from Device to Host failed\n"); }
  printf(" d_x_(j+1), \n ");
  for(i=0;i<N;i++){
     printf("%e\t",test[i]);
     test1[i]=test[i];
}
  printf("\n");
*/

printf("iteration = %3d, residual = %e\n", k, r1);

alpha=1.0;
beta=0.0;

cusparseStatus1=cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
    if(cusparseStatus1!=CUSPARSE_STATUS_SUCCESS){
    CLEANUP("Ax0 performing failed\n");}

/*
cudaStat1=cudaMemcpy(test,  d_Ax, N*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaStat1!=cudaSuccess){
    CLEANUP("Memcpy from Device to Host failed\n"); }
  printf(" A*x final, \n ");
  for(i=0;i<N;i++){
    printf("%e\n",test[i]);
  //   test1[i]=test[i];
}
  printf("\n");
*/



cudaStat1=cudaMemcpy(test,  d_Ax, N*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaStat1!=cudaSuccess){
    CLEANUP("Memcpy from Device to Host failed\n"); }
  printf(" A*x final, \n ");
  for(i=0;i<N;i++){
    fprintf(p1,"%e\n",test[i]);
}
  printf("\n");


cudaStat1=cudaMemcpy(test, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
  if(cudaStat1!=cudaSuccess){
    CLEANUP("Memcpy from Device to Host failed\n"); }
  printf(" x final, \n ");
  for(i=0;i<N;i++){
     fprintf(p2,"%e\n",test[i]);
     test1[i]=test[i];
}
  printf("\n");




 cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    fclose(p1);
    fclose(p2);
    free(I);
    free(J);
    free(val);
    free(r);
    free(test);
    free(hostResults);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);
   
    cudaDeviceReset();

    


 return test1;

}
}
