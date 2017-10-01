#include<stdio.h>
#include<cuda_runtime.h>
#include<math.h>
#include "cusparse_v2.h"
#include "cublas_v2.h"
#include "hello.cuh"
#include <cuda.h>
#include <stdlib.h>
#include<iostream>
using namespace std;

main(int argc, char *argv[])
{
//numero de nodos
int N=10;
//tamaño de la matriz
double nodos=N+2;
//numero de valores no cero dados lo nodos
int nnz=3*(nodos-2)-2;
int i;
double dx=0, dt=0,H0=10, g=9.81,beta,pi=0, L=0, alfa=0;
double *x0=0,*csrValA=0, *u=0, *b=0, *answer=0, *eta=0;
int *csrRowPtrA=0,*csrColIndA=0;

/*
csrValA=(double *)malloc(sizeof(double)*nnz);
csrRowPtrA=(int *)malloc(sizeof(int)*(N+1));
csrColIndA=(int *)malloc(sizeof(int)*nnz);
u=(double *)malloc(sizeof(double)*(N+1));
b=(double *)malloc(sizeof(double)*N);
answer = (double *)malloc(sizeof(double)*N);
x0=(double *)malloc(sizeof(double)*N);
*/

csrValA = new double[nnz];
csrRowPtrA = new int[N+1];
csrColIndA = new int[nnz];
u = new double[N+2];
b = new double[N];
answer = new double[N];
x0= new double[N];
eta= new double[N+1];

printf("Numero de nodos=%f\n",nodos);
printf("Tamaño de la matriz=%d\n",N);
printf("Numero de valores no cero=%d\n",nnz);
L=atoi(argv[1]);
printf("L=%f \n",L);
dx=L/(nodos-1);
printf("dx=%f\n",dx);
dt=dx/(10*sqrt(g*H0));
printf("dt=%f\n",dt);
beta=g*H0*(dt*dt)/(dx*dx);
printf("beta=%f\n",beta);
pi=atan(1.0f)*4.0f;
printf("pi=%f\n",pi);
alfa=g*dt/dx;
//se llena vector csrValA
for(i=0;i<=N-2;i++)
{
csrValA[3*i]=(1+2*beta);
csrValA[3*i+1]=-beta;
csrValA[3*i+2]=-beta;
}
csrValA[nnz-1]=(1+2*beta);
//se impreime csr
for(i=0;i<nnz;i++)
{
  printf("csrval[%d]=%f\n",i,csrValA[i]);
}

//se llena vector csrRowPtr
csrRowPtrA[0]=0;
for(i=0;i<=N-2;i++)
{
csrRowPtrA[i+1]=2+3*i;
}
csrRowPtrA[N]=nnz;
//se imprime csrRowPrt
for(i=0;i<=N;i++)
{
  printf("csrRowPtrA[%d]=%d\n",i,csrRowPtrA[i]);
}


//se llena csrColIndA
csrColIndA[0]=0;
csrColIndA[1]=1;
for(i=0;i<N-2;i++)
{
csrColIndA[2+3*i]=0+i;
csrColIndA[3+3*i]=1+i;
csrColIndA[4+3*i]=2+i;
}
csrColIndA[nnz-2]=N-2;
csrColIndA[nnz-1]=N-1;

//se imprime csrColIndA

for(i=0;i<nnz;i++)
{
  printf("csrColIndA[%d]=%d\n",i,csrColIndA[i]);
}

u[0]=u[N+2]=0.0f;
//condiciones de frontera
for(i=1;i<=N;i++)
{
  u[i]=0.0f;
}
//imprime condiciones de frontera
for(i=0;i<nodos;i++)
{
  printf("u[%d]=%f\n",i,u[i]);
}

for(i=1;i<=nodos-1;i++)
{
  eta[i-1]=0.5-0.5*cos(2*pi*i*dx/L);
}
for(i=0;i<nodos-1;i++)
{
  printf("eta[%d]=%e\n",i,eta[i]);
}



//se define y llena el vector b
b[0]=u[1]+beta*u[0]-alfa*(eta[1]-eta[0]);
b[N-1]=u[N-1]+beta*u[N]-alfa*(eta[N]-eta[N-1]);
for(i=1;i<N-1;i++)
{
  b[i]=u[i+1]-alfa*(eta[i+1]-eta[i]);
}
//se imprime el vector b
for(i=0;i<N;i++)
{
  printf("b[%d]=%f\n",i,b[i]);
}
//guess inicial
for(i=0;i<N;i++)
{
  x0[i]=0;
  printf("x0[%d]=%f\n",i,x0[i]);  
}


answer=solverbicg((int*) csrColIndA,(double*) csrValA,(int*) csrRowPtrA,(double*) x0,(double *) b, N, nnz);


printf("Desde main\n");
for(i=0;i<N;i++)
{
printf("%e \n",*(answer+i));
}

FILE *f;
f = fopen ("u_0.csv","w+");
fprintf(f,"x coord, y coord\n");
fprintf(f,"%f,%f\n",0.0f,u[0]);
for(i=0;i<N;i++)
{
  fprintf(f,"%f,%f\n",(i+1)*dx,answer[i]);
  printf("%f,%f\n",(i+1)*dx,answer[i]);
}
fprintf(f,"%f,%f\n",L,u[N+2]);
fclose(f);




/*
delete [] csrValA;
delete [] csrRowPtrA;
delete [] csrColIndA;
delete [] u;
delete [] b;
delete [] answer;
delete [] x0;
*/








return 0;

}
