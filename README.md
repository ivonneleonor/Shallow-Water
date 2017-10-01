# Shallow Water

The parallel code was made on Cuda C. An implicit scheme was used to solve the velocity and an explicit scheme to
solve the height. The algorithm to solve the implicit part was the conjugate gradient using Cublas and Cusparse 
libraries. On the other hand to solve the explicit part a funtion in parallel was implemented.