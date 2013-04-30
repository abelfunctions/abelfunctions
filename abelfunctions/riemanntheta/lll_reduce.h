/*
Grady Williams
April 2013

Performs Lenstra-Lenstra-Lovasv reduction on a given lattive n in 
n-dimensional real space. The lc and  uc are the upper and lower constants 
used in the LLL-Algorithm. The default is lc = 1/2 and uc = 3/4. We 
presume that our algorithm has been given a matrix represented in an 
array with standard C ordering. Reduction is performed on the columns of b.
*/

void lll_reduce(double* b, int n, double lc, double uc);

void gram_schmidt(double* b, double* mu, double* B, int n);
