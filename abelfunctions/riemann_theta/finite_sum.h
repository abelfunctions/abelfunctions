void
finite_sum_without_derivatives(double* fsum_real, double* fsum_imag,
                               double* X, double* Yinv, double* T,
                               double* x, double* y, double* S,
                               int g, int N);
void
finite_sum_with_derivatives(double* fsum_real, double* fsum_imag,
                            double* X, double* Yinv, double* T,
                            double* x, double* y, double* S,
                            double* deriv_real, double* deriv_imag,
                            int nderivs, int g, int N);
