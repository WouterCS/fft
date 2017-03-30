// include guard
#ifndef __NJET_H_INCLUDED__
#define __NJET_H_INCLUDED__

//=================================
// forward declared dependencies

//=================================
// included dependencies
#include <iostream>
#include <math.h>
#include <limits>

//=================================
// the actual code

#define FILTER_EXTENT 3.0

namespace njet
{
	int compute_number_of_filters(int order);

	void compute_range_and_size(double sigma, int& range, int& size);
	
	namespace continuous
	{
		namespace internal
		{
			// Auxilary math functions
			double signnum(double d);
			int factorial(int x);
				
			// Hermite polynomials	
			double evaluate_polynomial(double x, double const *coeff, int order);
			double* evaluate_polynomial(double* const x, int N, double* const coeffs, int order);
			double* hermite_polynomial_coeffs (int n);
			double* evaluate_hermite_polynomial(int order, double* const x, int N);
	
			// Gaussian, 1D and 2D Gaussian derivatives
			double* gaussian_1D(double sigma);
			double* gaussian_derivative_1D(int m, double sigma, bool normalize);
			double* gaussian_derivative_2D(int m, int n, double sigma_x, double sigma_y, bool normalize);
		
			// Sigma derivatives of gaussian derivatives
			double* gaussian_derivative_1D_sigma_derivative(int m, double sigma);
			double* gaussian_derivative_2D_sigmaX_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize);
			double* gaussian_derivative_2D_sigmaY_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize);
			double* gaussian_derivative_2D_sigma_derivative(int m, int n, double sigma, bool normalize);
		}
		
		// Anisotropic
		double** filters(int order, double sigma_x, double sigma_y, bool normalize);
		double** sigmaX_derivatives(int order, double sigma_x, double sigma_y, bool normalize);
		double** sigmaY_derivatives(int order, double sigma_x, double sigma_y, bool normalize);
		
		// Isotropic
		double** filters(int order, double sigma, bool normalize);
		double** sigma_derivatives(int order, double sigma, bool normalize);
	}
	
	namespace discrete
	{
		// Internal functions
		namespace internal
		{
			// Auxilary math functions
			double signnum(double d);
			
			// Modified Bessel function of the first kind [from: http://jean-pierre.moreau.pagesperso-orange.fr/Cplus/tbessi_cpp.txt]
		    double BESSI(int N, double X);
		    double BESSI0(double X);
		    double BESSI1(double X);
		
			// Difference operators
			double* first_order_difference(double* input, int N);
			double* second_order_difference(double* input, int N);
			
			// Gaussian, 1D and 2D Gaussian derivatives
			double* gaussian_1D(double sigma);
			double* gaussian_derivative_1D(int m, double sigma, bool normalize);
			double* gaussian_derivative_2D(int m, int n, double sigma_x, double sigma_y, bool normalize);
		
			// Sigma derivatives of gaussian derivatives
			double* gaussian_derivative_1D_sigma_derivative(int m, double sigma);
			double* gaussian_derivative_2D_sigmaX_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize);
			double* gaussian_derivative_2D_sigmaY_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize);
			double* gaussian_derivative_2D_sigma_derivative(int m, int n, double sigma, bool normalize);
		}
		
		// Anisotropic
		double** filters(int order, double sigma_x, double sigma_y, bool normalize);
		double** sigmaX_derivatives(int order, double sigma_x, double sigma_y, bool normalize);
		double** sigmaY_derivatives(int order, double sigma_x, double sigma_y, bool normalize);
		
		// Isotropic
		double** filters(int order, double sigma, bool normalize);
		double** sigma_derivatives(int order, double sigma, bool normalize);
	}
}

#endif // __NJET_H_INCLUDED__ 