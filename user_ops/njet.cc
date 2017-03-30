#include "njet.h"

namespace njet
{
	int compute_number_of_filters(int order)
	{
		// Number of filters = the number of elements in the upper triangle of a matrix (diagonal inclusive).
		// #elements = N*(N-1)/2 + N
		return (order+1) * ((order+1) - 1) / 2 + (order+1);
	}
	
	void compute_range_and_size(double sigma, int& range, int& size)
	{
		range = ceil(FILTER_EXTENT * sigma);
		size = 2 * range + 1;
		
		return;
	}
	
	namespace continuous
	{
		// Internal functions
		namespace internal
		{
			// Auxilary math functions
			double signnum(double d)
			{
				// It's important that we take machine precision into account when comparing floating point numbers!
				double eps = std::numeric_limits<double>::epsilon();
			    return (d < -eps)?-1:(d > eps);
			}
		
			int factorial(int x)
			{
				int val = 1;
				while(x > 0)
					val *= x--;
			
				return val;
			}
		
			// Hermite polynomials	
			double evaluate_polynomial(double x, double const *coeff, int order)
			{
				// Evaluate polynome at x 
				// Note: the coefficients are in reverse order!
				double value = 0;
				for(int i = 0; i < order+1; i++)
					value += coeff[i] * pow(x, order-i);
			
				return value;
			}	
	
			double* evaluate_polynomial(double* const x, int N, double* const coeffs, int order)
			{
				// Initialize the output array
				double* output = new double[N]();
			
				for(int i = 0; i < N; i++)
					output[i] = evaluate_polynomial(x[i], coeffs, order);
		
				return output;
			}
		
			double* hermite_polynomial_coeffs (int n)
			{
				// This function uses the explicit definition of hermite polynomials
				// to return the coefficients of a hermite polynomial of order n. 
				// Note that the coefficients are returned in a reversed order! 
			
				// Initialize the output array
				double* coeffs = new double[n+1]();
			
				for(int m = 0; m <= floor(n/2); m++)
					coeffs[2*m] = factorial(n) * pow(-1, m) / (factorial(m) * factorial(n-2*m)) * pow(2, n-2*m); 
			
				return coeffs;
			}
		
			double* evaluate_hermite_polynomial(int order, double* const x, int N)
			{
				if(order < 0)
				{
			 	   double* ones = new double[N]();
				   for(int i = 0; i < N; i++)
					   ones[i] = 1.0;
			   
				   return ones;
				 }
			
		 	   	// Create coefficients of the Hermite polynomial
		    	double* coeffs = hermite_polynomial_coeffs(order); 
			
				// Evaluate Hermite polynomial at the locations of the x-values
				double* values = evaluate_polynomial(x, N, coeffs, order);
			
				// Free heap memory
				delete[] coeffs;
			
				return values;
			}
	
			// Gaussian, 1D and 2D Gaussian derivatives
			double* gaussian_1D(double sigma)
			{
		    	// Compute the support width for the gaussian filter
		    	int range = ceil(FILTER_EXTENT * sigma);

				// Initialize output array
				double* gaussian = new double[2*range+1]();
	
				// Fill the output array with a 0th order gaussian
				int index = 0;
				for(double x = -range; x <= range; x++)
					gaussian[index++] = (1.0 / (sqrt(2.0 * M_PI) * sigma)) * exp(-0.5 * x*x / (sigma*sigma));

		    	return gaussian;
			}
		
			double* gaussian_derivative_1D(int m, double sigma, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range = ceil(FILTER_EXTENT * sigma);
				int size = range * 2 + 1;
		
				// Initialize the output array
				double* kernel = new double[size]();
			
				// Create range of x values: [-range,range] / (sqrt(2) * sigma)
				double* x = new double[size];
				for(int i = 0; i < size; i++)
					x[i] = (-range + i) / (sqrt(2) * sigma);
			
				// Evaluate a hermite polynomial of order m at the locations of the x-values
				double* Hm = evaluate_hermite_polynomial(m, x, size);
				
				// Create a 1d gaussian of 0th order
				double* gaussian = gaussian_1D(sigma);
			
				// Create the kernel by putting everything together
	       	 	double coeff = pow(-1 / (sigma * sqrt(2)), m);
				for(int i = 0; i < size; i++)
					kernel[i] = coeff * Hm[i] * gaussian[i];
			
				// Normalize if requested
				if(normalize)
				{
					double volume = 0;
					for(int i = 0; i < size; i++)
						volume += fabs(kernel[i]);
				
					for(int i = 0; i < size; i++)
						kernel[i] /= volume;
				}
				
				// Free heap memory
				delete[] x;
				delete[] Hm;
				delete[] gaussian;
			
				return kernel;
			}
		
			double* gaussian_derivative_2D(int m, int n, double sigma_x, double sigma_y, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range_x = ceil(FILTER_EXTENT * sigma_x);
		    	int range_y = ceil(FILTER_EXTENT * sigma_y);
				int size_x = range_x * 2 + 1;
				int size_y = range_y * 2 + 1;
		
				// Initialize the output array
				double* kernel = new double[size_x*size_y]();
			
				// Compute the two 1D derivatives
				double* Gm = gaussian_derivative_1D(m, sigma_x, normalize);
				double* Gn = gaussian_derivative_1D(n, sigma_y, normalize);
			
				// Combine the two 1D derivatives into a single 2D derivative
				for(int y = 0; y < size_y; y++)
					for(int x = 0; x < size_x; x++)
						kernel[y*size_x+x] = Gm[x] * Gn[y];
			
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
			
				return kernel;
			}
		
			// Sigma derivatives of gaussian derivatives
			double* gaussian_derivative_1D_sigma_derivative(int m, double sigma)
			{
		    	// Compute the range and kernel size
		    	int range = ceil(FILTER_EXTENT * sigma);
				int size = range * 2 + 1;
		
				// Initialize the output array
				double* sigma_derivative = new double[size]();
			
				// Create a range of x values: [-range, range]
				// and a range of scaled x values:  [-range, range] / (sqrt(2) * sigma)
				double* x = new double[size];
				double* x_scaled = new double[size];
				for(int i = 0; i < size; i++)
				{
					x[i] = (-range + i);
					x_scaled[i] = x[i] / (sqrt(2) * sigma);
				}
			
				// Evaluate a hermite polynomials at the locations of the scaled x values
				double* Hm_0 = evaluate_hermite_polynomial(m, x_scaled, size);
				double* Hm_1 = evaluate_hermite_polynomial(m-1, x_scaled, size);
			
				// Create a 1d gaussian of 0th order
				double* gaussian = gaussian_1D(sigma);
			
				// Create the sigma derivative by putting everything together
	       	 	double coeff = pow(-1 / (sigma * sqrt(2)), m) * 1 / pow(sigma, 3);
				for(int i = 0; i < size; i++)
					sigma_derivative[i] = coeff * gaussian[i] * ((pow(x[i],2)-(1+m)*pow(sigma,2)) * Hm_0[i] - sqrt(2) * m * x[i] * sigma * Hm_1[i]);
			
				// Free heap memory
				delete[] x;
				delete[] x_scaled;
				delete[] Hm_0;
				delete[] Hm_1;
				delete[] gaussian;
			
				return sigma_derivative;
			}
		
			double* gaussian_derivative_2D_sigmaX_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range_x = ceil(FILTER_EXTENT * sigma_x);
		    	int range_y = ceil(FILTER_EXTENT * sigma_y);
				int size_x = range_x * 2 + 1;
				int size_y = range_y * 2 + 1;
		
				// Initialize the output array
				double* sigma_derivative = new double[size_x*size_y]();
			
				// Compute each part of the product rule individually
				double* Gm = gaussian_derivative_1D(m, sigma_x, false);
				double* Gn = gaussian_derivative_1D(n, sigma_y, false);
				double* dGm_dsigma = gaussian_derivative_1D_sigma_derivative(m, sigma_x);
			
				// Use the product rule to find the combined derivative
				for(int y = 0; y < size_y; y++)
					for(int x = 0; x < size_x; x++)
						sigma_derivative[y*size_x+x] = dGm_dsigma[x] * Gn[y];
		
				// Compute the derivative of the normalized kernel instead if requested
				if(normalize)
				{
					double volume = 0;
					double sign_sum = 0;
					for(int y = 0; y < size_y; y++)
					{
						for(int x = 0; x < size_x; x++)
						{
							volume += fabs(Gm[x]*Gn[y]);
							sign_sum += signnum(Gm[x]*Gn[y]) * sigma_derivative[y*size_x+x];
						}
					}
				
					for(int y = 0; y < size_y; y++)
						for(int x = 0; x < size_x; x++)
							sigma_derivative[y*size_x+x] = (1 / volume) * sigma_derivative[y*size_x+x] - Gm[x]*Gn[y] / pow(volume,2) * sign_sum;
				}
			
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
				delete[] dGm_dsigma;
			
				return sigma_derivative;
			}
		
			double* gaussian_derivative_2D_sigmaY_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range_x = ceil(FILTER_EXTENT * sigma_x);
		    	int range_y = ceil(FILTER_EXTENT * sigma_y);
				int size_x = range_x * 2 + 1;
				int size_y = range_y * 2 + 1;
		
				// Initialize the output array
				double* sigma_derivative = new double[size_x*size_y]();
			
				// Compute each part of the product rule individually
				double* Gm = gaussian_derivative_1D(m, sigma_x, false);
				double* Gn = gaussian_derivative_1D(n, sigma_y, false);
				double* dGn_dsigma = gaussian_derivative_1D_sigma_derivative(n, sigma_y);
			
				// Use the product rule to find the combined derivative
				for(int y = 0; y < size_y; y++)
					for(int x = 0; x < size_x; x++)
						sigma_derivative[y*size_x+x] = dGn_dsigma[y] * Gm[x];
		
				// Compute the derivative of the normalized kernel instead if requested
				if(normalize)
				{
					double volume = 0;
					double sign_sum = 0;
					for(int y = 0; y < size_y; y++)
					{
						for(int x = 0; x < size_x; x++)
						{
							volume += fabs(Gm[x]*Gn[y]);
							sign_sum += signnum(Gm[x]*Gn[y]) * sigma_derivative[y*size_x+x];
						}
					}
				
					for(int y = 0; y < size_y; y++)
						for(int x = 0; x < size_x; x++)
							sigma_derivative[y*size_x+x] = (1 / volume) * sigma_derivative[y*size_x+x] - Gm[x]*Gn[y] / pow(volume,2) * sign_sum;
				}
			
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
				delete[] dGn_dsigma;
			
				return sigma_derivative;
			}
		
			double* gaussian_derivative_2D_sigma_derivative(int m, int n, double sigma, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range = ceil(FILTER_EXTENT * sigma);
				int size = range * 2 + 1;
		
				// Initialize the output array
				double* sigma_derivative = new double[size*size]();
			
				// Compute each part of the product rule individually
				double* Gm = gaussian_derivative_1D(m, sigma, false);
				double* Gn = gaussian_derivative_1D(n, sigma, false);
				double* dGn_dsigma = gaussian_derivative_1D_sigma_derivative(n, sigma);
				double* dGm_dsigma = gaussian_derivative_1D_sigma_derivative(m, sigma);
			
				// Use the product rule to find the combined derivative
				for(int y = 0; y < size; y++)
					for(int x = 0; x < size; x++)
						sigma_derivative[y*size+x] = Gm[x] * dGn_dsigma[y] + Gn[y] * dGm_dsigma[x];
		
				// Compute the derivative of the normalized kernel instead if requested
				if(normalize)
				{
					double volume = 0;
					double sign_sum = 0;
					for(int y = 0; y < size; y++)
					{
						for(int x = 0; x < size; x++)
						{
							volume += fabs(Gm[x]*Gn[y]);
							sign_sum += signnum(Gm[x]*Gn[y]) * sigma_derivative[y*size+x];
						}
					}
				
					for(int y = 0; y < size; y++)
						for(int x = 0; x < size; x++)
							sigma_derivative[y*size+x] = (1 / volume) * sigma_derivative[y*size+x] - Gm[x]*Gn[y] / pow(volume,2) * sign_sum;
				}
			
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
				delete[] dGm_dsigma;
				delete[] dGn_dsigma;
			
				return sigma_derivative;
			}
		}
		
		// Anisotropic
		double** filters(int order, double sigma_x, double sigma_y, bool normalize=true)
		{
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_x <= 0)
			{
	   	 		std::cout << "Error: sigma_x should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			if(sigma_y <= 0)
			{
	   	 		std::cout << "Error: sigma_y should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
			
			// Initialize the output array
			double** filters = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					filters[index++] = internal::gaussian_derivative_2D(m, n, sigma_x, sigma_y, normalize);
				}
			}
			
			return filters;
		}
		
		double** sigmaX_derivatives(int order, double sigma_x, double sigma_y, bool normalize=true)
		{
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_x <= 0)
			{
	   	 		std::cout << "Error: sigma_x should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_y <= 0)
			{
	   	 		std::cout << "Error: sigma_y should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
		
			// Initialize the output array
			double** sigma_derivatives = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					sigma_derivatives[index++] = internal::gaussian_derivative_2D_sigmaX_derivative(m, n, sigma_x, sigma_y, normalize);
				}
			}
			
			return sigma_derivatives;
		}
		
		double** sigmaY_derivatives(int order, double sigma_x, double sigma_y, bool normalize=true)
		{
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_x <= 0)
			{
	   	 		std::cout << "Error: sigma_x should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_y <= 0)
			{
	   	 		std::cout << "Error: sigma_y should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
		
			// Initialize the output array
			double** sigma_derivatives = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					sigma_derivatives[index++] = internal::gaussian_derivative_2D_sigmaY_derivative(m, n, sigma_x, sigma_y, normalize);
				}
			}
			
			return sigma_derivatives;
		}
		
		// Isotropic
		double** filters(int order, double sigma, bool normalize=true)
		{
			return filters(order, sigma, sigma, normalize);
		}
		
		double** sigma_derivatives(int order, double sigma, bool normalize=true)
		{
			// Output of the isotropic case shoudl be the same as
			// filter_bank_sigmaX_derivatives + filter_bank_sigmaY_derivatives
			
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma <= 0)
			{
	   	 		std::cout << "Error: sigma should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
		
			// Initialize the output array
			double** sigma_derivatives = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					sigma_derivatives[index++] = internal::gaussian_derivative_2D_sigma_derivative(m, n, sigma, normalize);
				}
			}
			
			return sigma_derivatives;
		}
		
	}
	
	namespace discrete
	{
		// Internal functions
		namespace internal
		{
			// Auxilary math functions
			double signnum(double d)
			{
				// It's important that we take machine precision into account when comparing floating point numbers!
				double eps = std::numeric_limits<double>::epsilon();
			    return (d < -eps)?-1:(d > eps);
			}
			
			// Modified Bessel function of the first kind
			// Code from: http://jean-pierre.moreau.pagesperso-orange.fr/Cplus/tbessi_cpp.txt
		    double BESSI0(double X);
		    double BESSI1(double X);

		    double BESSI(int N, double X) 
			{
				N = abs(N); // Sten: added this line in order to handle negative N as well!
			
				// ----------------------------------------------------------------------//
				//    This subroutine calculates the first kind modified Bessel function
				//    of integer order N, for any REAL X. We use here the classical
				//    recursion formula, when X > N. For X < N, the Miller's algorithm
				//    is used to avoid overflows. 
				//    REFERENCE:
				//    C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
				//    MATHEMATICAL TABLES, VOL.5, 1962.
				//------------------------------------------------------------------------//

		        int IACC = 40; 
		  	  	double BIGNO = 1e10, BIGNI = 1e-10;
		        double TOX, BIM, BI, BIP, BSI;
		        int J, M;

		        if (N==0)  return (BESSI0(X));
		        if (N==1)  return (BESSI1(X));
		        if (X==0.0) return 0.0;

		        TOX = 2.0/X;
		        BIP = 0.0;
		        BI  = 1.0;
		        BSI = 0.0;
		        M = (int) (2*((N+floor(sqrt(IACC*N)))));
		        for (J = M; J>0; J--) {
		          BIM = BIP+J*TOX*BI;
		          BIP = BI;
		          BI  = BIM;
		          if (fabs(BI) > BIGNO) {
		            BI  = BI*BIGNI;
		            BIP = BIP*BIGNI;
		            BSI = BSI*BIGNI;
		          }
		          if (J==N)  BSI = BIP;
		        }
		        return (BSI*BESSI0(X)/BI);
		    }

		  	//  Auxiliary Bessel functions for N=0, N=1
		    double BESSI0(double X) 
			{
		        double Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX;
		        P1=1.0; P2=3.5156229; P3=3.0899424; P4=1.2067492;
		        P5=0.2659732; P6=0.360768e-1; P7=0.45813e-2;
		        Q1=0.39894228; Q2=0.1328592e-1; Q3=0.225319e-2;
		        Q4=-0.157565e-2; Q5=0.916281e-2; Q6=-0.2057706e-1;
		        Q7=0.2635537e-1; Q8=-0.1647633e-1; Q9=0.392377e-2;
		        if (fabs(X) < 3.75) {
		          Y=(X/3.75)*(X/3.75);
		          return (P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))));
		        }
		        else {
		          AX=fabs(X);
		          Y=3.75/AX;
		          BX=exp(AX)/sqrt(AX);
		          AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))));
		          return (AX*BX);
		        }
		    }
			
		    double BESSI1(double X) 
			{
		        double Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX;
		        P1=0.5; P2=0.87890594; P3=0.51498869; P4=0.15084934;
		        P5=0.2658733e-1; P6=0.301532e-2; P7=0.32411e-3;
		        Q1=0.39894228; Q2=-0.3988024e-1; Q3=-0.362018e-2;
		        Q4=0.163801e-2; Q5=-0.1031555e-1; Q6=0.2282967e-1;
		        Q7=-0.2895312e-1; Q8=0.1787654e-1; Q9=-0.420059e-2;
		        if (fabs(X) < 3.75) {
		          Y=(X/3.75)*(X/3.75);
		          return(X*(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7)))))));
		        }
		        else {
		          AX=fabs(X);
		          Y=3.75/AX;
		          BX=exp(AX)/sqrt(AX);
		          AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))));
		          return (AX*BX);
		        }
		    }
		
			double* first_order_difference(double* input, int N)
			{
				double* output = new double[N]();
			
				output[0] = 0.5*input[1];
				for(int x = 1; x < N-1; x++)
					output[x] = 0.5*(input[x+1] - input[x-1]);
				output[N-1] = -0.5*input[N-2];
			
				return output;
			}
		
			double* second_order_difference(double* input, int N)
			{
				double* output = new double[N]();
			
				output[0] = (input[1] - 2 * input[0]);
				for(int x = 1; x < N-1; x++)
					output[x] = input[x+1] - 2 * input[x] + input[x-1];
				output[N-1] = -2 * input[N-1] + input[N-2];
			
				return output;
			}
		
			// Gaussian, 1D and 2D Gaussian derivatives
			double* gaussian_1D(double sigma)
			{
		    	// Compute the support width for the gaussian filter
		    	int range = ceil(FILTER_EXTENT * sigma);

				// Initialize output array
				double* gaussian = new double[2*range+1]();
	
				// Fill the output array with a 0th order gaussian
				int index = 0;
				double t = pow(sigma, 2);
				double coeff = exp(-t);
				for(int x = -range; x <= range; x++)
					gaussian[index++] = coeff * BESSI(x, t);

		    	return gaussian;
			}
		
			double* gaussian_derivative_1D(int m, double sigma, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range = ceil(FILTER_EXTENT * sigma);
				int size = range * 2 + 1;
		
				// Create a 1d gaussian of 0th order
				double* kernel = gaussian_1D(sigma);
			
				for(int i = 0; i < floor(m/2); i++)
					kernel = second_order_difference(kernel, size);
			
				if(m%2 == 1)
					kernel = first_order_difference(kernel, size);
			
				// Normalize if requested
				if(normalize)
				{
					double volume = 0;
					for(int i = 0; i < size; i++)
						volume += fabs(kernel[i]);
				
					for(int i = 0; i < size; i++)
						kernel[i] /= volume;
				}
			
				return kernel;
			}
		
			double* gaussian_derivative_2D(int m, int n, double sigma_x, double sigma_y, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range_x = ceil(FILTER_EXTENT * sigma_x);
		    	int range_y = ceil(FILTER_EXTENT * sigma_y);
				int size_x = range_x * 2 + 1;
				int size_y = range_y * 2 + 1;
		
				// Initialize the output array
				double* kernel = new double[size_x*size_y]();
			
				// Compute the two 1D derivatives
				double* Gm = gaussian_derivative_1D(m, sigma_x, normalize);
				double* Gn = gaussian_derivative_1D(n, sigma_y, normalize);
			
				// Combine the two 1D derivatives into a single 2D derivative
				for(int y = 0; y < size_y; y++)
					for(int x = 0; x < size_x; x++)
						kernel[y*size_x+x] = Gm[x] * Gn[y];
			
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
			
				return kernel;
			}
		
			// Sigma derivatives of gaussian derivatives
			double* gaussian_derivative_1D_sigma_derivative(int m, double sigma)
			{
		    	// Compute the range and kernel size
		    	int range = ceil(FILTER_EXTENT * sigma);
				int size = range * 2 + 1;
			
				// Initialize output array
				double* sigma_derivative = new double[size];
		
				// First differentiate the 1D gaussian w.r.t. sigma
				double t = pow(sigma, 2);
				double coeff = sigma * exp(-t);
				int index = 0;
				for(int x = -range; x <= range; x++)
					sigma_derivative[index++] = coeff * (BESSI(x-1, t) - 2*BESSI(x, t) + BESSI(x+1, t));
			
				// Then differentiate w.r.t. the spatial coordinate
				for(int i = 0; i < floor(m/2); i++)
					sigma_derivative = second_order_difference(sigma_derivative, size);
			
				if(m % 2 == 1)
					sigma_derivative = first_order_difference(sigma_derivative, size);
			
				return sigma_derivative;
			}
			
			double* gaussian_derivative_2D_sigmaX_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range_x = ceil(FILTER_EXTENT * sigma_x);
		    	int range_y = ceil(FILTER_EXTENT * sigma_y);
				int size_x = range_x * 2 + 1;
				int size_y = range_y * 2 + 1;
		
				// Initialize the output array
				double* sigma_derivative = new double[size_x*size_y]();
			
				// Compute each part of the product rule individually
				double* Gm = gaussian_derivative_1D(m, sigma_x, false);
				double* Gn = gaussian_derivative_1D(n, sigma_y, false);
				double* dGm_dsigma = gaussian_derivative_1D_sigma_derivative(m, sigma_x);
			
				// Use the product rule to find the combined derivative
				for(int y = 0; y < size_y; y++)
					for(int x = 0; x < size_x; x++)
						sigma_derivative[y*size_x+x] = dGm_dsigma[x] * Gn[y];
		
				// Compute the derivative of the normalized kernel instead if requested
				if(normalize)
				{
					double volume = 0;
					double sign_sum = 0;
					for(int y = 0; y < size_y; y++)
					{
						for(int x = 0; x < size_x; x++)
						{
							volume += fabs(Gm[x]*Gn[y]);
							sign_sum += signnum(Gm[x]*Gn[y]) * sigma_derivative[y*size_x+x];
						}
					}
				
					for(int y = 0; y < size_y; y++)
						for(int x = 0; x < size_x; x++)
							sigma_derivative[y*size_x+x] = (1 / volume) * sigma_derivative[y*size_x+x] - Gm[x]*Gn[y] / pow(volume,2) * sign_sum;
				}
			
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
				delete[] dGm_dsigma;
			
				return sigma_derivative;
			}
		
			double* gaussian_derivative_2D_sigmaY_derivative(int m, int n, double sigma_x, double sigma_y, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range_x = ceil(FILTER_EXTENT * sigma_x);
		    	int range_y = ceil(FILTER_EXTENT * sigma_y);
				int size_x = range_x * 2 + 1;
				int size_y = range_y * 2 + 1;
		
				// Initialize the output array
				double* sigma_derivative = new double[size_x*size_y]();
			
				// Compute each part of the product rule individually
				double* Gm = gaussian_derivative_1D(m, sigma_x, false);
				double* Gn = gaussian_derivative_1D(n, sigma_y, false);
				double* dGn_dsigma = gaussian_derivative_1D_sigma_derivative(n, sigma_y);
			
				// Use the product rule to find the combined derivative
				for(int y = 0; y < size_y; y++)
					for(int x = 0; x < size_x; x++)
						sigma_derivative[y*size_x+x] = dGn_dsigma[y] * Gm[x];
		
				// Compute the derivative of the normalized kernel instead if requested
				if(normalize)
				{
					double volume = 0;
					double sign_sum = 0;
					for(int y = 0; y < size_y; y++)
					{
						for(int x = 0; x < size_x; x++)
						{
							volume += fabs(Gm[x]*Gn[y]);
							sign_sum += signnum(Gm[x]*Gn[y]) * sigma_derivative[y*size_x+x];
						}
					}
				
					for(int y = 0; y < size_y; y++)
						for(int x = 0; x < size_x; x++)
							sigma_derivative[y*size_x+x] = (1 / volume) * sigma_derivative[y*size_x+x] - Gm[x]*Gn[y] / pow(volume,2) * sign_sum;
				}
			
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
				delete[] dGn_dsigma;
			
				return sigma_derivative;
			}
		
			double* gaussian_derivative_2D_sigma_derivative(int m, int n, double sigma, bool normalize=true)
			{
		    	// Compute the range and kernel size
		    	int range = ceil(FILTER_EXTENT * sigma);
				int size = range * 2 + 1;
		
				// Initialize the output array
				double* sigma_derivative = new double[size*size]();
			
				// Compute each part of the product rule individually
				double* Gm = gaussian_derivative_1D(m, sigma, false);
				double* Gn = gaussian_derivative_1D(n, sigma, false);
				double* dGn_dsigma = gaussian_derivative_1D_sigma_derivative(n, sigma);
				double* dGm_dsigma = gaussian_derivative_1D_sigma_derivative(m, sigma);
			
				// Use the product rule to find the combined derivative
				for(int y = 0; y < size; y++)
					for(int x = 0; x < size; x++)
						sigma_derivative[y*size+x] = Gm[x] * dGn_dsigma[y] + Gn[y] * dGm_dsigma[x];
		
				// Compute the derivative of the normalized kernel instead if requested
				if(normalize)
				{
					double volume = 0;
					double sign_sum = 0;
					for(int y = 0; y < size; y++)
					{
						for(int x = 0; x < size; x++)
						{
							volume += fabs(Gm[x]*Gn[y]);
							sign_sum += signnum(Gm[x]*Gn[y]) * sigma_derivative[y*size+x];
						}
					}
				
					for(int y = 0; y < size; y++)
						for(int x = 0; x < size; x++)
							sigma_derivative[y*size+x] = (1 / volume) * sigma_derivative[y*size+x] - Gm[x]*Gn[y] / pow(volume,2) * sign_sum;
				}
				
				// Free heap memory
				delete[] Gm;
				delete[] Gn;
				delete[] dGm_dsigma;
				delete[] dGn_dsigma;
			
				return sigma_derivative;
			}
		}
		
		// Anisotropic
		double** filters(int order, double sigma_x, double sigma_y, bool normalize=true)
		{
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_x <= 0)
			{
	   	 		std::cout << "Error: sigma_x should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			if(sigma_y <= 0)
			{
	   	 		std::cout << "Error: sigma_y should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
			
			// Initialize the output array
			double** filters = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					filters[index++] = internal::gaussian_derivative_2D(m, n, sigma_x, sigma_y, normalize);
				}
			}
			
			return filters;
		}
		
		double** sigmaX_derivatives(int order, double sigma_x, double sigma_y, bool normalize=true)
		{
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_x <= 0)
			{
	   	 		std::cout << "Error: sigma_x should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_y <= 0)
			{
	   	 		std::cout << "Error: sigma_y should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
		
			// Initialize the output array
			double** sigma_derivatives = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					sigma_derivatives[index++] = internal::gaussian_derivative_2D_sigmaX_derivative(m, n, sigma_x, sigma_y, normalize);
				}
			}
			
			return sigma_derivatives;
		}
		
		double** sigmaY_derivatives(int order, double sigma_x, double sigma_y, bool normalize=true)
		{
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_x <= 0)
			{
	   	 		std::cout << "Error: sigma_x should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma_y <= 0)
			{
	   	 		std::cout << "Error: sigma_y should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
		
			// Initialize the output array
			double** sigma_derivatives = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					sigma_derivatives[index++] = internal::gaussian_derivative_2D_sigmaY_derivative(m, n, sigma_x, sigma_y, normalize);
				}
			}
			
			return sigma_derivatives;
		}
		
		// Isotropic
		double** filters(int order, double sigma, bool normalize=true)
		{
			return filters(order, sigma, sigma, normalize);
		}
		
		double** sigma_derivatives(int order, double sigma, bool normalize=true)
		{
			// Output of the isotropic case shoudl be the same as
			// filter_bank_sigmaX_derivatives + filter_bank_sigmaY_derivatives
			
			// Validate the input
	    	if(order < 0)
			{
	   	 		std::cout << "Error: order should be non-negative ( >= 0 )!" << std::endl;
				return NULL;
			}
	    	if(sigma <= 0)
			{
	   	 		std::cout << "Error: sigma should be positive and non zero ( > 0 )!" << std::endl;
				return NULL;
			}
			
			// Compute the total number of filters
	    	int number_of_filters = compute_number_of_filters(order);
		
			// Initialize the output array
			double** sigma_derivatives = new double*[number_of_filters];
			
			// Loop over the different orders of derivatives in x- and y-direction
	   	 	int index = 0;
	   		for(int total_order = 0; total_order < order+1; total_order++)
	   	 	{
				for(int n = 0; n < total_order+1; n++)
		   	 	{
					int m = total_order - n;
					sigma_derivatives[index++] = internal::gaussian_derivative_2D_sigma_derivative(m, n, sigma, normalize);
				}
			}
			
			return sigma_derivatives;
		}
	}
}