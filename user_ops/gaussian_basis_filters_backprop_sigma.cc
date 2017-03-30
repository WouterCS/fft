#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "njet.h"

using namespace tensorflow;

// Register the operation in the framework
REGISTER_OP("GaussianBasisFiltersBackpropSigma")
	.Attr("grid: {'continuous', 'discrete'}")
	.Attr("order: int >= 0")
	.Attr("normalize: bool = true")
    .Input("sigma_x: float32")
    .Input("sigma_y: float32")
	.Input("grad: float32")
    .Output("sigma_x_grad: float32")
	.Output("sigma_y_grad: float32")
    .Doc(R"doc(Computes the gradient of the output (a set of gaussian basis filters) w.r.t. the input (sigma) evaluated at the input sigma)doc");

// Implement a CPU kernel
class GaussianBasisFiltersBackpropSigmaOp : public OpKernel
{
 public:
  explicit GaussianBasisFiltersBackpropSigmaOp(OpKernelConstruction* context) : OpKernel(context)
  {
      // Get attributes
      OP_REQUIRES_OK(context, context->GetAttr("grid", &grid_));
      OP_REQUIRES_OK(context, context->GetAttr("order", &order_));
      OP_REQUIRES_OK(context, context->GetAttr("normalize", &normalize_));

      // Check that order is positive
      OP_REQUIRES(context, order_ >= 0, errors::InvalidArgument("Order should be non-negative ( >= 0 )! Got ", order_));

	  // Compute the number of filters
	  N_ = njet::compute_number_of_filters(order_);
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensors
    const Tensor& input_tensor0 = context->input(0);
    const Tensor& input_tensor1 = context->input(1);
    const Tensor& input_tensor2 = context->input(2);
	
	// Validate input tensor is scalar
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor0.shape()), errors::InvalidArgument("GaussianBasisFilters expects a scalar for sigmaX."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor1.shape()), errors::InvalidArgument("GaussianBasisFilters expects a scalar for sigmaY."));
    
	// Fetch the sigma from the input tensor
	double sigmaX = input_tensor0.scalar<float>()();
	double sigmaY = input_tensor1.scalar<float>()();
	
	// Fetch the gradients from the input tensor
	auto grad = input_tensor2.tensor<float, 3>();
	
	// Compute the range and kernel size
	int rangeX, rangeY;
	int sizeX, sizeY;
	njet::compute_range_and_size(sigmaX, rangeX, sizeX);
	njet::compute_range_and_size(sigmaY, rangeY, sizeY);
	
	//std::cout << "sigma = " << sigma << " N = " << N_ << " grid = " << grid_ << " size = " << size << " order = " << order_ << " norm = " << normalize_ << std::endl;
	
	// Compute the derivatives
	double** tempX;
	double** tempY;
	if(grid_ == "continuous")
	{
	 	tempX = njet::continuous::sigmaX_derivatives(order_, sigmaX, sigmaY, normalize_);
	 	tempY = njet::continuous::sigmaY_derivatives(order_, sigmaY, sigmaY, normalize_);
	}
	else if (grid_ == "discrete")
	{
	 	tempX = njet::discrete::sigmaX_derivatives(order_, sigmaX, sigmaY, normalize_);
	 	tempY = njet::discrete::sigmaY_derivatives(order_, sigmaY, sigmaY, normalize_);
	}
	
    // Create an output tensors
    Tensor* output_tensor0 = nullptr;
	TensorShape output_shape0 = TensorShape();
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_tensor0));
	Tensor* output_tensor1 = nullptr;
	TensorShape output_shape1 = TensorShape();
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_tensor1));
	
	// Fetch the output tensor
	float& sigmaX_grad = output_tensor0->scalar<float>()();
	float& sigmaY_grad = output_tensor1->scalar<float>()();
	
	// Copy the filters to the output tensor
	sigmaY_grad = 0;
	sigmaX_grad = 0;
	for(int y = 0; y < sizeY; y++)
	{
		for(int x = 0; x < sizeX; x++)
		{
			for(int i = 0; i < N_; i++)
			{
				sigmaX_grad += (float)tempX[i][y*sizeX+x] * grad(y, x, i);
				sigmaY_grad += (float)tempY[i][y*sizeX+x] * grad(y, x, i);
			}
		}
	}
	
	// Free heap memory
	delete[] tempX;
	delete[] tempY;
  }

  private:
   string grid_;
   int order_;
   bool normalize_;
   int N_;
};

// Register a CPU kernel to the operation
REGISTER_KERNEL_BUILDER(Name("GaussianBasisFiltersBackpropSigma").Device(DEVICE_CPU), GaussianBasisFiltersBackpropSigmaOp);