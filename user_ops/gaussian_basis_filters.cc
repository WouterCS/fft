#include "./tensorflow/include/tensorflow/core/framework/op.h"
#include "./tensorflow/include/tensorflow/core/framework/op_kernel.h"

#include "njet.h"

using namespace tensorflow;

// Register the operation in the framework
REGISTER_OP("GaussianBasisFilters")
	.Attr("grid: {'continuous', 'discrete'}")
    .Attr("order: int >= 0")
	.Attr("normalize: bool = true")
    .Input("sigma_x: float32")
	.Input("sigma_y: float32")
    .Output("filters: float32")
    .Doc(R"doc(Creates a set of gaussian basis filters up to a certain order with a certain sigma)doc");

// Implement a CPU kernel
class GaussianBasisFiltersOp : public OpKernel
{
 public:
  explicit GaussianBasisFiltersOp(OpKernelConstruction* context) : OpKernel(context)
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
    const Tensor& input_tensor1 = context->input(0);
	const Tensor& input_tensor2 = context->input(1);
	
	// Validate input tensors are scalar
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor1.shape()), errors::InvalidArgument("GaussianBasisFilters expects a scalar for sigmaX."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor2.shape()), errors::InvalidArgument("GaussianBasisFilters expects a scalar for sigmaY."));
	
	// Fetch the sigmas from the input tensors
	double sigmaX = input_tensor1.scalar<float>()();
	double sigmaY = input_tensor2.scalar<float>()();
	
	// Compute the range and kernel size
	int rangeX, rangeY;
	int sizeX, sizeY;
	njet::compute_range_and_size(sigmaX, rangeX, sizeX);
	njet::compute_range_and_size(sigmaY, rangeY, sizeY);
	
	// Compute the filters
	double** temp;
	if(grid_ == "continuous")
	 	temp = njet::continuous::filters(order_, sigmaX, sigmaY, normalize_);
	else if (grid_ == "discrete")
	 	temp = njet::discrete::filters(order_, sigmaX, sigmaY, normalize_);
	
    // Create an output tensor
    Tensor* output_tensor = nullptr;
	TensorShape output_shape = TensorShape({sizeY, sizeX, N_});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
	
	// Fetch the output tensor
	auto filters = output_tensor->tensor<float, 3>();
	
	// Copy the filters to the output tensor
	for(int y = 0; y < sizeY; y++)
		for(int x = 0; x < sizeX; x++)
			for(int i = 0; i < N_; i++)
				filters(y, x, i) = (float)temp[i][y*sizeX+x];
	
	// Free heap memory
	delete[] temp;
  }

  private:
   string grid_;
   int order_;
   bool normalize_;
   int N_;
};

// Register a CPU kernel to the operation
REGISTER_KERNEL_BUILDER(Name("GaussianBasisFilters").Device(DEVICE_CPU), GaussianBasisFiltersOp);