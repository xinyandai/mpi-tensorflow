#include "mpi_ops.h"

using namespace tensorflow;


REGISTER_OP("TfAllreduce")
        .Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
        .Attr("P: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
        .Input("buffer: T")
        .Input("pre_node: P")
        .Output("output: T")
        .SetShapeFn(
            [](::tensorflow::shape_inference::InferenceContext* c) {
                c->set_output(0, c->input(0));
                return Status::OK();
            }
        );


class TfAllreduceOp : public OpKernel {
public:
    explicit TfAllreduceOp(OpKernelConstruction* context) : OpKernel(context) {
    }
    void Compute(OpKernelContext* context) override {

        // Grab the input tensor
        Tensor input_tensor = context->input(0);
        context->forward_ref_input_to_ref_output(0, 0);

        MPI_Datatype datatype = GetMPIDataType(input_tensor);

        MPI_Allreduce(
                MPI_IN_PLACE,
                (void*)input_tensor.tensor_data().data(),
                (int)input_tensor.NumElements(),
                datatype,
                MPI_SUM,
                MPI_COMM_WORLD);

        context->set_output(0, input_tensor);
    }
};


REGISTER_KERNEL_BUILDER(Name("TfAllreduce").Device(DEVICE_CPU), TfAllreduceOp);
REGISTER_KERNEL_BUILDER(Name("TfAllreduce").Device(DEVICE_GPU), TfAllreduceOp);
