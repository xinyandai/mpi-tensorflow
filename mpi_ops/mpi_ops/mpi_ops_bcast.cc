#include "mpi_ops.h"

using namespace tensorflow;


REGISTER_OP("TfBroadcast")
        .Attr("root: int")
        .Attr("size: int")
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


class TfBroadcastOp : public OpKernel {
public:
    explicit TfBroadcastOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("root", &root_));
    }
    void Compute(OpKernelContext* context) override {

        // Grab the input tensor

        Tensor input_tensor = context->input(0);
        context->forward_ref_input_to_ref_output(0, 0);

        MPI_Datatype datatype = GetMPIDataType(input_tensor);

        MPI_Bcast(
                (void*)input_tensor.tensor_data().data(),
                (int)input_tensor.NumElements(),
                datatype,
                root_,
                MPI_COMM_WORLD);

        context->set_output(0, input_tensor);
    }

private:
    int root_;
};


REGISTER_KERNEL_BUILDER(Name("TfBroadcast").Device(DEVICE_CPU), TfBroadcastOp);
REGISTER_KERNEL_BUILDER(Name("TfBroadcast").Device(DEVICE_GPU), TfBroadcastOp);
