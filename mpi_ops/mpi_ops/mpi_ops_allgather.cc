#include "mpi_ops.h"

using namespace tensorflow;

REGISTER_OP("TfAllgather")
        .Attr("size: int")
        .Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
        .Attr("P: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
        .Input("input: T")
        .Input("pre_node: P")
        .Output("output: T")
        .SetShapeFn(
                [](::tensorflow::shape_inference::InferenceContext* c) {
                    int cluster_size, cluster_rank, cluster_root;
                    c->GetAttr("size", &cluster_size);
                    ::tensorflow::shape_inference::ShapeHandle output;
                    ::tensorflow::shape_inference::ShapeHandle shape_size = c->Vector(
                            ::tensorflow::shape_inference::DimensionOrConstant(cluster_size));
                    TF_RETURN_IF_ERROR(c->Concatenate(shape_size, c->input(0), &output));
                    c->set_output(0, output);
                    return Status::OK();
                }
        );



class TfAllgatherOp : public OpKernel {
public:
    explicit TfAllgatherOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("size", &size_));
    }
    void Compute(OpKernelContext* context) override {

        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        MPI_Datatype datatype = GetMPIDataType(input_tensor);
        int count = (int)input_tensor.NumElements();

        void * recv_buf = NULL;
        // Create an output tensor
        Tensor* output_tensor = NULL;

        TensorShape tensor_shape;
        tensor_shape.AddDim(size_);
        tensor_shape.AppendShape(input_tensor.shape());

        OP_REQUIRES_OK(
                context,
                context->allocate_output(0, tensor_shape, &output_tensor)
        );

        recv_buf = (void*)output_tensor->tensor_data().data();

        MPI_Allgather(
                (void*)input_tensor.tensor_data().data(),
                count,
                datatype,
                recv_buf,
                count,
                datatype,
                MPI_COMM_WORLD
        );
    }

private:
    int size_;
};



REGISTER_KERNEL_BUILDER(Name("TfAllgather").Device(DEVICE_CPU), TfAllgatherOp);
REGISTER_KERNEL_BUILDER(Name("TfAllgather").Device(DEVICE_GPU), TfAllgatherOp);
