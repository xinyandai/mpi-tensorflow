#ifndef __MPI_OPS_HEADER__
#define __MPI_OPS_HEADER__
//header file content

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "mpi.h"

MPI_Datatype GetMPIDataType(const tensorflow::Tensor tensor);

#endif