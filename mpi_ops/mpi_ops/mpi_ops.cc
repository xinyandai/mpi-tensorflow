#include "mpi_ops.h"


MPI_Datatype GetMPIDataType(const tensorflow::Tensor tensor) {
    switch (tensor.dtype()) {
        case tensorflow::DT_UINT8:
            return MPI_UINT8_T;
        case tensorflow::DT_INT8:
            return MPI_INT8_T;
        case tensorflow::DT_UINT16:
            return MPI_UINT16_T;
        case tensorflow::DT_INT16:
            return MPI_INT16_T;
        case tensorflow::DT_INT32:
            return MPI_INT32_T;
        case tensorflow::DT_INT64:
            return MPI_INT64_T;
        case tensorflow::DT_FLOAT:
            return MPI_FLOAT;
        case tensorflow::DT_DOUBLE:
            return MPI_DOUBLE;
        default:
            return 0;
    }
}