#pragma once
#include <common/cuda_tools.hpp>
namespace YoloPose
{
	void decode_kernel_invoker_yolo_pose(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream);
	void nms_kernel_invoker_yolo_pose(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);
}
