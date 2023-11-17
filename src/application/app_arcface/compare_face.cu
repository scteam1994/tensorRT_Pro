#include <common/cuda_tools.hpp>
#include <stdio.h>
namespace Face {
	void __global__ compare_face_kernel(
		const float* feature1,//1*512
		const float* bank,    //n*512
		float* res,			  //n*1
		int bank_size,
		int feature_size
	) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		if (col < feature_size && row < bank_size) {
			float sum = 0;
			for (int i = 0; i < feature_size; i++) {
				sum += (feature1[i] - bank[row * feature_size + i]) * (feature1[i] - bank[row * feature_size + i]);
			}
			res[row] = sum;

		}
	}
	void compare_face(const float* feature1, const float* bank, float* res, int bank_size, int feature_size) {
		int block_size = 32;
		dim3 block(block_size, block_size);
		dim3 grid((feature_size + block.x - 1) / block.x, (bank_size + block.y - 1) / block.y);
		checkCudaKernel(compare_face_kernel <<<grid, block >>> (feature1, bank, res, bank_size, feature_size));

	}

}