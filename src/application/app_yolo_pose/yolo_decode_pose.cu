#include"yolo_decode_pose.h"
namespace YoloPose{

    const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence,class, keepflag
    const int DSTWIDTH = 58;
    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __global__ void decode_yolov8_pose_device_kernel(float* predict, int num_bboxes, int src_kpbox_len, float confidence_threshold, float* invert_affine_matrix, float* dst, int max_objects)
    {

        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= num_bboxes)
        {
            return;
        }

        float* pitem     = predict + src_kpbox_len * position;
        float confidence = pitem[4];
        if(confidence < confidence_threshold)
        {
            return;
        }

        int index = atomicAdd(dst, 1);
        if(index >= max_objects)
        {
            return;
        }

        float cx         = *pitem++;
        float cy         = *pitem++;
        float width      = *pitem++;
        float height     = *pitem++;
        float left   = cx - width * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width * 0.5f;
        float bottom = cy + height * 0.5f;
        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float* pout_item = dst + 1 + index * DSTWIDTH;  //NUM_BOX_ELEMENT
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = 0; //label
        *pout_item++ = 1; // 1 = keep, 0 = ignore
        memcpy(pout_item, pitem + 1, (DSTWIDTH - 7) * sizeof(float));

    }


    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom
    ){

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;
        
        // left, top, right, bottom, confidence, class, keepflag
        float* pcurrent = bboxes + 1 + position * DSTWIDTH;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * DSTWIDTH;
            if (i == position || pcurrent[5] != pitem[5])
            {
                continue;
            }

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && i < position)
                {
                    continue;
                }

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold){
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    } 

 //   __global__ void nms_fast_kernel(int topK, int batch_size, float iou_thresh,
	//		float* src, int srcWidth, int srcHeight, int srcArea)
	//{
	//	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	//	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	//	if (dy >= batch_size)
	//	{
	//		return;
	//	}
	//	float* p_count = src + dy * srcArea;
	//	int count = min(int(p_count[0]), topK);
	//	if (dx >= count)
	//	{
	//		return;
	//	}
	//	float* pcurrent = src + dy * srcArea + 1 + dx * srcWidth;
	//	for (int i = 0; i < count; ++i)
	//	{
	//		float* pitem = src + dy * srcArea + 1 + i * srcWidth;
	//		if (i == dx || pcurrent[5] != pitem[5])
	//			continue;
	//		if (pitem[4] >= pcurrent[4])
	//		{
	//			if (pitem[4] == pcurrent[4] && i < dx)
	//				continue;
	//			float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
	//				pitem[0], pitem[1], pitem[2], pitem[3]);
	//			if (iou > iou_thresh)
	//			{
	//				pcurrent[6] = 0;
	//				return;
	//			}
	//		}
	//	}
	//}

    void decode_kernel_invoker_yolo_pose(float* predict, int num_bboxes, int src_kpbox_len, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_yolov8_pose_device_kernel <<<grid, block, 0, stream>>>(predict, num_bboxes, src_kpbox_len, confidence_threshold, invert_affine_matrix, parray, max_objects));
    }

    void nms_kernel_invoker_yolo_pose(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(max_objects);
        auto block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }

};