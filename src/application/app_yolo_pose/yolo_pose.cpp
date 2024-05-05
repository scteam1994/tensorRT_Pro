#include "yolo_pose.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/monopoly_allocator.hpp>
#include <common/preprocess_kernel.cuh>
#include "yolo_decode_pose.h"
namespace YoloPose {

    struct AffineMatrix {
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& image_size, const cv::Rect& box, const cv::Size& net_size) {
            Rect box_ = box;
            if (box_.width == 0 || box_.height == 0) {
                box_.width = image_size.width;
                box_.height = image_size.height;
                box_.x = 0;
                box_.y = 0;
            }

            float rate = 0.0f;
            float pad_width = box_.width * (1 + 2 * rate);
            float pad_height = box_.height * (1 + 2 * rate);
            float scale = min(net_size.width / pad_width, net_size.height / pad_height);
            i2d[0] = scale;  i2d[1] = 0;      i2d[2] = -(box_.x + box_.width * 0.5) * scale + net_size.width * 0.5;
            i2d[3] = 0;      i2d[4] = scale;  i2d[5] = -(box_.y + box_.height * 0.5) * scale + net_size.height * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat() {
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    static tuple<float, float> affine_project(float x, float y, float* pmatrix) {

        float newx = x * pmatrix[0] + y * pmatrix[1] + pmatrix[2];
        float newy = x * pmatrix[3] + y * pmatrix[4] + pmatrix[5];
        return make_tuple(newx, newy);
    }

    using ControllerImpl = InferController
        <
        Input,                     // input
        vector<vector<Point3f>>,           // output
        tuple<string, int>,        // start param
        AffineMatrix               // additional
        >;
    class InferImpl : public Infer, public ControllerImpl {
    public:
        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl() {
            TRT::set_device(gpu_);
            stop();
        }


        virtual bool startup(const string& file, int gpuid, float confidence_threshold, float nms_threshold, NMSMethod nms_method, int max_objects) 
        {
            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            confidence_threshold_ = confidence_threshold;
            nms_threshold_ = nms_threshold;
            nms_method_ = nms_method;
            max_objects_ = max_objects;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }
        virtual void worker(promise<bool>& result) override {

            string file = get<0>(start_param_);
            int gpuid = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if (engine == nullptr) {
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            const int MAX_IMAGE_BBOX = max_objects_;
            const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input = engine->tensor("images");
            auto output = engine->tensor("output0");
            int src_kpbox_len = output->size(2);

            input_width_ = input->size(3);
            input_height_ = input->size(2);
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_ = engine->get_stream();
            gpu_ = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes + 17kpt*3 ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * (NUM_BOX_ELEMENT+51)).to_gpu();

            vector<Job> fetch_jobs;
            while (get_jobs_and_wait(fetch_jobs, max_batch_size)) {

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                    auto& job = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();

                    if (mono->get_stream() != stream_) {
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }

                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                output_array_device.to_gpu(false);
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {

                    auto& job = fetch_jobs[ibatch];
                    float* image_based_output = output->gpu<float>(ibatch);
                    float* output_array_ptr = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker_yolo_pose(image_based_output, output->size(1), src_kpbox_len, confidence_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);

                    if (nms_method_ == NMSMethod::FastGPU) {
                        nms_kernel_invoker_yolo_pose(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                }

                output_array_device.to_cpu();
                for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto& job = fetch_jobs[ibatch];
                    auto& image_based_keypoints = job.output;
                    for (int i = 0; i < count; ++i) {
                        vector<Point3f> keypoints;
                        float* pbox = parray + 1 + i * (NUM_BOX_ELEMENT + 51);
                        int label = pbox[5];
                        int keepflag = pbox[6];
                        if (keepflag == 1) {

                            // output -> batch x 19 x 3
                            keypoints.resize(2 + num_kpt);
                            keypoints[0].x = pbox[0];
                            keypoints[0].y = pbox[1];
                            keypoints[0].z = pbox[4];
                            keypoints[1].x = pbox[2];
                            keypoints[1].y = pbox[3];
                            keypoints[1].z = pbox[4];
                            for (int j = 0; j < num_kpt; ++j) {
                                float x = pbox[NUM_BOX_ELEMENT + j * 3 + 0];
                                float y = pbox[NUM_BOX_ELEMENT + j * 3 + 1];
                                float confidence = pbox[NUM_BOX_ELEMENT + j * 3 + 2];
                                auto& output_point = keypoints[2 + j];
                                tie(output_point.x, output_point.y) = affine_project(x, y, job.additional.d2i);
                                output_point.z = confidence;
                            }
                            image_based_keypoints.push_back(keypoints);
                        }
                    }
                    job.pro->set_value(image_based_keypoints);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            engine.reset();
            INFO("Engine destroy.");
        }
        virtual shared_future<vector<vector<Point3f>>> commit(const Input& input) override {
            return ControllerImpl::commit(input);
        }

        virtual vector<shared_future<vector<vector<Point3f>>>> commits(const vector<Input>& inputs) override {
            return ControllerImpl::commits(inputs);
        }

        virtual bool preprocess(Job& job, const Input& input) override {

            if (tensor_allocator_ == nullptr) {
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if (job.mono_tensor == nullptr) {
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto& tensor = job.mono_tensor->data();
            if (tensor == nullptr) {
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());
            }

            auto& image = get<0>(input);
            auto& box = get<1>(input);
            Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), box, input_size);

            tensor->set_stream(stream_);
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            size_t size_matrix = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace = tensor->get_workspace();
            uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image + size_matrix);
            float* affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device = gpu_workspace + size_matrix;
            checkCudaRuntime(cudaMemcpyAsync(image_device, image.data, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, job.additional.d2i, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));
            auto normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device, image.cols * 3, image.cols, image.rows,
                tensor->gpu<float>(), input_width_, input_height_,
                affine_matrix_device, 127,
                normalize, stream_
            );
            return true;
        }

    private:
        int input_width_ = 0;
        int input_height_ = 0;
        int gpu_ = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_ = 0;
        int max_objects_ = 1008;
        TRT::CUStream stream_ = nullptr;
        NMSMethod nms_method_ = NMSMethod::FastGPU;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
        int num_kpt = 17;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects
    ) 
    {
        shared_ptr<InferImpl> instance(new InferImpl());
        if (!instance->startup(
            engine_file, gpuid, confidence_threshold,
            nms_threshold, nms_method, max_objects)
            ) 
        {
            instance.reset();
        }
        return instance;
    }
};