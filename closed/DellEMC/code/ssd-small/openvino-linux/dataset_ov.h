#ifndef DATASET_H__
#define DATASET_H__

#include <boost/property_tree/json_parser.hpp>

// loadgen
#include "loadgen.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "bindings/c_api.h"

#include <inference_engine.hpp>
#include <ie_blob.h>

#include "item_ov.h"

using namespace InferenceEngine;
using namespace std;
using namespace cv;

class Dataset {
public:
    Dataset() {
    }

    ~Dataset() {
    }

    int getItemCount() {
        return image_list_.size();
    }

    void loadQuerySamples(const mlperf::QuerySampleIndex* samples,
            size_t num_samples) {
        if (image_list_inmemory_.size() == 0) {
            if ((this->settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (this->settings_.scenario == mlperf::TestScenario::Server)) {
                this->image_list_inmemory_.resize(total_count_);
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::Offline) {
                image_list_inmemory_.resize(num_samples);
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_.resize(num_samples);
                sample_list_inmemory_.resize(num_samples);
            }
        }

        mlperf::QuerySampleIndex sample;

        handle = (new float[num_samples * num_channels_ * image_height_
                * image_width_]);

        for (uint i = 0; i < num_samples; ++i) {
            cv::Mat processed_image;
            sample = (*samples);
            samples++;

            std::string image_path;

            if (dataset_.compare("imagenet") == 0) {
                image_path = this->datapath_ + "/" + image_list_[sample];
            } else if (dataset_.compare("coco") == 0) {
                image_path = this->datapath_ + "/val2017/"
                        + image_list_[sample];
            }

            auto image = cv::imread(image_path);

            if (image.empty()) {
                throw std::logic_error("Invalid image at path: " + i);
            }

            /////////
            if (this->workload_.compare("resnet50") == 0) {
                preprocessVGG(&image, &processed_image);
            } else if (this->workload_.compare("mobilenet") == 0) {
                preprocessMobilenet(&image, &processed_image);
            } else if (this->workload_.compare("ssd-mobilenet") == 0) {
                preprocessSSDMobilenet(&image, &processed_image);
            } else if (this->workload_.compare("ssd-resnet34") == 0) {
                preprocessSSDResnet(&image, &processed_image);
            }

            processed_image.copyTo(image);
            ////////
            size_t image_size = (image_height_ * image_width_);

            InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32,
                    { 1, num_channels_, image_height_, image_width_ },
                    InferenceEngine::Layout::NCHW);

            auto input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type
                    > (tDesc, ((((float *) handle)
                            + (i * image_size * num_channels_))), image_size
                            * num_channels_);

            for (size_t image_id = 0; image_id < 1; ++image_id) {

                for (size_t pid = 0; pid < image_size; pid++) {

                    for (size_t ch = 0; ch < num_channels_; ++ch) {
                        input->data()[image_id * image_size * num_channels_
                                + ch * image_size + pid] = image.at < cv::Vec3f
                                > (pid)[ch];
                    }
                }
            }

            if (settings_.scenario == mlperf::TestScenario::Offline) {
                image_list_inmemory_[i] = input;
                //image_list_inmemory_.push_back(input);
            } else if ((settings_.scenario == mlperf::TestScenario::SingleStream)
                    || (settings_.scenario == mlperf::TestScenario::Server)) {
                image_list_inmemory_[sample] = input;
            } else if (this->settings_.scenario
                    == mlperf::TestScenario::MultiStream) {
                image_list_inmemory_[i] = input;
                sample_list_inmemory_[i] = sample;
            }
        }
    }

    void unloadQuerySamples(const mlperf::QuerySampleIndex* samples,
            size_t num_samples) {
        if (num_samples > 0) {
            mlperf::QuerySampleIndex sample;

            for (uint i = 0; i < num_samples; ++i) {
                sample = (*samples);
                samples++;
                image_list_inmemory_.erase(
                        image_list_inmemory_.begin() + sample);
            }
        } else {
            image_list_inmemory_.clear();
        }

        image_list_inmemory_.clear();
        delete [] handle;
        //this->image_list_inmemory_.resize(this->total_count_);
    }

    void getSamples(const mlperf::QuerySampleIndex* samples, Blob::Ptr* data,
            int* label) {
        *data = image_list_inmemory_[samples[0]];
        if (dataset_.compare("imagenet") == 0) {
            *label = label_list_[samples[0]];
        }
    }

    void getSamplesBatched(const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {

        size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {
                (bs), num_channels_, image_height_, image_width_ },
                InferenceEngine::Layout::NCHW);
        TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr input;
        for (int i = 0; i < num_batches; ++i) {
            auto start = (i * bs) % perf_count_;

            // TODO: rps: Fix: Assuming batch_size is a multiple of perf_sample_count
            input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type
                    > (tDesc, image_list_inmemory_[start]->buffer(), (bs
                            * image_size * num_channels_));

            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;
            for (size_t j = (i * bs); j < (unsigned) ((i * bs) + bs); ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(input, ids, idxs));
        }
    }

    void getSamplesBatchedServer(
            const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {

        size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {
                (bs), num_channels_, image_height_, image_width_ },
                InferenceEngine::Layout::NCHW);

        for (int i = 0; i < num_batches; ++i) {
            auto input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type > (tDesc);
            input->allocate();

            // TODO: rps: Fix: Assuming batch_size is a multiple of perf_sample_count
            for (size_t k = 0; k < bs; ++k) {
                auto start = samples[k];

                std::memcpy((input->data() + (k * image_size * num_channels_)),
                        image_list_inmemory_[start]->buffer().as<
                                PrecisionTrait<Precision::FP32>::value_type *>(),
                        (image_size * num_channels_ * sizeof(float)));
            }
            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;
            for (size_t j = 0; j < (unsigned) bs; ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(input, ids, idxs));
        }
    }

    void getSamplesBatchedMultiStream(
            const std::vector<mlperf::QuerySampleIndex> samples,
            std::vector<mlperf::ResponseId> queryIds, size_t bs,
            int num_batches, std::vector<Item> &items) {
        size_t image_size = (image_height_ * image_width_);
        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32, {
                (bs), num_channels_, image_height_, image_width_ },
                InferenceEngine::Layout::NCHW);

        // find sample
        std::vector<mlperf::QuerySampleIndex>::iterator it = std::find(
                sample_list_inmemory_.begin(), sample_list_inmemory_.end(),
                samples[0]);
        if (!(it != std::end(sample_list_inmemory_))) {
            std::cout
                    << "getSamplesBatchedMultiStream: ERROR: No sample found\n ";
        }

        int index = std::distance(sample_list_inmemory_.begin(), it);
        auto start = index;
        auto query_start = 0;
        for (int i = 0; i < num_batches; ++i) {
            TBlob<PrecisionTrait<Precision::FP32>::value_type>::Ptr input;

            // TODO: rps: Fix: Assuming batch_size is a multiple of perf_sample_count
            input = make_shared_blob < PrecisionTrait < Precision::FP32
                    > ::value_type
                    > (tDesc, image_list_inmemory_[start]->buffer(), (bs
                            * image_size * num_channels_));

            std::vector < mlperf::QuerySampleIndex > idxs;
            std::vector < mlperf::ResponseId > ids;
            for (size_t j = (query_start); j < (unsigned) (query_start + bs);
                    ++j) {
                ids.push_back(queryIds[j]);
                idxs.push_back(samples[j]);
            }

            items.push_back(Item(input, ids, idxs));

            start = start + bs;
            query_start = query_start + bs;
        }
    }

    // Preprocessing routines
    void centerCrop(cv::Mat* image, int out_height, int out_width,
            cv::Mat* cropped_image) {
        int width = (*image).cols;
        int height = (*image).rows;
        int left = int((width - out_width) / 2);
        //int right = int((width + out_width) / 2);
        int top = int((height - out_height) / 2);
        //int bottom = int((height + out_height) / 2);
        cv::Rect customROI(left, top, out_width, out_height);

        (*cropped_image) = (*image)(customROI);
    }

    void resizeWithAspectratio(cv::Mat* image, cv::Mat* resized_image,
            int out_height, int out_width, int interpol, float scale = 87.5) {
        int width = (*image).cols;
        int height = (*image).rows;
        int new_height = int(100. * out_height / scale);
        int new_width = int(100. * out_width / scale);

        int h, w = 0;
        if (height > width) {
            w = new_width;
            h = int(new_height * height / width);
        } else {
            h = new_height;
            w = int(new_width * width / height);
        }

        cv::resize((*image), (*resized_image), cv::Size(w, h), interpol);
    }

    void preprocessSSDMobilenet(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, float_image, u8_image;

        image->convertTo(float_image, CV_32F);
        if (num_channels_ < 3) {
            cv::cvtColor(float_image, img, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(float_image, img, cv::COLOR_BGR2RGB);
        }

        cv::resize((img), (resized_image),
                cv::Size(image_width_, image_height_), cv::INTER_LINEAR);

        resized_image.copyTo(*processed_image);
    }

    void preprocessSSDResnet(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, sub_image, float_image, div_image,
                std_image;

        image->convertTo(float_image, CV_32F);
        if (num_channels_ < 3) {
            cv::cvtColor(float_image, img, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(float_image, img, cv::COLOR_BGR2RGB);
        }

        cv::resize((img), (resized_image),
                cv::Size(image_width_, image_height_), cv::INTER_LINEAR);

        cv::Mat div(image_height_, image_width_, CV_32FC3,
                cv::Scalar(255.0, 255.0, 255.0));
        cv::divide(resized_image, div, div_image);

        cv::Mat means(image_height_, image_width_, CV_32FC3,
                cv::Scalar(0.485, 0.456, 0.406));
        cv::subtract(div_image, means, sub_image);

        cv::Mat std(image_height_, image_width_, CV_32FC3,
                cv::Scalar(0.229, 0.224, 0.225));
        cv::divide(sub_image, std, std_image);
        std_image.copyTo(*processed_image);
    }

    void preprocessVGG(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, cropped_image, float_image, norm_image,
                sub_image;
        cv::cvtColor(*image, img, cv::COLOR_BGR2RGB);

        resizeWithAspectratio(&img, &resized_image, image_height_, image_width_,
                cv::INTER_AREA);

        centerCrop(&resized_image, image_height_, image_width_, &cropped_image);

        cropped_image.convertTo(float_image, CV_32FC3);
        int width = cropped_image.cols;
        int height = cropped_image.rows;

        cv::Mat means(height, width, CV_32FC3,
                cv::Scalar(123.68, 116.78, 103.94));
        cv::subtract(float_image, means, sub_image);
        sub_image.copyTo(*processed_image);
    }

    void preprocessMobilenet(cv::Mat* image, cv::Mat* processed_image) {
        cv::Mat img, resized_image, cropped_image, float_image, norm_image,
                sub_image;
        cv::cvtColor(*image, img, cv::COLOR_BGR2RGB);

        resizeWithAspectratio(&img, &resized_image, image_height_, image_width_,
                cv::INTER_LINEAR);

        centerCrop(&resized_image, image_height_, image_width_, &cropped_image);

        cropped_image.convertTo(float_image, CV_32FC3);
        int width = cropped_image.cols;
        int height = cropped_image.rows;

        cv::Mat div(height, width, CV_32FC3, cv::Scalar(255.0, 255.0, 255.0));
        cv::divide(float_image, div, float_image);

        cv::Mat sub(height, width, CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
        cv::subtract(float_image, sub, float_image);

        cv::Mat mul(height, width, CV_32FC3, cv::Scalar(2.0, 2.0, 2.0));
        cv::multiply(float_image, mul, float_image);

        float_image.copyTo(*processed_image);
    }

//protected: // TODO: rps: Fix this back to protected
public:
    std::vector<string> image_list_;
    std::vector<int> label_list_;
    std::vector<Blob::Ptr> image_list_inmemory_;
    std::vector<mlperf::QuerySampleIndex> sample_list_inmemory_;
    size_t image_width_;
    size_t image_height_;
    size_t num_channels_;
    string datapath_;
    string image_format_;
    size_t total_count_;
    size_t perf_count_;
    bool need_transpose_ = false;
    string workload_ = "resnet50";
    mlperf::TestSettings settings_;
    string dataset_;
    float * handle;
}
;

class Imagenet: public Dataset {
public:
    Imagenet(mlperf::TestSettings settings, int image_width, int image_height,
            int num_channels, string datapath, string image_format,
            int total_count, int perf_count, string workload, string dataset) {
        this->image_width_ = image_width;
        this->image_height_ = image_height;
        this->num_channels_ = num_channels;
        this->datapath_ = datapath;
        this->image_format_ = image_format;
        this->total_count_ = total_count;
        this->perf_count_ = perf_count;
        this->settings_ = settings;

        this->workload_ = workload;
        this->dataset_ = dataset;

        this->need_transpose_ = image_format == "NHWC" ? false : true;

        string image_list_file = datapath + "/val_map.txt";
        std::ifstream imglistfile;
        imglistfile.open(image_list_file, std::ios::binary);

        std::string line, image_name, label;
        if (imglistfile.is_open()) {
            while (getline(imglistfile, line)) {
                std::regex ws_re("\\s+");
                std::vector < std::string
                        > img_label { std::sregex_token_iterator(line.begin(),
                                line.end(), ws_re, -1), { } };
                label = img_label.back();
                image_name = img_label.front();
                img_label.clear();

                this->image_list_.push_back(image_name);
                this->label_list_.push_back(stoi(label));

                // limit dataset
                if (total_count_
                        && (image_list_.size() >= (uint) total_count_)) {
                    break;
                }
            }
        }

        imglistfile.close();

        if (!image_list_.size()) {
            std::cout << "No images in image list found";
        }
    }

    ~Imagenet() {
    }
};

class Coco: public Dataset {
public:
    Coco(mlperf::TestSettings settings, int image_width, int image_height,
            int num_channels, string datapath, string image_format,
            int total_count, int perf_count, string workload, string dataset) {
        this->image_width_ = image_width;
        this->image_height_ = image_height;
        this->num_channels_ = num_channels;
        this->datapath_ = datapath;
        this->image_format_ = image_format;
        this->total_count_ = total_count;
        this->perf_count_ = perf_count;
        this->settings_ = settings;

        this->workload_ = workload;
        this->dataset_ = dataset;

        this->need_transpose_ = image_format == "NHWC" ? false : true;

        string image_list_file = datapath
                + "/annotations/instances_val2017.json";

        std::ifstream imglistfile(image_list_file);
        if (imglistfile) {
            std::stringstream buffer;

            buffer << imglistfile.rdbuf();
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(buffer, pt);

            // Parse JSON for filenames
            for (auto& img : pt.get_child("images")) {

                for (auto& prop : img.second) {
                    if (prop.first == "file_name") {
                        this->image_list_.push_back(
                                prop.second.get_value<std::string>());
                    }

                    // limit dataset
                    if (total_count_
                            && (image_list_.size() >= (uint) total_count_)) {
                        break;
                    }
                }
            }
        }

        if (!image_list_.size()) {
            std::cout << "No images in image list found";
        }
    }

    ~Coco() {
    }
};

#endif
