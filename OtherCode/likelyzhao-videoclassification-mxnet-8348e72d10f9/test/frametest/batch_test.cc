/*!
    irint(mAP)
 *  Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 */

//
//  File: image-classification-predict.cpp
//  This is a simple predictor which shows
//  how to use c api for image classfication
//  It uses opencv for image reading
//  Created by liuxiao on 12/9/15.
//  Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
//  Home Page: www.liuxiao.org
//  E-mail: liuxiao@foxmail.com
//

#include <stdio.h>

// Path for c_predict_api
#include <mxnet/c_predict_api.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <sys/stat.h> 　
#include <sys/types.h> 

const mx_float DEFAULT_MEAN_R = 0 ;
const mx_float DEFAULT_MEAN_G = 0 ;
const mx_float DEFAULT_MEAN_B = 0;
const std::string OUTPUTLAYERNAME = "flatten0";

// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            length_ = 0;
            buffer_ = NULL;
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        if (buffer_) {
          delete[] buffer_;
          buffer_ = NULL;
        }
    }
};

void GetImageFile(const std::string image_file,
                  mx_float* image_data, const int channels,
                  const cv::Size resize_size, const mx_float* mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);
//    std::cout<< image_file <<std::endl;
    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im256;
    float resize_ratio;
//    std::cout<<"im rows :"<<im_ori.rows<<std::ndl;  
    resize_ratio = float(std::min(im_ori.rows,im_ori.cols))/256;
//   std::cout<<"ratio  :"<<resize_ratio<<std::endl;  
    resize(im_ori,im256,cv::Size(int(im_ori.rows/resize_ratio),int(im_ori.cols/resize_ratio)));
    
    cv::Rect rect = cv::Rect(im256.rows/2 - 112,im256.cols/2 - 112,224,224);
//    std::cout<<"im rows :"<<im256.rows<<std::endl;  
//    std::cout<<"im cols :"<<im256.cols<<std::endl;  

//    std::cout<<"rect :"<<rect.x<<rect.y<<rect.width<<rect.height<<std::endl;  
    cv::Mat im;
    im = im256(rect);
//    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = DEFAULT_MEAN_B;
    mean_g = DEFAULT_MEAN_G;
    mean_r = DEFAULT_MEAN_R;

    for (int i = 0; i < im.rows; i++) {
        uchar* data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
               mean_data++;
            }
            if (channels > 1) {
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
            }

            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
        }
    }
}

// oadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(std::string synset_file) {
    std::ifstream fi(synset_file.c_str());

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while ( fi >> synset ) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

void  LoadTestlist(std::string test_file, std::vector<std::vector<std::string> > & fileList, std::vector<std::string> &labels) {
    std::ifstream fi(test_file.c_str());
  
    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << test_file << std::endl;
        assert(false);
    }

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << test_file << std::endl;
        assert(false);
    }

    std::string synset, label;
    int number;
    while ( fi >> label >> number){
//        std::cout <<  "number:" << number << std::endl; 
//        std::cout <<  "label:" << label << std::endl; 
        std::vector<std::string> tempname;
        getline(fi,synset);
//        std::cout <<  "synset:" << synset << std::endl; 
	for (int j = 0;j<number;j++){
            std::string name;
	    getline(fi, name);
//            std::cout<< name<<std::endl;
            tempname.push_back(name);
	}
        fileList.push_back(tempname);
        labels.push_back(label);
    }

    fi.close();

    return ;
}
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
void  LoadTestlistwithlabel(std::string test_file, std::vector<std::vector<std::string> > & fileList, std::vector<std::string> &labels) {
    std::ifstream fi(test_file.c_str());
  
    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << test_file << std::endl;
        assert(false);
    }

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << test_file << std::endl;
        assert(false);
    }

    std::string synset, label;
    int number;
    while ( fi >> synset >> label >> number){
//        std::cout <<  "number:" << number << std::endl; 
//        std::cout <<  "synset:" << synset << std::endl; 
        std::vector<std::string> tempname;
        getline(fi,synset);
//        std::cout <<  "synset:" << synset << std::endl; 
	for (int j = 0;j<number;j++){
            std::string name;
	    getline(fi, name);
//            std::cout<< name<<std::endl;
            tempname.push_back(name);
	}
        fileList.push_back(tempname);
        labels.push_back(label);
    }

    fi.close();

    return ;
}


void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
        printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if ( data[i] > best_accuracy ) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
    synset[best_idx].c_str(), best_idx, best_accuracy);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "No test image here." << std::endl
        << "Usage: ./image-classification-predict apple.jpg" << std::endl;
        return 0;
    }

    std::string test_file;
    test_file = std::string(argv[1]);
    std::cout<<"testfile:"<<test_file<<std::endl;
    int gpuidx ;
    sscanf(argv[2],"%d",&gpuidx);
    std::cout<<"gpuidx"<<gpuidx<<std::endl;
    // Models path for your model, you have to modify it
    std::string json_file = "model/resnext_finetune-symbol.json";
    std::string param_file = "model/resnext_finetune-0001.params";
    std::string synset_file = "model/synset.txt";
    std::string nd_file = "";
    std::string fea_path = "/workspace/data-out/trainval/";
    
    int isCreate = mkdir(fea_path.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    if( !isCreate ) printf("create path:%s\n",fea_path.c_str());
    else printf("create path failed! error code : %s \n",fea_path.c_str());

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    // Parameters
    int dev_type = 2;  // 1: cpu, 2: gpu
    int dev_id = gpuidx;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;

    mx_uint num_output_nodes =1;
    const char* output_key[1] = {OUTPUTLAYERNAME.c_str()};
    const char** output_keys = output_key;

    // Image size and channels
    int width = 224;
    int height = 224;
    int channels = 3;

    int batch_size =32;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    const mx_uint input_shape_data[4] = { batch_size,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(height), 
                                        static_cast<mx_uint>(width)};
    PredictorHandle pred_hnd = 0;

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }


//    MXPredCreatePartialOut((const char*)json_data.GetBuffer(),
//                 (const char*)param_data.GetBuffer(),
//                 static_cast<size_t>(param_data.GetLength()),
//                 dev_type,
//                 dev_id,
//                 num_input_nodes,
//                 input_keys,
//                 input_shape_indptr,
//                 input_shape_data,
//                 num_output_nodes,
//                 output_keys,
 //                &pred_hnd);
    // Create Predictor
   MXPredCreate((const char*)json_data.GetBuffer(),
                 (const char*)param_data.GetBuffer(),
                 static_cast<size_t>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pred_hnd);
    assert(pred_hnd);

    int image_size = width * height * channels;

    // Read Mean Data
    const mx_float* nd_data = NULL;
    NDListHandle nd_hnd = 0;
    BufferFile nd_buf(nd_file);

    if (nd_buf.GetLength() > 0) {
        mx_uint nd_index = 0;
        mx_uint nd_len;
        const mx_uint* nd_shape = 0;
        const char* nd_key = 0;
        mx_uint nd_ndim = 0;

        MXNDListCreate((const char*)nd_buf.GetBuffer(),
                   nd_buf.GetLength(),
                   &nd_hnd, &nd_len);

        MXNDListGet(nd_hnd, nd_index, &nd_key, &nd_data, &nd_shape, &nd_ndim);
    }

    // Read Image Data
    std::vector<mx_float> image_data = std::vector<mx_float>(batch_size*image_size);

    // Synset path for your model, you have to modify it
    std::vector<std::string> synset = LoadSynset(synset_file);

    std::vector<std::vector<std::string> > filelist;
    std::vector<std::string> labels;

    LoadTestlist(test_file,filelist,labels);
    int right =0;
    mx_uint shape_len;
  
    size_t size = 1;
    FILE * fpout = fopen("results.txt","w");
    for (mx_uint i =0;i<filelist.size();i++){ 
        std::vector<std::string> temp = filelist[i];
        std::string label = labels[i];
        std::vector<float* > videoFea;
        int output_fealen = 0;
        for (mx_uint j =0; j< temp.size()/batch_size;j++){
            for (mx_uint k =0; k < batch_size;k++){
                std::cout<<temp[j*batch_size+k]<<std::endl;
	        GetImageFile(temp[j*batch_size+k], image_data.data() + k * image_size,
                     channels, cv::Size(width, height), nd_data);
		}
            MXPredSetInput(pred_hnd, "data", image_data.data(), batch_size* image_size);
	    // Do Predict Forward
   	    MXPredForward(pred_hnd);

    	    mx_uint output_index = 0;

    	    mx_uint *shape = 0;

    	    // Get Output Result
    	    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

            size = 1;
            for (mx_uint ii = 0; ii < shape_len; ++ii) size *= shape[ii];

            std::vector<float> data(size);
            //std::cout<< "size" << size << std::endl;
            MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);
            output_fealen = size/batch_size;
   //         int labelidx = 0;
   //         sscanf(label.c_str(),"%d",&labelidx);
   //         std::cout<< "data306 " << data[318] << std::endl;
            for (mx_uint ii =0;ii<batch_size;ii++){
                float * data_single = (float*)malloc(sizeof(float)*size/batch_size);   
                memcpy(data_single,&(data[ii * size/batch_size]),sizeof(float)*size/batch_size);
                videoFea.push_back(data_single);
             }
	  
	   }
        // deal with remain 
           int remain = temp.size() - temp.size()/batch_size*batch_size; 
           std::cout<<"remain : "<<remain<<std::endl;
           for (mx_uint j = temp.size()/batch_size * batch_size ;j<temp.size();j++){
                    std::cout<<temp[j]<<std::endl;
                    GetImageFile(temp[j], image_data.data() + (j-temp.size()/batch_size*batch_size) * image_size,
                     channels, cv::Size(width, height), nd_data);
		}

            MXPredSetInput(pred_hnd, "data", image_data.data(), batch_size* image_size);
            // Do Predict Forward
            MXPredForward(pred_hnd);

            mx_uint output_index = 0;

            mx_uint *shape = 0;

            // Get Output Result
            MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

            size = 1;
            for (mx_uint ii = 0; ii < shape_len; ++ii) size *= shape[ii];

            std::vector<float> data(size);
   //         std::cout<< "size" << size << std::endl;
            MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);
   //         int labelidx = 0;
   //         sscanf(label.c_str(),"%d",&labelidx);
   //         std::cout<< "data306 " << data[318] << std::endl;
            for (mx_uint ii =0;ii<remain;ii++){
                float * data_single = (float*)malloc(sizeof(float)*size/batch_size);
                memcpy(data_single,&(data[ii * output_fealen]),sizeof(float)*size/batch_size);
                videoFea.push_back(data_single);
             }
        //lsvc075999_fc6_vgg19_frame.binary1
        fprintf(fpout,"%s, ",label.c_str());
        std::cout<<label<<std::endl;
        float * tempres =(float*)malloc(sizeof(float)*output_fealen);
        memset(tempres,0,sizeof(float)*output_fealen);
        for (mx_uint ii =0;ii<videoFea.size();ii++)
            for (mx_uint jj =0;jj<output_fealen;jj++){
            tempres[jj] += videoFea[ii][jj];
	}
        for (mx_uint ii =0;ii<output_fealen ;ii++){
            fprintf(fpout,"%f ",tempres[ii]/videoFea.size());
 	}
        fprintf(fpout,"\n");
        free(tempres);
        for (mx_uint ii =0;ii<videoFea.size();ii++){
      	    free(videoFea[ii]);
	}
 //       lsvc075999_fc6_vgg19_frame.binary
//        std::cout<<"idx"<<i<<std::endl;
    }

        fclose(fpout);
    // Release NDList
    if (nd_hnd)
      MXNDListFree(nd_hnd);

    // Release Predictor
    MXPredFree(pred_hnd);

    // Synset path for your model, you have to modify it
//    std::vector<std::string> synset = LoadSynset(synset_file);

    // Print Output Data
//    PrintOutputResult(data, synset);

    return 0;
}
