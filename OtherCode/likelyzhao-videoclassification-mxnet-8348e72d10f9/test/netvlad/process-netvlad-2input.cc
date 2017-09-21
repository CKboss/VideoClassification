/*!
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

#include <sstream>
#include <opencv2/opencv.hpp>

const mx_float DEFAULT_MEAN_R = 123.68;
const mx_float DEFAULT_MEAN_G = 116.779;
const mx_float DEFAULT_MEAN_B = 103.939;
const int  FEATURE_LEN_INPUT_1  =2048;
const int  FEATURE_LEN_INPUT_2  =2048;
const int  NUM_INPUT = 800;

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


#include <sys/stat.h>

unsigned long get_file_size(const char *path)
{
    unsigned long filesize = -1;    
    struct stat statbuff;
    if(stat(path, &statbuff) < 0){
        return filesize;
    }else{
        filesize = statbuff.st_size;
    }
    return filesize;
}

#define random(x) (rand()%x)
void GetImageFile_1(const std::string image_file,
                  mx_float* image_data) {
    // Read all kinds of file into a BGR color 3 channels image
    std::string fea_name = "/workspace/data-1/trainval/" + image_file + "_pool5_senet.binary";
    std::cout<<fea_name<<std::endl;
    int filesize = get_file_size(fea_name.c_str());
    std::cout<<filesize<<std::endl;
    float * tempfea = (float*)malloc(filesize);    
    FILE * fp = fopen(fea_name.c_str(),"rb");
    fread(tempfea,sizeof(float),filesize/sizeof(float),fp);
    fclose(fp);
    
//    std::cout<< image_file <<std::endl;
    if (tempfea == NULL) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    int fealen = filesize/sizeof(float)/FEATURE_LEN_INPUT_1;
    for (int i =0;i<NUM_INPUT;i++){
        int randidx = random(fealen);
        memcpy(image_data+ i* FEATURE_LEN_INPUT_1,tempfea + randidx*FEATURE_LEN_INPUT_1,sizeof(float)*FEATURE_LEN_INPUT_1);
	}
}

void GetImageFile_2(const std::string image_file,
                  mx_float* image_data) {
    // Read all kinds of file into a BGR color 3 channels image
    std::string fea_name = "/workspace/data-2/trainval/" + image_file + "_pool5_place365_frame.binary";
    std::cout<<fea_name<<std::endl;
    int filesize = get_file_size(fea_name.c_str());
    std::cout<<filesize<<std::endl;
    float * tempfea = (float*)malloc(filesize);    
    FILE * fp = fopen(fea_name.c_str(),"rb");
    fread(tempfea,sizeof(float),filesize/sizeof(float),fp);
    fclose(fp);
    
//    std::cout<< image_file <<std::endl;
    if (tempfea == NULL) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    int fealen = filesize/sizeof(float)/FEATURE_LEN_INPUT_2;
    for (int i =0;i<NUM_INPUT;i++){
        int randidx = random(fealen);
        memcpy(image_data+ i* FEATURE_LEN_INPUT_2,tempfea + randidx*FEATURE_LEN_INPUT_2,sizeof(float)*FEATURE_LEN_INPUT_2);
	}
}

// LoadSynsets
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

// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
void  LoadTestlistwithlabel(std::string test_file, std::vector<std::string>  & fileList, std::vector<std::string> &labels) {
    std::ifstream fi(test_file.c_str());
  
    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << test_file << std::endl;
        assert(false);
    }

    if ( !fi.is_open() ) {
        std::cerr << "Error opening synset file " << test_file << std::endl;
        assert(false);
    }

    std::string synset, label, filename;
    while ( fi >> filename >> label){
//        std::cout <<  "number:" << number << std::endl; 
//        std::cout <<  "synset:" << synset << std::endl; 
//        std::cout <<  "synset:" << synset << std::endl; 
        fileList.push_back(filename);
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

std::string Convert(float Num)
{
    std::ostringstream oss;
    oss<<Num;
    std::string str(oss.str());
    return str;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "No test image here." << std::endl
        << "Usage: ./image-classification-predict apple.jpg" << std::endl;
        return 0;
    }

    std::string test_file;
    test_file = std::string(argv[1]);

    // Models path for your model, you have to modify it
    std::string json_file = "model/netvlad-symbol.json";
    std::string param_file = "model/netvlad-0025.params";
    std::string synset_file = "model/lsvc_class_index.txt";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    FILE * fp = fopen("results.txt","w");
    int dev_type = 2;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 2;  // 1 for feedforward
    const char* input_key[2] = {"data-1","data-2"};
    const char** input_keys = input_key;

    // Image size and channels

    const mx_uint input_shape_indptr[3] = { 0, 3, 6 };

    const mx_uint input_shape_data[6] ={1,
					static_cast<mx_uint>(NUM_INPUT),
					static_cast<mx_uint>(FEATURE_LEN_INPUT_1),
					1,
					static_cast<mx_uint>(NUM_INPUT),
					static_cast<mx_uint>(FEATURE_LEN_INPUT_2)
					};


    PredictorHandle pred_hnd = 0;

    if (json_data.GetLength() == 0 ||
        param_data.GetLength() == 0) {
        return -1;
    }

       std::cout<<"init finished"<<std::endl;
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

       std::cout<<"init finished"<<std::endl;
    int image_size_1 = FEATURE_LEN_INPUT_1 * NUM_INPUT;

    // Read Image Data
    std::vector<mx_float> image_data_1 = std::vector<mx_float>(image_size_1);
    int image_size_2 = FEATURE_LEN_INPUT_2 * NUM_INPUT;

    std::vector<mx_float> image_data_2 = std::vector<mx_float>(image_size_2);
    // Synset path for your model, you have to modify it
    std::vector<std::string> synset = LoadSynset(synset_file);

    std::vector<std::string> filelist;
    std::vector<std::string> labels;

    LoadTestlistwithlabel(test_file,filelist,labels);
    int right =0;
    mx_uint shape_len;
  
    size_t size = 1;
    for (mx_uint i =0;i<filelist.size();i++){ 
        std::string label = labels[i];
        std::vector<float> score;
        std::vector<int> predidx;

	GetImageFile_1(filelist[i], image_data_1.data());
        MXPredSetInput(pred_hnd, "data-1", image_data_1.data(), image_size_1);
	GetImageFile_2(filelist[i], image_data_2.data());
        MXPredSetInput(pred_hnd, "data-2", image_data_2.data(), image_size_2);
	    // Do Predict Forward
   	MXPredForward(pred_hnd);

    	mx_uint output_index = 0;

    	mx_uint *shape = 0;

    	    // Get Output Result
    	MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

        size = 1;
        for (mx_uint ii = 0; ii < shape_len; ++ii) size *= shape[ii];

        std::vector<float> data(size);

        MXPredGetOutput(pred_hnd, output_index, &(data[0]), size);
        std::string res;
        fprintf(fp,"%s, ",filelist[i].c_str());
        for (int i =0;i<size;i++){
            fprintf(fp,"%f ",data[i]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
//        std::cout<<"idx"<<i<<std::endl;

    // Release Predictor
    MXPredFree(pred_hnd);

    // Synset path for your model, you have to modify it
//    std::vector<std::string> synset = LoadSynset(synset_file);

    // Print Output Data
//    PrintOutputResult(data, synset);

    return 0;
}
