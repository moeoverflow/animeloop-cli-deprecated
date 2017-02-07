//
//  main.cpp
//  animeloop
//
//  Created by ShinCurry on 2017/2/7.
//  Copyright © 2017年 ShinCurry. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "convertRoutine.hpp"
#include "modelHandler.hpp"
#include "CmdLine.h"


TCLAP::CmdLine cmd("waifu2x reimplementation using OpenGL shader", ' ', "1.1.1");

TCLAP::ValueArg<std::string> cmdInputFile("i", "input_file",
                                          "path to input image file (you should input full path)", true, "",
                                          "string", cmd);

TCLAP::ValueArg<std::string> cmdOutputFile("o", "output_file",
                                           "path to output image file (you should input full path)", false,
                                           "(auto)", "string", cmd);

std::vector<std::string> cmdModeConstraintV({"noise", "scale", "noise_scale"});
TCLAP::ValuesConstraint<std::string> cmdModeConstraint(cmdModeConstraintV);
TCLAP::ValueArg<std::string> cmdMode("m", "mode", "image processing mode",
                                     false, "noise_scale", &cmdModeConstraint, cmd);

std::vector<int> cmdNRLConstraintV({1, 2});

TCLAP::ValuesConstraint<int> cmdNRLConstraint(cmdNRLConstraintV);
TCLAP::ValueArg<int> cmdNRLevel("", "noise_level", "noise reduction level",
                                false, 1, &cmdNRLConstraint, cmd);

TCLAP::ValueArg<double> cmdScaleRatio("", "scale_ratio",
                                      "custom scale ratio", false, 2.0, "double", cmd);

TCLAP::ValueArg<std::string> cmdModelPath("", "model_dir",
                                          "path to custom model directory (don't append last / )", false,
                                          "models", "string", cmd);

TCLAP::ValueArg<int> cmdNumberOfJobs("j", "jobs",
                                     "number of threads launching at the same time (dummy command)", false, 4, "integer",
                                     cmd);

TCLAP::ValueArg<int> cmdBlockSize("b", "block_size",
                                  "block size of split processing. default=512", false, 512, "integer",
                                  cmd);



cv::Mat doWaifu2x(cv::Mat image) {
    
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    
    int blockSize = cmdBlockSize.getValue();
    w2xc::modelUtility::getInstance().setBlockSize(cv::Size(blockSize, blockSize));
    
    // ===== Noise Reduction Phase =====
    if (cmdMode.getValue() == "noise" || cmdMode.getValue() == "noise_scale") {
        
        std::cout << "Noise reduction (Lv." << cmdNRLevel.getValue() << ") filtering..." << std::endl;
        
        std::string modelFileName(cmdModelPath.getValue());
        modelFileName = modelFileName + "/noise"
        + std::to_string(cmdNRLevel.getValue()) + "_model.bin";
        std::vector<std::unique_ptr<w2xc::Model> > models;
        
        if (!w2xc::modelUtility::generateModelFromBin(modelFileName, models))
            std::exit(-1);
        
        cv::Mat imageYUV;
        cv::cvtColor(image, imageYUV, cv::COLOR_RGB2YUV);
        std::vector<cv::Mat> imageSplit;
        cv::Mat imageY;
        cv::split(imageYUV, imageSplit);
        imageSplit[0].copyTo(imageY);
        
        w2xc::convertWithModels(imageY, imageSplit[0], models);
        
        cv::merge(imageSplit, imageYUV);
        cv::cvtColor(imageYUV, image, cv::COLOR_YUV2RGB);
        
    } // noise reduction phase : end
    
    // ===== scaling phase =====
    
    if (cmdMode.getValue() == "scale" || cmdMode.getValue() == "noise_scale") {
        
        // calculate iteration times of 2x scaling and shrink ratio which will use at last
        int iterTimesTwiceScaling = static_cast<int>(std::ceil(
                                                               std::log2(cmdScaleRatio.getValue())));
        double shrinkRatio = 0.0;
        if (static_cast<int>(cmdScaleRatio.getValue())
            != std::pow(2, iterTimesTwiceScaling)) {
            shrinkRatio = cmdScaleRatio.getValue()
            / std::pow(2.0, static_cast<double>(iterTimesTwiceScaling));
        }
        
        std::string modelFileName(cmdModelPath.getValue());
        modelFileName = modelFileName + "/scale2.0x_model.bin";
        std::vector<std::unique_ptr<w2xc::Model> > models;
        
        if (!w2xc::modelUtility::generateModelFromBin(modelFileName, models))
            std::exit(-1);
        
        // 2x scaling
        for (int nIteration = 0; nIteration < iterTimesTwiceScaling;
             nIteration++) {
            
            std::cout << "#" << std::to_string(nIteration + 1)
            << " 2x Scaling..." << std::endl;
            
            cv::Mat imageYUV;
            cv::Size imageSize = image.size();
            imageSize.width *= 2;
            imageSize.height *= 2;
            cv::Mat image2xNearest;
            cv::resize(image, image2xNearest, imageSize, 0, 0, cv::INTER_NEAREST);
            cv::cvtColor(image2xNearest, imageYUV, cv::COLOR_RGB2YUV);
            std::vector<cv::Mat> imageSplit;
            cv::Mat imageY;
            cv::split(imageYUV, imageSplit);
            imageSplit[0].copyTo(imageY);
            
            // generate bicubic scaled image and
            // convert RGB -> YUV and split
            imageSplit.clear();
            cv::Mat image2xBicubic;
            cv::resize(image,image2xBicubic,imageSize,0,0,cv::INTER_CUBIC);
            cv::cvtColor(image2xBicubic, imageYUV, cv::COLOR_RGB2YUV);
            cv::split(imageYUV, imageSplit);
            
            if(!w2xc::convertWithModels(imageY, imageSplit[0], models)){
                std::cerr << "w2xc::convertWithModels : something error has occured.\n"
                "stop." << std::endl;
                std::exit(1);
            };
            
            cv::merge(imageSplit, imageYUV);
            cv::cvtColor(imageYUV, image, cv::COLOR_YUV2RGB);
            
        } // 2x scaling : end
        
        if (shrinkRatio != 0.0) {
            cv::Size lastImageSize = image.size();
            lastImageSize.width =
            static_cast<int>(static_cast<double>(lastImageSize.width
                                                 * shrinkRatio));
            lastImageSize.height =
            static_cast<int>(static_cast<double>(lastImageSize.height
                                                 * shrinkRatio));
            cv::resize(image, image, lastImageSize, 0, 0, cv::INTER_LINEAR);
        }
        
    }
    
    image.convertTo(image, CV_8U, 255.0);
    
    return image;
}

int main(int argc, const char * argv[]) {
    
    try {
        cmd.parse(argc, argv);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Error : cmd.parse() threw exception" << std::endl;
        std::exit(-1);
    }
    cv::VideoCapture capture;
    try {
        capture.open(cmdInputFile.getValue());
    } catch (cv::Exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Error : VideoCapture init threw exception" << std::endl;
        std::exit(-1);
    }
    
    auto fps = capture.get(CV_CAP_PROP_FPS);
    auto fourcc = capture.get(CV_CAP_PROP_FOURCC);
    auto width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    auto height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    cv::Size size(width, height);

    cv::VideoWriter writer(cmdOutputFile.getValue(), fourcc, fps, size);
    
    cv::Mat image;
    bool end = false;
    do {
        
        capture >> image;
        
        if (image.empty()) {
            end = true;
        } else {
            writer << doWaifu2x(image);
        }
     } while (capture.isOpened());
    
    
    return 0;
}


