/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <dirent.h>
#include <unistd.h>

#include "Lightcnn.h"
#include "MxBase/Log/Log.h"
void getdataname(std::string *filename, const std::string &imgpath);
APP_ERROR ScanImages(const std::string &path,
                     std::vector<std::string> *imgFiles) {
    DIR *dirPtr = opendir(path.c_str());
    if (dirPtr == nullptr) {
      LogError << "opendir failed. dir:" << path;
      return APP_ERR_INTERNAL_ERROR;
    }
    dirent *direntPtr = nullptr;
    while ((direntPtr = readdir(dirPtr)) != nullptr) {
      std::string fileName = direntPtr->d_name;
      if (fileName == "." || fileName == "..") {
        continue;
      }

      (*imgFiles).emplace_back(path + "/" + fileName);
    }
    closedir(dirPtr);
    return APP_ERR_OK;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './test.png'";
        return APP_ERR_OK;
    }
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "../model/lightcnn4Test1.om";
    std::string imgPath = argv[1];
    std::string dataname = "";
    getdataname(&dataname, imgPath);
    std::string resultPath = "output/";
    auto lightcnn = std::make_shared<Lightcnn>();
    APP_ERROR ret = lightcnn->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "ligthcnn init failed, ret=" << ret << ".";
        return ret;
    }

    std::vector<std::string> imgFilePaths;
    ret = ScanImages(imgPath, &imgFilePaths);
    if (ret != APP_ERR_OK) {
       return ret;
    }
    auto startTime = std::chrono::high_resolution_clock::now();
    int cnt = 0;
    for (auto &imgFile : imgFilePaths) {
      ret = lightcnn->Process(imgFile, resultPath);
      if (ret != APP_ERR_OK) {
        LogError << "Lightcnn process failed, ret=" << ret << ".";
        lightcnn->DeInit();
        return ret;
      }
      if (cnt++ % 1000 == 0) {
        LogError << cnt << std::endl;
      }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    lightcnn->DeInit();
    double costMilliSecs =
      std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double fps = 1000.0 * imgFilePaths.size()  / lightcnn->GetInferCostMilliSec();
    LogInfo << "[Process Delay] cost: " << costMilliSecs << " ms\tfps: " << fps
          << " imgs/sec";
    return APP_ERR_OK;
}

void getdataname(std::string *filename, const std::string &imgpath) {
    int i;
    for (i = imgpath.length() - 1; i >= 0; i--) {
      if (imgpath[i] == '/') {
        break;
      }
    }
    while (imgpath[++i] != '\0') {
      *filename += imgpath[i];
    }
}
