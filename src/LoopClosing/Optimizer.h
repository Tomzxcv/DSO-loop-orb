//
// Created by lhm on 18-5-8.
//

#ifndef DSO_OPTIMIZER_H
#define DSO_OPTIMIZER_H

#include <vector>
#include "util/FrameShell.h"
#include "g2o/types/sim3.h"


namespace dso
{

class Optimizer {

public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Optimizer(){};

    void LoopCorrect(std::vector<FrameShell*>& allKeyFramesHistory,
                            const int loopId,const int currentId,
                            cv::Mat& Ploop,cv::Mat& Pcurrent);

private:
    g2o::Sim3 ComputeSim3(cv::Mat& P1, cv::Mat& P2);

    void ChangePoints(cv::Mat& P, cv::Mat& Pr, cv::Mat& O);
};

}

#endif //DSO_OPTIMIZER_H
