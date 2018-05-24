//
// Created by lhm on 18-5-8.
//


#include "Optimizer.h"

#include "g2o/types/types_seven_dof_expmap.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/core/block_solver.h"

namespace dso
{

void Optimizer::LoopCorrect(std::vector<FrameShell *>& allKeyFramesHistory, const int loopId, const int currentId,cv::Mat& Ploop,cv::Mat& Pcurrent)
{
    /*设置optimizer*/
    g2o::SparseOptimizer opt;

    //7:PoseDim, 3:LandmarkDim,线性方程求解类型设置
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3* solver73 = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg * LM_algorithm = new g2o::OptimizationAlgorithmLevenberg(solver73);

    LM_algorithm->setUserLambdaInit(1e-16);
    opt.setAlgorithm(LM_algorithm);

    /*设置顶点*/
    for (int i = 0; i < allKeyFramesHistory.size(); ++i) {

        FrameShell* KF = allKeyFramesHistory[i];
        g2o::Sim3 Scw(KF->camToWorld.inverse().unit_quaternion(),//反了？？？？？？？
                      KF->camToWorld.inverse().translation(),
                      1);

        //顶点的Sim3的指数映射
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();
        if(i == 0)
            VSim3->setFixed(true);

        VSim3->setEstimate(Scw);
        VSim3->setId(KF->id);
        VSim3->setMarginalized(false);

        opt.addVertex(VSim3);
    }
    /*设置普通边*/
    for (int i = 1; i < allKeyFramesHistory.size(); ++i) {

        FrameShell* KFi = allKeyFramesHistory[i];
        g2o::Sim3 Swi(KFi->camToWorld.unit_quaternion(),
                      KFi->camToWorld.translation(),
                /*KFi->m_translationScale*/1);


        FrameShell *KFj = allKeyFramesHistory[i-1];
        g2o::Sim3 Swj(KFj->camToWorld.unit_quaternion(),
                      KFj->camToWorld.translation(),
                /*KFj->m_translationScale*/1);

        g2o::Sim3 Sji = Swj.inverse() * Swi;

        g2o::EdgeSim3 *edge = new g2o::EdgeSim3();
//        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(opt.vertex(KFj->id)));//j
//        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(opt.vertex(KFi->id)));//i
        edge->setVertex(1, opt.vertex(KFj->id));//j
        edge->setVertex(0, opt.vertex(KFi->id));//i
        edge->setMeasurement(Sji);
        edge->setInformation(Mat77::Identity());

        opt.addEdge(edge);

        if(KFi->mm_pointsSize.size() != 0)
        {
            for(auto iter = KFi->mm_pointsSize.begin(),iterEnd = KFi->mm_pointsSize.end(); iter!=iterEnd; ++iter)
            {
                if(KFj == iter->first || iter->second < 50)
                    continue;

                FrameShell* KFk = iter->first;
                g2o::Sim3 Swk(KFk->camToWorld.unit_quaternion(),
                              KFk->camToWorld.translation(),
                              1);

                Sji = Swk.inverse() * Swi;

                g2o::EdgeSim3* edgeK = new g2o::EdgeSim3();
                edgeK->setVertex(1, opt.vertex(KFk->id));
                edgeK->setVertex(0, opt.vertex(KFi->id));
                edgeK->setMeasurement(Sji);
                edgeK->setInformation(Mat77::Identity());

                opt.addEdge(edgeK);

                g2o::Sim3 e = Sji*Swi.inverse()*Swk;
                std::cout<<"i "<<KFi->id<<"    j "<<KFk->id<<"    "
                         <<e.translation().transpose()<<"  "
                         <<e.rotation().coeffs().transpose()<<"  "
                         <<e.scale()<<std::endl;

            }
        }
    }
    /*设置loop边*/
    {
        g2o::Sim3 Scl = ComputeSim3(Pcurrent,Ploop);//loop To current
        g2o::EdgeSim3* edge = new g2o::EdgeSim3();
//        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(opt.vertex(allKeyFramesHistory[currentId]->id)));//j
//        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(opt.vertex(allKeyFramesHistory[loopId]->id)));//i
        edge->setVertex(1, opt.vertex(allKeyFramesHistory[currentId]->id));//j
        edge->setVertex(0, opt.vertex(allKeyFramesHistory[loopId]->id));//i
        edge->setMeasurement(Scl);
        edge->setInformation(Mat77::Identity());
        opt.addEdge(edge);

        g2o::VertexSim3Expmap* vSl = new g2o::VertexSim3Expmap();
        g2o::VertexSim3Expmap* vSc = new g2o::VertexSim3Expmap();
        vSl = dynamic_cast<g2o::VertexSim3Expmap*>(opt.vertex(allKeyFramesHistory[loopId]->id));
        vSc = dynamic_cast<g2o::VertexSim3Expmap*>(opt.vertex(allKeyFramesHistory[currentId]->id));
        g2o::Sim3 Si = vSl->estimate();
        g2o::Sim3 Sj = vSc->estimate();

        g2o::Sim3 e = Scl*Si*Sj.inverse();
        std::cout<<"before loop error =  "<<loopId<<"    "<<currentId<<"    "
                    <<e.translation().transpose()<<"  "
                    <<e.rotation().coeffs().transpose()<<"  "
                    <<e.scale()<<std::endl;
    }
    /*优化*/
    opt.setVerbose(true);//输出优化信息
    opt.initializeOptimization();
    opt.optimize(50);

    //opt.save("loop_after.g2o");

    //error
    g2o::Sim3 Scl = ComputeSim3(Pcurrent,Ploop);//loop To current
    g2o::VertexSim3Expmap* vSl = new g2o::VertexSim3Expmap();
    g2o::VertexSim3Expmap* vSc = new g2o::VertexSim3Expmap();
    vSl = dynamic_cast<g2o::VertexSim3Expmap*>(opt.vertex(allKeyFramesHistory[loopId]->id));
    vSc = dynamic_cast<g2o::VertexSim3Expmap*>(opt.vertex(allKeyFramesHistory[currentId]->id));
    g2o::Sim3 Si = vSl->estimate();
    g2o::Sim3 Sj = vSc->estimate();

    g2o::Sim3 e = Scl*Si*Sj.inverse();
    std::cout<<"after loop error =  "<<loopId<<"    "<<currentId<<"    "
             <<e.translation().transpose()<<"  "
             <<e.rotation().coeffs().transpose()<<"  "
             <<e.scale()<<std::endl;

    /*矫正回环,全部矫正到scale上了*/
    for(int i = 0,end = allKeyFramesHistory.size(); i < end; ++i){

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(opt.vertex(allKeyFramesHistory[i]->id));

        if(VSim3 == NULL)
            continue;
        g2o::Sim3 Siw = VSim3->estimate();

        SE3 Tiw = SE3(Siw.rotation(),Siw.translation()/Siw.scale());

//        std::cout<<"before "<<i<<" = " << allKeyFramesHistory[i]->camToWorld.translation().transpose()<<
//                 " " << allKeyFramesHistory[i]->camToWorld.so3().unit_quaternion().x()<<
//                 " " << allKeyFramesHistory[i]->camToWorld.so3().unit_quaternion().y()<<
//                 " " << allKeyFramesHistory[i]->camToWorld.so3().unit_quaternion().z()<<
//                 " " << allKeyFramesHistory[i]->camToWorld.so3().unit_quaternion().w() << "\n";

        allKeyFramesHistory[i]->camToWorld = Tiw.inverse();

        Tiw = Tiw.inverse();
//        std::cout<<"after "<<i<<" = " << Tiw.translation().transpose()<<
//                 " " << Tiw.so3().unit_quaternion().x()<<
//                 " " << Tiw.so3().unit_quaternion().y()<<
//                 " " << Tiw.so3().unit_quaternion().z()<<
//                 " " << Tiw.so3().unit_quaternion().w() << "\n"<<"\n";
//        std::cout<<i<<"   "<<Tiw.rotationMatrix()<<Tiw.translation()<<std::endl;
//        std::cout<<"after   "<<allKeyFramesHistory[i]->camToWorld<< std::endl;
    }



    std::cout<<"回环优化完成！"<<std::endl;
}


g2o::Sim3 Optimizer::ComputeSim3(cv::Mat& P1, cv::Mat& P2)
{
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    //-1- compute centroid and relative coordinates
    assert(P1.type() == P2.type());

    cv::Mat Pr1(P1.size(),P1.type());//relative coordinates
    cv::Mat Pr2(P2.size(),P2.type());
    cv::Mat O1(3,1,P1.type());       //centroid
    cv::Mat O2(3,1,P2.type());

    ChangePoints(P1,Pr1,O1);
    ChangePoints(P2,Pr2,O2);

//    std::cout<<"Pr1"<<Pr1<<std::endl;
//    std::cout<<"Pr2"<<Pr2<<std::endl;
//    std::cout<<"O1"<<O1<<std::endl;
//    std::cout<<"O2"<<O2<<std::endl;

    //-2- compute M
    cv::Mat M = Pr2*Pr1.t();//T 2->1

    //-3- compute N
    double N11,N12,N13,N14,N22,N23,N24,N33,N34,N44;
    N11 = M.at<double>(0,0) + M.at<double>(1,1) + M.at<double>(2,2);
    N12 = M.at<double>(1,2) - M.at<double>(2,1);
    N13 = M.at<double>(2,0) - M.at<double>(0,2);
    N14 = M.at<double>(0,1) - M.at<double>(1,0);
    N22 = M.at<double>(0,0) - M.at<double>(1,1) - M.at<double>(2,2);
    N23 = M.at<double>(0,1) + M.at<double>(1,0);
    N24 = M.at<double>(2,0) + M.at<double>(0,2);
    N33 = -M.at<double>(0,0) + M.at<double>(1,1) - M.at<double>(2,2);
    N34 = M.at<double>(2,1) + M.at<double>(1,2);
    N44 = -M.at<double>(0,0) - M.at<double>(1,1) + M.at<double>(2,2);

    //std::cout<<N11<<"  "<<N12<<"  "<<N13<<"  "<<N14<<"  "<<N22<<"  "<<N23<<"  "<<N24<<"  "<<N33<<"  "<<N34<<"  "<<N44<< std::endl;

    cv::Mat N = (cv::Mat_<double>(4,4)<< N11, N12, N13, N14,
                                         N12, N22, N23, N24,
                                         N13, N23, N33, N34,
                                         N14, N24, N34, N44);


    //-4- compute eigenvector of hightest eigenvalue
    cv::Mat eigVal,eigVec;
    cv::eigen(N,eigVal,eigVec);

    cv::Mat RotationVec(1,3,eigVec.type());
    (eigVec.row(0).colRange(1,4)).copyTo(RotationVec);

    double angle = atan2(norm(RotationVec),eigVec.at<double>(0,0));//计算旋转角度

    RotationVec = 2*angle*RotationVec/norm(RotationVec);//旋转向量，模代表角度，单位向量代表方向

    cv::Mat R12(3,3,P2.type());
    cv::Rodrigues(RotationVec,R12);

    //-5- Rotate set 2
    cv::Mat P3 = R12*Pr2;


    //-6- compute scale
    double A = Pr1.dot(P3);
    cv::Mat Pr2Square = Pr2*Pr2.t();
    double B = cv::trace(Pr2Square)(0);
    double scale1 = A/B;
    printf("scale1 = %f  \n",scale1);

    //-7- compute translation
    cv::Mat t12(3,1,P2.type());
    t12 = O1 - scale1*R12*O2;

    //-8- Sim3
    Mat33 R;
    R <<  R12.at<double>(0,0),R12.at<double>(0,1),R12.at<double>(0,2),
          R12.at<double>(1,0),R12.at<double>(1,1),R12.at<double>(1,2),
          R12.at<double>(2,0),R12.at<double>(2,1),R12.at<double>(2,2);
    Eigen::Vector3d t(t12.at<double>(0,0),t12.at<double>(1,0),t12.at<double>(2,0));

    g2o::Sim3 S12(R,t,scale1);

    return S12;

}

void Optimizer::ChangePoints(cv::Mat& P, cv::Mat& Pr, cv::Mat& O)
{
    assert(P.data && Pr.data && O.data);

    cv::reduce(P,O,1,CV_REDUCE_AVG);

    for(int i=0; i<P.cols; ++i)
    {
        Pr.col(i) = P.col(i) - O;
    }
}

}