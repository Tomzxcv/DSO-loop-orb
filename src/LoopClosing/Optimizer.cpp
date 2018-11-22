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
#include "LoopClosing/EdgeAndVertex.h"

namespace dso
{


void Optimizer::LoopCorrect(std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>>& v_Sim3,
                            std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>>& v_newSim3,
                            std::list<Constraint>& list_constrint)
{
    /*设置optimizer*/
    g2o::SparseOptimizer opt;

    //7:PoseDim, 3:LandmarkDim,线性方程求解类型设置

    //typedef g2o::BlockSolver< g2o::BlockSolverTraits<7, 7> > BlockSolver_7_7;
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3* solver73 = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg * LM_algorithm = new g2o::OptimizationAlgorithmLevenberg(solver73);

    LM_algorithm->setUserLambdaInit(1e-16);
    opt.setAlgorithm(LM_algorithm);


    /*设置顶点*/
    for (int i = 0; i < v_Sim3.size(); i++) {

        //顶点的Sim3的指数映射
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        if(i == 0)//设置loop帧为固定帧
            VSim3->setFixed(true);

        VSim3->setEstimate(v_Sim3[i]);
        VSim3->setId(i);
        VSim3->setMarginalized(false);
        opt.addVertex(VSim3);
    }



    /*设置边*/
    for (auto iter = list_constrint.begin(); iter!=list_constrint.end(); iter++) {

        g2o::EdgeSim3 *edge = new g2o::EdgeSim3();
        edge->setVertex(1, opt.vertex(iter->id_2));//2
        edge->setVertex(0, opt.vertex(iter->id_1));//1
        edge->setMeasurement(iter->mean);
        edge->setInformation(iter->information);

        opt.addEdge(edge);
    }
    /*优化*/
    opt.setVerbose(true);//输出优化信息
    opt.initializeOptimization();
    opt.optimize(50);



    /*矫正回环,全部矫正到scale上了*/
    //std::cout<<"old Sim3 size = "<<v_Sim3.size()<<"  new Sim3 size = "<<v_newSim3.size()<<std::endl;
    for(int i = 0; i < v_Sim3.size(); i++){

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(opt.vertex(i));
        g2o::Sim3 Siw = VSim3->estimate();

        v_newSim3[i] = Siw;
    }

    std::cout<<"回环优化完成！"<<std::endl;
}




void Optimizer::OptimizeSim3(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &Puvd_1,
                             std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &Puvd_2,
                             g2o::Sim3 &S12, cv::Mat K1, cv::Mat K2, std::vector<bool>& isInliersBest, bool isFixed)
{
    //先对3D点和Sim3一起优化
    g2o::SparseOptimizer opt;

    g2o::BlockSolverX::LinearSolverType* linearSolver =
            new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solverX = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* LMsolver = new g2o::OptimizationAlgorithmLevenberg(solverX);
    opt.setAlgorithm(LMsolver);


    std::cout<<"1"<<std::endl;

    //顶点sim3
    g2o::VertexSim3Expmap* S = new g2o::VertexSim3Expmap();
    S->_focal_length1[0] = K1.at<double>(0,0);
    S->_focal_length1[1] = K1.at<double>(1,1);
    S->_principle_point1[0] = K1.at<double>(0,2);
    S->_principle_point1[1] = K1.at<double>(1,2);
    S->_fix_scale = false;
    S->_focal_length2[0] = K2.at<double>(0,0);
    S->_focal_length2[1] = K2.at<double>(1,1);
    S->_principle_point2[0] = K2.at<double>(0,2);
    S->_principle_point2[1] = K2.at<double>(1,2);
    S->setEstimate(S12);//P2=>P1
    S->setId(0);
    S->setMarginalized(false);
    opt.addVertex(S);
    std::cout<<"2"<<std::endl;


    //顶点3D
    assert(Puvd_1.size() == Puvd_2.size());
    if(Puvd_1.size() == 0)
        return;
    int size = Puvd_1.size();

    //P1状态量,1~size
    for (int i = 1; i <= size; ++i)
    {
        if(!isInliersBest[i-1])
            continue;
        VertexUVD* P1 = new VertexUVD();
        P1->setEstimate(Puvd_1[i-1]);
        P1->setId(i);
        P1->setMarginalized(false);
        opt.addVertex(P1);
    }
    //P2状态量,size+1~2*size
    for (int i = size+1; i <= 2*size; ++i)
    {
        if(!isInliersBest[i-size-1])
            continue;
        VertexUVD* P2 = new VertexUVD();
        P2->setEstimate(Puvd_2[i-size-1]);
        P2->setId(i);
        P2->setMarginalized(false);
        if(isFixed)
            P2->setFixed(true);//固定loop中的特征点s和深度d
        opt.addVertex(P2);
    }
    std::cout<<"3"<<std::endl;

    //边P2=>P1
    for (int j = 1; j <= size; ++j) {
        if(!isInliersBest[j-1])
            continue;
        EdgeProjectP2ToP1* edge = new EdgeProjectP2ToP1();
        edge->setVertex(1,(opt.vertex(0)));//1是Sim3
        edge->setVertex(0,(opt.vertex(j+size)));//0是3D点,P2
        edge->setMeasurement(Eigen::Vector2d(Puvd_1[j-1][0], Puvd_1[j-1][1]));//P1
        edge->setInformation(Eigen::Matrix<double,2,2>::Identity()*0.5);
        opt.addEdge(edge);
    }
    //边P1=>P2
    for (int j = 1; j <= size; ++j) {
        if(!isInliersBest[j-1])
            continue;
        EdgeProjectP1ToP2* edge = new EdgeProjectP1ToP2();
        edge->setVertex(1,(opt.vertex(0)));//1是Sim3
        edge->setVertex(0,(opt.vertex(j)));//0是3D点,P1
        edge->setMeasurement(Eigen::Vector2d(Puvd_2[j-1][0], Puvd_2[j-1][1]));//P2
        edge->setInformation(Eigen::Matrix<double,2,2>::Identity()*0.5);
        opt.addEdge(edge);
    }

    std::cout<<"4"<<std::endl;
    opt.setVerbose(true);
    opt.initializeOptimization();
    std::cout<<"5"<<std::endl;
    opt.optimize(5);//有问题

    printf("Sim3 优化完成！\n");

    g2o::VertexSim3Expmap* S_cor = static_cast<g2o::VertexSim3Expmap*>(opt.vertex(0));
    S12 = S_cor->estimate();

//    for (int i = 1; i < size; ++i) {
//        std::set<BinEdgeSim3*> e = std::set<static_cast<BinEdgeSim3*>>(opt.edges());
//        //Eigen::Vector2d s = e->measurement();
//        //std::cout<<i<<"  "<<s<< std::endl;
//    }
    for (int i = 1; i <= size; ++i) {
        if(!isInliersBest[i-1])
            continue;
        VertexUVD* Pi = static_cast<VertexUVD*>(opt.vertex(i));
        Puvd_1[i-1] = Pi->estimate();
    }
    for (int i = size+1; i <= 2*size; ++i) {
        if(!isInliersBest[i-size-1])
            continue;
        VertexUVD* Pi = static_cast<VertexUVD*>(opt.vertex(i));
        Puvd_2[i-size-1] = Pi->estimate();
    }

}

void Optimizer::OptimizeSim3(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &P1,
                             std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &P2,
                             g2o::Sim3 &S12, cv::Mat K, std::vector<bool>& isInliersBest)
{
    //先对3D点和Sim3一起优化
    g2o::SparseOptimizer opt;

    g2o::BlockSolverX::LinearSolverType* linearSolver =
            new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solverX = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* LMsolver = new g2o::OptimizationAlgorithmLevenberg(solverX);
    opt.setAlgorithm(LMsolver);


    std::cout<<"1"<<std::endl;

    //顶点sim3
    g2o::VertexSim3Expmap* S = new g2o::VertexSim3Expmap();
    S->_focal_length1[0] = K.at<double>(0,0);
    S->_focal_length1[1] = K.at<double>(1,1);
    S->_principle_point1[0] = K.at<double>(0,2);
    S->_principle_point1[1] = K.at<double>(1,2);
    S->_fix_scale = false;
    S->setEstimate(S12.inverse());//T   P3=>P2
    S->setId(0);
    S->setMarginalized(false);
    opt.addVertex(S);
    std::cout<<"2"<<std::endl;


    //顶点3D
    assert(P1.size() == P2.size());
    if(P1.size() == 0)
        return;
    int size = P1.size();

    //P1为状态量，P2为测量量
    for (int i = 1; i <= size; ++i)
    {
        if(!isInliersBest[i-1])
            continue;
        Vertex3D* P = new Vertex3D();
        P->setEstimate(P1[i-1]);
        P->setId(i);
        P->setMarginalized(false);
        opt.addVertex(P);
    }
    std::cout<<"3"<<std::endl;

    //边
    for (int j = 1; j <= size; ++j) {
        if(!isInliersBest[j-1])
            continue;
        BinEdgeSim3* edge = new BinEdgeSim3();
        edge->setVertex(1,(opt.vertex(0)));//1是Sim3
        edge->setVertex(0,(opt.vertex(j)));//0是3D点
        edge->setMeasurement(P2[j-1]);
        edge->setInformation(Eigen::Matrix<double,2,2>::Identity()*0.5);
        opt.addEdge(edge);
    }

    std::cout<<"4"<<std::endl;
    opt.setVerbose(false);
    opt.initializeOptimization();
    std::cout<<"5"<<std::endl;
    opt.optimize(5);//有问题

    printf("Sim3 优化完成！\n");

    g2o::VertexSim3Expmap* S_cor = static_cast<g2o::VertexSim3Expmap*>(opt.vertex(0));
    S12 = S_cor->estimate().inverse();

//    for (int i = 1; i < size; ++i) {
//        std::set<BinEdgeSim3*> e = std::set<static_cast<BinEdgeSim3*>>(opt.edges());
//        //Eigen::Vector2d s = e->measurement();
//        //std::cout<<i<<"  "<<s<< std::endl;
//    }
    for (int i = 1; i <= size; ++i) {
        if(!isInliersBest[i-1])
            continue;
        Vertex3D* Pi = static_cast<Vertex3D*>(opt.vertex(i));
        P1[i-1] = Pi->estimate();
    }

}
}