//
// Created by lhm on 18-5-8.
//

#ifndef DSO_OPTIMIZER_H
#define DSO_OPTIMIZER_H

//#include <Eigen/StdVector>
//#include <Eigen/Core>

//#include <vector>//注释掉了jacobian_workspace.h，base_vertex.h，sparse_block_matrix_diagonal.h中头文件#include<Eigen/StdVector>


#include "util/FrameShell.h"
#include "g2o/types/sim3.h"

//#include "g2o/types/se3_ops.h"

namespace dso
{


class Constraint
{
public:
    Constraint(int id1, int id2, const g2o::Sim3 _mean, const Mat77 _information):
            id_1(id1), id_2(id2), mean(_mean), information(_information){}

    int id_1;
    int id_2;
    g2o::Sim3 mean;//T2*T1(-1)
    Mat77 information;
};

class Optimizer {

public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Optimizer(){}

    //回环矫正
    void LoopCorrect(std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>>& v_Sim3,
                     std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>>& v_newSim3,
                     std::list<Constraint>& list_constrint);

    //优化Sim3+3D点
    void OptimizeSim3(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &P1,
                      std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> &P2,
                      g2o::Sim3 &S12, cv::Mat K, std::vector<bool>& isInliersBest);

    //优化Sim3+uvd
    void OptimizeSim3(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &Puvd_1,
                      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &Puvd_2,
                      g2o::Sim3 &S12, cv::Mat K1, cv::Mat K2, std::vector<bool>& isInliersBest, bool isFixed = false);



private:



};



//class UnaryEdgeSim3 : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,g2o::VertexSim3Expmap>
//{
//public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//    UnaryEdgeSim3(Eigen::Vector3d x): BaseUnaryEdge(), _x(x){}
//    virtual bool read(std::istream& is){}
//    virtual bool write(std::ostream& os) const {}
//
//    void computeError()
//    {
//        const g2o::VertexSim3Expmap* S1 = static_cast<g2o::VertexSim3Expmap*>(_vertices[0]);
//        g2o::Sim3 S = S1->estimate();
//        _error = _measurement - S.rotation().matrix()*_x*S.scale() + S.translation();
//    }
//
//private:
//    Eigen::Vector3d _x;
//};
}

#endif //DSO_OPTIMIZER_H
