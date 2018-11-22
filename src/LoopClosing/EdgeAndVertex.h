//
// Created by lhm on 18-5-30.
//

#ifndef DSO_EDGEANDVERTEX_H
#define DSO_EDGEANDVERTEX_H

#include "g2o/core/base_binary_edge.h"
#include "g2o/types/types_seven_dof_expmap.h"
#include "g2o/core/base_vertex.h"

namespace dso{


class EdgeAndVertex {

};

//xyz顶点
class Vertex3D : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vertex3D(): g2o::BaseVertex<3, Eigen::Vector3d>(){}

    virtual bool read(std::istream& is){}//记得将虚函数实例化
    virtual bool write(std::ostream& os) const{}

    virtual void setToOriginImpl() {
        _estimate = Eigen::Vector3d();
    }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<Eigen::Vector3d> update(const_cast<double*>(update_));
        setEstimate(_estimate + update);
    }


};

//xyz与Sim3的边
class BinEdgeSim3 : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, Vertex3D, g2o::VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BinEdgeSim3(): g2o::BaseBinaryEdge<2, Eigen::Vector2d, Vertex3D, g2o::VertexSim3Expmap>(){}
    virtual bool read(std::istream& is){}
    virtual bool write(std::ostream& os) const{}

    void computeError()
    {
        const g2o::VertexSim3Expmap* vs = static_cast<const g2o::VertexSim3Expmap*>(_vertices[1]);
        const Vertex3D* vp = static_cast<const Vertex3D*>(_vertices[0]);

        _error = _measurement - vs->cam_map1(g2o::project(vs->estimate().map(vp->estimate())));
    }

};

class VertexUVD : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexUVD(): g2o::BaseVertex<3, Eigen::Vector3d>(){}

    virtual bool read(std::istream& is){}//记得将虚函数实例化
    virtual bool write(std::ostream& os) const{}

    virtual void setToOriginImpl() {
        _estimate = Eigen::Vector3d();
    }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<Eigen::Vector3d> update(const_cast<double*>(update_));
        setEstimate(_estimate + update);
    }

};

//优化VertexUVD的边
class EdgeProjectP1ToP2 : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexUVD, g2o::VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectP1ToP2():g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexUVD, g2o::VertexSim3Expmap>(){}
    virtual bool read(std::istream& is){}
    virtual bool write(std::ostream& os) const{}

    void computeError()
    {
        const g2o::VertexSim3Expmap* v1 = static_cast<const g2o::VertexSim3Expmap*>(_vertices[1]);
        const VertexUVD* v2 = static_cast<const VertexUVD*>(_vertices[0]);//P1=>3D

        Eigen::Vector3d Puvd = v2->estimate();
        Eigen::Vector3d P1w = Eigen::Vector3d((Puvd[0] - v1->_principle_point1[0])*Puvd[2]/v1->_focal_length1[0],
                                              (Puvd[1] - v1->_principle_point1[1])*Puvd[2]/v1->_focal_length1[1],
                                              Puvd[2]);

        Eigen::Vector2d obs(_measurement);
        _error = obs - v1->cam_map2(g2o::project(v1->estimate().inverse().map(P1w)));
    }
};


class EdgeProjectP2ToP1 : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexUVD, g2o::VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectP2ToP1():g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexUVD, g2o::VertexSim3Expmap>(){}
    virtual bool read(std::istream& is){}
    virtual bool write(std::ostream& os) const{}

    void computeError()
    {
        const g2o::VertexSim3Expmap* v1 = static_cast<const g2o::VertexSim3Expmap*>(_vertices[1]);
        const VertexUVD* v2 = static_cast<const VertexUVD*>(_vertices[0]);//P2-3D

        Eigen::Vector3d Puvd = v2->estimate();
        Eigen::Vector3d P2w = Eigen::Vector3d((Puvd[0] - v1->_principle_point1[0])*Puvd[2]/v1->_focal_length1[0],
                                              (Puvd[1] - v1->_principle_point1[1])*Puvd[2]/v1->_focal_length1[1],
                                               Puvd[2]);

        Eigen::Vector2d obs(_measurement);
        _error = obs-v1->cam_map1(g2o::project(v1->estimate().map(P2w)));
    }
};
}
#endif //DSO_EDGEANDVERTEX_H
