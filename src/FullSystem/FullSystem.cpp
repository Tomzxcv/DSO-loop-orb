/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include "LoopClosing/ORBextractor.h"

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;



FullSystem::FullSystem()
{

    //read vocabulary own
    printf("读取词典......\n");
    m_vocabulary = DBoW3::Vocabulary ("thirdparty/ORBvoc.txt");
    assert(!m_vocabulary.empty());
    m_dataBase.setVocabulary(m_vocabulary,false,0);

    orb = std::make_shared<ORBextractor>(ORBextractor(600,1.2f,8,50,20));

	loopCandidateId = -1;
	loopCurrentId = -1;

    hasLoop = false;
	stop = false;

	covLog = new std::ofstream();
	covLog->open("logs/covLog.txt", std::ios::trunc | std::ios::out);
	if(covLog) {
		cout << endl << "success to open file" << endl;
		covLog->clear();
	}
	covLog->precision(10);

	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);



	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
        covLog->close(); delete covLog;
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allKeyFramesHistory)
    {
        for(ImmaturePoint * impt : s->mv_immatureKeypoints)
            delete impt;
        delete s;
    }

	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file, bool isKeyFrame)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	if(isKeyFrame)
	{
		for(FrameShell* s : allKeyFramesHistory)
		{
			if(!s->poseValid) continue;

			myfile << s->timestamp <<
				   " " << s->camToWorld.translation().transpose()<<
				   " " << s->camToWorld.so3().unit_quaternion().x()<<
				   " " << s->camToWorld.so3().unit_quaternion().y()<<
				   " " << s->camToWorld.so3().unit_quaternion().z()<<
				   " " << s->camToWorld.so3().unit_quaternion().w() << "\n";
		}
	}
	else
	{
		for(FrameShell* s : allFrameHistory)
		{
			if(!s->poseValid) continue;

			if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

			myfile << s->timestamp <<
				   " " << s->camToWorld.translation().transpose()<<
				   " " << s->camToWorld.so3().unit_quaternion().x()<<
				   " " << s->camToWorld.so3().unit_quaternion().y()<<
				   " " << s->camToWorld.so3().unit_quaternion().z()<<
				   " " << s->camToWorld.so3().unit_quaternion().w() << "\n";
		}
	}
	myfile.close();
}
//backup
//void FullSystem::printResult(std::string file)
//{
//	boost::unique_lock<boost::mutex> lock(trackMutex);
//	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
//
//	std::ofstream myfile;
//	myfile.open (file.c_str());
//	myfile << std::setprecision(15);
//
//	for(FrameShell* s : allFrameHistory)
//	{
//		if(!s->poseValid) continue;
//
//		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;
//
//		myfile << s->timestamp <<
//			   " " << s->camToWorld.translation().transpose()<<
//			   " " << s->camToWorld.so3().unit_quaternion().x()<<
//			   " " << s->camToWorld.so3().unit_quaternion().y()<<
//			   " " << s->camToWorld.so3().unit_quaternion().z()<<
//			   " " << s->camToWorld.so3().unit_quaternion().w() << "\n";
//	}
//	myfile.close();
//}

/*粗略计算fh的位姿，需要用到allFrameHistory的后两帧coarseTracker，和*/
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);



	FrameHessian* lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	if(allFrameHistory.size() == 2)
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());
	else
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
        if(i == 5 && use_ORB_tracking) {
			//boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            SE3 T_cw = computeLastF_2_fhByFeatures(fh, true);
			if(T_cw.data()) {
				lastF_2_fh_this = T_cw * lastF->shell->camToWorld;
				cout << "use orb tracking!" << endl;
			}
			else
				cout<<"orb tracking SE3 is null"<<endl;
        }

		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		if(i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
		}


		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}


        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
    //设置当前frame的位姿、reference和曝光参数
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;//Twi=Twr*Tri


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

//利用fh的信息，更新FrameHessian中点的信息
void FullSystem::traceNewCoarse(FrameHessian* fh, bool isKeyFrame = false)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints)
		{
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;

			if(isKeyFrame)
			{
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD ||
				   ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB)
				{
                    fh->shell->mm_pointsSize[host->shell]++;
                    host->shell->mm_pointsSize[fh->shell]++;
                }
			}
		}
		/*own 追踪特征点*/
		for (ImmaturePoint* ph : host->shell->mv_immatureKeypoints) {
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}




void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}



void FullSystem::activatePointsMT()
{

	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);


	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
//				immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}

			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
//					immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i]=0;
				}
//				immature_notReady_skipped++;
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{

				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=currentMinActDist* ph->my_type)
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}


	for(FrameHessian* host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}


}






void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}



	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					for(PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0;
			}
		}


		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}


void FullSystem::addActiveFrame( ImageAndExposure* image, int id )
{

    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);


	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;

	/*load image to shell own*/
	shell->image_mat = image->image_mat;



	fh->shell = shell;
	allFrameHistory.push_back(shell);


	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);




	if(!initialized)
	{
		// use initializer!
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
		{

			coarseInitializer->setFirst(&Hcalib, fh);
		}
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{

			initializeFromInitializer(fh);
			lock.unlock();
			deliverTrackedFrame(fh, true);
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
		}
		return;
	}
	else	// do front-end operation.
	{
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
		}

		//粗略计算fh位姿
		Vec4 tres = trackNewCoarse(fh);
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }

        //判断是否需要添加关键帧
		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0)
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			needToMakeKF = allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||
					2*coarseTracker->firstCoarseRMSE < tres[0];

		}



        //发布当前帧fh的相机位姿
        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);




		lock.unlock();
		deliverTrackedFrame(fh, needToMakeKF);
		return;
	}
}
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{


	if(linearizeOperation)
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) makeKeyFrame(fh);
		else makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);
	delete fh;
}

void FullSystem::correctLoop(bool isDebug = true)
{
	//====================计算匹配=================
    if(loopCurrentId == -1 || loopCandidateId == -1)
        return;

	std::vector<cv::DMatch> matches;
	computeMatches(allKeyFramesHistory[loopCurrentId]->m_matDescriptor,allKeyFramesHistory[loopCandidateId]->m_matDescriptor,matches);
	if(isDebug) printf("match size = %d\n",(int)matches.size());


	if(matches.size()>=20)
	{
		//==================获取current 和loop的3D点=================
		std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vp3_WorldLoop;
		std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vp3_WorldCurrent;
		std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> vp2_Loop;
        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vp3_PuvdC;
        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vp3_PuvdL;

		if(isDebug)
		{
			std::cout<<"loop candidates depth pt size = "<<allKeyFramesHistory[loopCandidateId]->mv_immatureKeypoints.size()<<
					 "  key size = "<<allKeyFramesHistory[loopCandidateId]->mv_Keypoints.size()<<std::endl;
			std::cout<<"current depth pt size = "<<allKeyFramesHistory[loopCurrentId]->mv_immatureKeypoints.size()<<
					 "  key size = "<<allKeyFramesHistory[loopCurrentId]->mv_Keypoints.size()<<std::endl;
			//输出特征点对
			for(int i=0;i<matches.size();i++)
			{
				cout<<i<<"  loop =  "<<allKeyFramesHistory[loopCandidateId]->mv_Keypoints[matches[i].trainIdx].pt<<"  current =  "<<
					allKeyFramesHistory[loopCurrentId]->mv_Keypoints[matches[i].queryIdx].pt<<endl;
			}
		}
		for(int i=0; i<matches.size(); )
		{

			ImmaturePoint* imptLoop = allKeyFramesHistory[loopCandidateId]->mv_immatureKeypoints[matches[i].trainIdx];
			ImmaturePoint* imptCurrent = allKeyFramesHistory[loopCurrentId]->mv_immatureKeypoints[matches[i].queryIdx];

			if(imptLoop == NULL || std::isnan(imptLoop->idepth_min) ||
			   std::isnan(imptLoop->idepth_max) ||
			   absf(imptLoop->idepth_max - imptLoop->idepth_min)>0.5 ||
			   !imptLoop->idepth_max || !imptLoop->idepth_min)
			{
				cout<<"erase "<<i<<"   "<<imptLoop->idepth_max<<"   "<<imptLoop->idepth_min<<"  "<<
					allKeyFramesHistory[loopCandidateId]->mv_Keypoints[matches[i].trainIdx].pt<<endl;
				matches.erase(matches.begin()+i);
				continue;
			}


			if(imptCurrent == NULL || std::isnan(imptCurrent->idepth_min) ||
			   std::isnan(imptCurrent->idepth_max) ||
			   absf(imptCurrent->idepth_max - imptCurrent->idepth_min)>0.5 ||
			   !imptCurrent->idepth_max || !imptCurrent->idepth_min)
			{
				cout<<"erase "<<i<<"   "<<imptCurrent->idepth_max<<"   "<<imptCurrent->idepth_min<<"  "<<
					allKeyFramesHistory[loopCurrentId]->mv_Keypoints[matches[i].queryIdx].pt<<endl;
				matches.erase(matches.begin()+i);
				continue;
			}

			/*p3 current*/
			cv::Point2f p2current = allKeyFramesHistory[loopCurrentId]->mv_Keypoints[matches[i].queryIdx].pt;
			double idepthC = (imptCurrent->idepth_min + imptCurrent->idepth_max)*0.5f;
			Eigen::Vector3d p3Current((p2current.x-Hcalib.cxl())/(Hcalib.fxl()*idepthC),
									  (p2current.y-Hcalib.cyl())/(Hcalib.fyl()*idepthC),
									  1/idepthC);
			if(isDebug)  cout<<i<<"   p2current = "<<p2current<<endl;
			{
				boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
				p3Current = allKeyFramesHistory[loopCurrentId]->camToWorld*p3Current;
			}
			vp3_WorldCurrent.push_back(p3Current);
            vp3_PuvdC.push_back(Eigen::Vector3d(p2current.x, p2current.y, 1/idepthC));

			/*p3 loop*/
			cv::Point2f p2loop = allKeyFramesHistory[loopCandidateId]->mv_Keypoints[matches[i].trainIdx].pt;
			double idepthL = (imptLoop->idepth_min + imptLoop->idepth_max)*0.5f;
			Eigen::Vector3d p3Loop((p2loop.x-Hcalib.cxl())/(Hcalib.fxl()*idepthL),
								   (p2loop.y-Hcalib.cyl())/(Hcalib.fyl()*idepthL),
								   1/idepthL);
			if(isDebug) cout<<i<<"   p2loop = "<<p2loop<<endl<<endl;
			{
				boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
				p3Loop = allKeyFramesHistory[loopCandidateId]->camToWorld*p3Loop;
			}
			vp3_WorldLoop.push_back(p3Loop);
			vp2_Loop.push_back(Eigen::Vector2d(p2loop.x, p2loop.y));
            vp3_PuvdL.push_back(Eigen::Vector3d(p2loop.x, p2loop.y, 1/idepthL));
			++i;
		}

		//================优化=======================
		if(vp3_WorldCurrent.size()>=10)
		{
			assert(vp3_WorldCurrent.size() == vp3_WorldLoop.size());

			int size = vp3_WorldLoop.size();
			cv::Mat matLoop(3,size,CV_64F), matCurrent(3,size,CV_64F);//3*n

			for (int i = 0; i < 3; ++i) {/*只取3个点*/
				double* Pl = matLoop.ptr<double>(i);
				double* Pc = matCurrent.ptr<double>(i);
				for (int j = 0; j < size; ++j) {
					Pl[j] = vp3_WorldLoop[j][i];
					Pc[j] = vp3_WorldCurrent[j][i];
				}
			}

			if(isDebug)
			{
				printf("找到回环恢复出的loop 3D点为%d个，current 3D点为%d个\n",(int)vp3_WorldLoop.size(),(int)vp3_WorldCurrent.size());
				std::string match = "match "+std::to_string(loopCandidateId)+"--"+std::to_string(loopCurrentId);
				cv::Mat out;
				cv::drawMatches(allKeyFramesHistory[loopCurrentId]->image_mat,allKeyFramesHistory[loopCurrentId]->mv_Keypoints,
								allKeyFramesHistory[loopCandidateId]->image_mat,allKeyFramesHistory[loopCandidateId]->mv_Keypoints,matches,out);
				cv::imshow(match,out);
				cv::waitKey(100);
			}
			cv::Mat K = (cv::Mat_<double>(3,3)<< Hcalib.fxl(), 0, Hcalib.cxl(),
					0,Hcalib.fyl(),Hcalib.cyl(),
					0,0,1);
			if(isDebug)
			{
				cout<<"K = "<<endl<<K<<endl;
				cout<<"match size = "<<matches.size()<<"  point size = "<<matLoop.cols<<"   "<<matCurrent.cols<<endl;
			}

			std::vector<bool> isInliersBest(matCurrent.cols, false);
			/*计算相对Sim3*/
			g2o::Sim3 Scl = ComputeSim3RANSAC(matCurrent,matLoop,10,50,K,isInliersBest);

			if(isDebug)
			{
				//current投影
				printf("\n************************current projection**********************\n");
				SE3 Tcw = allKeyFramesHistory[loopCurrentId]->camToWorld.inverse();
				cv::Mat Rcw_mat = (cv::Mat_<double>(3, 3) <<
                        Tcw.rotationMatrix()(0, 0), Tcw.rotationMatrix()(0, 1), Tcw.rotationMatrix()(0, 2),
						Tcw.rotationMatrix()(1, 0), Tcw.rotationMatrix()(1, 1), Tcw.rotationMatrix()(1, 2),
						Tcw.rotationMatrix()(2, 0), Tcw.rotationMatrix()(2, 1), Tcw.rotationMatrix()(2, 2));

				cv::Mat Tcw_mat = (cv::Mat_<double>(3, 1) <<
                        Tcw.translation()(0), Tcw.translation()(1), Tcw.translation()(2));

				cv::resize(Tcw_mat, Tcw_mat, cv::Size(vp3_WorldLoop.size(), 3));

				cv::Mat Pu_c = K * (Rcw_mat * matCurrent + Tcw_mat);
				for (int i = 0; i < matCurrent.cols; ++i) {
					cout << i << "   " << (Pu_c.col(i) / Pu_c.at<double>(2, i)).t() << endl;
					cout << i << "   " << allKeyFramesHistory[loopCurrentId]->mv_Keypoints[matches[i].queryIdx].pt
						 << endl << endl;
				}

				//loop投影
				printf("\n************************loop projection**********************\n");
				SE3 Tlw = allKeyFramesHistory[loopCandidateId]->camToWorld.inverse();
				cv::Mat Rlw_mat = (cv::Mat_<double>(3, 3) << Tlw.rotationMatrix()(0, 0), Tlw.rotationMatrix()(0,
																											  1), Tlw.rotationMatrix()(
						0, 2),
						Tlw.rotationMatrix()(1, 0), Tlw.rotationMatrix()(1, 1), Tlw.rotationMatrix()(1, 2),
						Tlw.rotationMatrix()(2, 0), Tlw.rotationMatrix()(2, 1), Tlw.rotationMatrix()(2, 2));
				cv::Mat Tlw_mat = (cv::Mat_<double>(3, 1) << Tlw.translation()(0), Tlw.translation()(
						1), Tlw.translation()(2));
				cv::resize(Tlw_mat, Tlw_mat, cv::Size(vp3_WorldLoop.size(), 3));

				cv::Mat Pu_l = K * (Rlw_mat * matLoop + Tlw_mat);
				for (int i = 0; i < matLoop.cols; ++i) {
					cout << i << "   " << (Pu_l.col(i) / Pu_l.at<double>(2, i)).t() << endl;
					cout << i << "   " << allKeyFramesHistory[loopCandidateId]->mv_Keypoints[matches[i].trainIdx].pt
						 << endl << endl;
				}

				//loop=>current
				printf("\n***********************loop to current projection**********************\n");
				SE3 Tcl = SE3(Scl.rotation(), Scl.translation());
				Tcl.rotationMatrix() * Scl.scale();
				Mat44 Tc_cor = Tcl.matrix();//*Tlw.matrix()
				cout << "Tc_cor = " << Tc_cor << endl;

				cv::Mat Rc_cor = (cv::Mat_<double>(3, 3) <<
														 Tc_cor(0, 0), Tc_cor(0, 1), Tc_cor(0, 2),
						Tc_cor(1, 0), Tc_cor(1, 1), Tc_cor(1, 2),
						Tc_cor(2, 0), Tc_cor(2, 1), Tc_cor(2, 2));
				cv::Mat tc_cor = (cv::Mat_<double>(3, 1) << Tc_cor(0, 3), Tc_cor(1, 3), Tc_cor(2, 3));
				cv::resize(tc_cor, tc_cor, cv::Size(vp3_WorldLoop.size(), 3));

				cv::Mat Pu_c_corr = K * (Rc_cor * matLoop + tc_cor);
				for (int i = 0; i < matCurrent.cols; ++i) {
					cout << i << "   " << (Pu_c_corr.col(i) / Pu_c_corr.at<double>(2, i)).t() << endl;
					cout << i << "   " << allKeyFramesHistory[loopCurrentId]->mv_Keypoints[matches[i].queryIdx].pt
						 << endl << endl;
				}


				printf("\n********************after correct Sim3 loop to current projection**********************\n");
				std::cout << "before optimization Sim =  loop " << loopCandidateId << "    current " << loopCurrentId
						  << "    " << std::endl
						  << Scl << std::endl;
			}
			Optimizer *optimizer = new Optimizer();
            /*优化Sim3*/
			g2o::Sim3 Scl_fixed = Scl;
			//optimizer->OptimizeSim3(vp3_WorldCurrent, vp2_Loop, Scl, K, isInliersBest);//优化3D点
            optimizer->OptimizeSim3(vp3_PuvdC,vp3_PuvdL,Scl,K,K,isInliersBest);//优化uvd
			std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vp3_PuvdC_copy;
			std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vp3_PuvdL_copy;
            vp3_PuvdL_copy = vp3_PuvdL;
            vp3_PuvdC_copy = vp3_PuvdC;
			optimizer->OptimizeSim3(vp3_PuvdC_copy,vp3_PuvdL_copy,Scl_fixed,K,K,isInliersBest, true);//优化d

            //优化uvd时误差输出
            double error_duv = 0.0;
            if(isDebug)
            {
                std::cout << "after  optimization Sim =  loop " << loopCandidateId << "    current " << loopCurrentId
                          << "    " << std::endl
                          << Scl << std::endl;
                g2o::Sim3 Slc = Scl.inverse();
                SE3 Tlc = SE3(Slc.rotation(), Slc.translation());
                Tlc.rotationMatrix() * Slc.scale();
                Mat44 Tl_cor_after = Tlc.matrix();//*Tlw.matrix()

                cv::Mat Rc_cor_after = (cv::Mat_<double>(3, 3) <<
                        Tl_cor_after(0, 0), Tl_cor_after(0, 1), Tl_cor_after(0, 2),
                        Tl_cor_after(1, 0), Tl_cor_after(1, 1), Tl_cor_after(1, 2),
                        Tl_cor_after(2, 0), Tl_cor_after(2, 1), Tl_cor_after(2, 2));
                cv::Mat tc_cor_after = (cv::Mat_<double>(3, 1) << Tl_cor_after(0, 3), Tl_cor_after(1, 3), Tl_cor_after(2, 3));

                int inlierNum = 0;
                for (int i = 0; i < vp3_WorldCurrent.size(); ++i) {

                    cv::Mat P3 = (cv::Mat_<double>(3, 1) <<
                                    (vp3_PuvdC[i][0]-K.at<double>(0,2))*vp3_PuvdC[i][2]/K.at<double>(0,0),
                                    (vp3_PuvdC[i][1]-K.at<double>(1,2))*vp3_PuvdC[i][2]/K.at<double>(1,1),
                                     vp3_PuvdC[i][2]);
                    cv::Mat Pu_c_corr_after = K * (Rc_cor_after * P3 + tc_cor_after);
                    cv::Point2d p = allKeyFramesHistory[loopCandidateId]->mv_Keypoints[matches[i].trainIdx].pt;
                    Pu_c_corr_after = Pu_c_corr_after / Pu_c_corr_after.at<double>(2,0);
                    cout << i << "   is inlier = " << isInliersBest[i] << endl;
                    cout << i << "   " << Pu_c_corr_after.t() << endl;
                    cout << i << "   " << p << endl << endl;
                    if(isInliersBest[i])
                    {
                        error_duv += (absf(Pu_c_corr_after.at<double>(0,0) - p.x) + absf(Pu_c_corr_after.at<double>(1,0) - p.y));
                        inlierNum++;
                    }
                }
                error_duv /= (double)inlierNum;
                cout<<"optimize： loop + current + Sim3 error(inliersBest) = "<<error_duv<<"  inliers = "<<inlierNum<<endl<<endl;
            }
			//固定loop-uv，优化d时的误差
            double error_d = 0.0;
			if(isDebug)
			{
				std::cout << "after  optimization Sim =  loop " << loopCandidateId << "    current " << loopCurrentId
						  << "    " << std::endl
						  << Scl_fixed << std::endl;
				g2o::Sim3 Slc = Scl_fixed.inverse();
				SE3 Tlc = SE3(Slc.rotation(), Slc.translation());
				Tlc.rotationMatrix() * Slc.scale();
				Mat44 Tl_cor_after = Tlc.matrix();//*Tlw.matrix()

				cv::Mat Rc_cor_after = (cv::Mat_<double>(3, 3) <<
															   Tl_cor_after(0, 0), Tl_cor_after(0, 1), Tl_cor_after(0, 2),
						Tl_cor_after(1, 0), Tl_cor_after(1, 1), Tl_cor_after(1, 2),
						Tl_cor_after(2, 0), Tl_cor_after(2, 1), Tl_cor_after(2, 2));
				cv::Mat tc_cor_after = (cv::Mat_<double>(3, 1) << Tl_cor_after(0, 3), Tl_cor_after(1, 3), Tl_cor_after(2, 3));

				int inlierNum = 0;
				for (int i = 0; i < vp3_WorldCurrent.size(); ++i) {

					cv::Mat P3 = (cv::Mat_<double>(3, 1) <<
							(vp3_PuvdC_copy[i][0]-K.at<double>(0,2))*vp3_PuvdC_copy[i][2]/K.at<double>(0,0),
							(vp3_PuvdC_copy[i][1]-K.at<double>(1,2))*vp3_PuvdC_copy[i][2]/K.at<double>(1,1),
							vp3_PuvdC_copy[i][2]);
					cv::Mat Pu_c_corr_after = K * (Rc_cor_after * P3 + tc_cor_after);
					cv::Point2d p = allKeyFramesHistory[loopCandidateId]->mv_Keypoints[matches[i].trainIdx].pt;
					Pu_c_corr_after = Pu_c_corr_after / Pu_c_corr_after.at<double>(2,0);
					cout << i << "   is inlier = " << isInliersBest[i] << endl;
					cout << i << "   " << Pu_c_corr_after.t() << endl;
					cout << i << "   " << p << endl << endl;
					if(isInliersBest[i])
					{
						error_d += (absf(Pu_c_corr_after.at<double>(0,0) - p.x) + absf(Pu_c_corr_after.at<double>(1,0) - p.y));
						inlierNum++;
					}
				}
                error_d /= (double)inlierNum;
				cout<<"optimize： current + Sim3 error(inliersBest) = "<<error_d<<"  inliers = "<<inlierNum<<endl<<endl;
			}

            Scl = error_d<error_duv? Scl_fixed:Scl;


			if(isDebug)//优化3D点时误差输出
			{
				std::cout << "after  optimization Sim =  loop " << loopCandidateId << "    current " << loopCurrentId
						  << "    " << std::endl
						  << Scl << std::endl;
				g2o::Sim3 Slc = Scl.inverse();
				SE3 Tlc = SE3(Slc.rotation(), Slc.translation());
				Tlc.rotationMatrix() * Slc.scale();
				Mat44 Tl_cor_after = Tlc.matrix();//*Tlw.matrix()

				cv::Mat Rc_cor_after = (cv::Mat_<double>(3, 3) <<
						Tl_cor_after(0, 0), Tl_cor_after(0, 1), Tl_cor_after(0, 2),
						Tl_cor_after(1, 0), Tl_cor_after(1, 1), Tl_cor_after(1, 2),
						Tl_cor_after(2, 0), Tl_cor_after(2, 1), Tl_cor_after(2, 2));
				cv::Mat tc_cor_after = (cv::Mat_<double>(3, 1) << Tl_cor_after(0, 3), Tl_cor_after(1, 3), Tl_cor_after(2, 3));


				for (int i = 0; i < vp3_WorldCurrent.size(); ++i) {

					cv::Mat P3 = (cv::Mat_<double>(3, 1) << vp3_WorldCurrent[i](0), vp3_WorldCurrent[i](1), vp3_WorldCurrent[i](2));
					cv::Mat Pu_c_corr_after = K * (Rc_cor_after * P3 + tc_cor_after);
					cout << i << "   is inlier = " << isInliersBest[i] << endl;
					cout << i << "   " << (Pu_c_corr_after / Pu_c_corr_after.at<double>(2)).t() << endl;
					cout << i << "   " << allKeyFramesHistory[loopCandidateId]->mv_Keypoints[matches[i].trainIdx].pt
						 << endl << endl;
				}
			}

			/*-*************************新的结构,用于回环矫正*******************************-*/
			int N = loopCurrentId;//allKeyFramesHistory.size() - 1
			std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>> v_oldSim3(N-loopCandidateId+1);
			std::vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3>> v_newSim3(N-loopCandidateId+1);
			{
				boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
				for (int i = loopCandidateId; i <= N; ++i) {
					SE3 Tiw = allKeyFramesHistory[i]->camToWorld.inverse();
					v_oldSim3[i-loopCandidateId] = g2o::Sim3(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
				}
			}
			std::list<Constraint> list_constrint;
			int trans_id = 0;
			for (int i = 1; i < v_oldSim3.size(); ++i) {
				list_constrint.push_back(
						Constraint(trans_id, trans_id+1,
								   v_oldSim3[i]*v_oldSim3[i-1].inverse(),
								   Mat77::Identity()));
				++trans_id;
			}
			list_constrint.push_back(Constraint(trans_id, 0, Scl.inverse(), Mat77::Identity()));
			//list_constrint.push_back(Constraint(loopCurrentId-loopCandidateId, 0, Scl.inverse(), Mat77::Identity()));

			if(isDebug)  cout<<"old Sim3 size = "<<v_oldSim3.size()<<"  new Sim3 size = "<<v_newSim3.size()<<endl;

			//=============优化====================
			runMapping = false;
			{
				boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

				optimizer->LoopCorrect(v_oldSim3, v_newSim3, list_constrint);
				//stop = true;

				if(isDebug)
				{
                    printf("*********************old Sim3 and new Sim3*************************\n");
					for (int i = 0; i < v_oldSim3.size(); ++i) {
						cout<<i<<endl;
						cout<<"v_oldSim3 = "<<v_oldSim3[i].log().transpose()<<endl;
						cout<<"v_newSim3 = "<<v_newSim3[i].log().transpose()<<endl<<endl;
					}
				}
				/*更新allKyeFramesHistory中的pose，并且转换要显示的keyframes中的pose，显示*/
				for (int i = loopCandidateId; i <= N; ++i) {
					SE3 Tiw(v_newSim3[i-loopCandidateId].rotation(),
							v_newSim3[i-loopCandidateId].translation()/v_newSim3[i-loopCandidateId].scale());

                    cout<<allKeyFramesHistory[i]->id<<endl;
                    cout<<"before Twc = "<<allKeyFramesHistory[i]->camToWorld.log().transpose()<<endl;
					allKeyFramesHistory[i]->camToWorld = Tiw.inverse();
                    cout<<"after  Twc = "<<allKeyFramesHistory[i]->camToWorld.log().transpose()<<endl<<endl;
					outputWrapper[0]->setKFiPose(i,Tiw);//显示的关键帧位姿
					outputWrapper[0]->setCurrentPose(Tiw);//设置当前相机位姿
				}

				cout<<"all = "<<allKeyFramesHistory.size()<<" output size = "<<outputWrapper.size()<<endl;
				cout<<"all frame= "<<outputWrapper[0]->getAllFramePosesSize()<<" all KeyFrame = "<<outputWrapper[0]->getKeyframesSize()<<endl;


				/*更新普通帧allFrameHistory，进行位姿追踪时用*/
				{
					boost::unique_lock<boost::mutex> lock(trackMutex);
					//更新最后的5帧
					int N = allFrameHistory.size();
					for(int i=1; i<= 5; ++i)
					{
						FrameShell* fh = allFrameHistory[N-i];
						allFrameHistory[N-i]->camToWorld = fh->trackingRef->camToWorld * fh->camToTrackingRef;
					}
				}
				/*更新Hessian中用于计算雅克比的位姿信息*/
				{
                    for (auto fH : frameHessians)
                    {
                        fH->PRE_camToWorld = fH->shell->camToWorld;
                        fH->PRE_worldToCam = fH->PRE_camToWorld.inverse();
                    }
				}


				/*矫正worldToCam_evalPT，这个是干嘛用的?*/
				for (int i = 0; i < frameHessians.size(); ++i) {
					frameHessians[i]->setEvalPT_scaled(frameHessians[i]->shell->camToWorld.inverse(),
													   frameHessians[i]->shell->aff_g2l);
				}

                if(isDebug)
                {
                    //输出误差
                    printf("\n************************error********************************\n");
                    for (auto iter = list_constrint.begin(); iter!=list_constrint.end(); iter++) {

                        g2o::Sim3 error_old=iter->mean*v_oldSim3[iter->id_1]*v_oldSim3[iter->id_2].inverse();
                        g2o::Sim3 error_new=iter->mean*v_newSim3[iter->id_1]*v_newSim3[iter->id_2].inverse();
                        cout<<"old = "<<error_old.log().transpose()<<endl;
                        cout<<"new = "<<error_new.log().transpose()<<endl<<endl;
                    }
                    printf("****************over*********************************\n");
                }

			}

			runMapping = true;
            KF_num = 0;
		}

        else  hasLoop = false;

	}
    else  hasLoop = false;


}


void FullSystem::makeKeyFrame( FrameHessian* fh) {
    KF_num++;

//	for(ImmaturePoint* impt : allKeyFramesHistory.back()->mv_immatureKeypoints)
//	{
//		cout<<"min idepth = "<<impt->idepth_min<<"  max idepth = "<<impt->idepth_max<<endl;
//	}

    cout << "KF_num = " << KF_num << endl;

//	if(KF_num == 20) {
//        SE3 T_cw = computeLastF_2_fhByFeatures(fh, true);
//        cout << T_cw << endl;
//    }

    /*计算当前图像与回环图像之间的匹配关系与sim3*/
    if(hasLoop)//&& abs(loopCurrentId - allKeyFramesHistory.size())>3
    {
        cout<<"last KF location = "<<allKeyFramesHistory.back()<<endl;
        cout<<"reference location = "<<fh->shell->trackingRef<<endl;
        cout<<"loop = "<<loopCandidateId<<"  current = "<<loopCurrentId<<endl;
        cout<<"bake fm = "<<allKeyFramesHistory.back()->id<<endl;
        cout<<"loopCurrent id = "<<allKeyFramesHistory[loopCurrentId]->id<<endl;
        cout<<"Tracker id = "<<coarseTracker->lastRef->shell->id<<endl;
		cout<<"NweKF = "<<coarseTracker_forNewKF->lastRef->shell->id<<endl;
		cout<<"fh id = "<<fh->shell->id<<endl;

        int s = allFrameHistory.size() -1;
        for(int i=s;i>s-3;--i)
            cout<<"before  fh "<<i<<" = "<<allFrameHistory[i]->camToWorld.log().transpose()<<endl;
        cout<<"before  fh Toreference pose = "<<fh->shell->camToTrackingRef.log().transpose()<<endl;
        cout<<"before  fh reference id = "<<fh->shell->trackingRef->id<<endl;
        cout<<"before  fh reference pose = "<<fh->shell->trackingRef->camToWorld.log().transpose()<<endl;
        cout<<"before  fh pose = "<<fh->shell->camToWorld.log().transpose()<<endl;


        correctLoop();
        hasLoop = false;
		loopCurrentId = loopCandidateId = -1;

        /*需要重新估计fh的位姿*/
        {
            Vec4 tres = trackNewCoarse(fh);
            if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
            {
                printf("---Initial Tracking failed: LOST!\n");
                isLost=true;
                return;
            }
            //更新allFramePose中fh的位姿
			Vec3 t = fh->shell->camToWorld.translation();
            outputWrapper[0]->setNewfhPose(Vec3f(t.x(), t.y(), t.z()));
        }
        for(int i=s;i>s-3;--i)
            cout<<"after   fh "<<i<<" = "<<allFrameHistory[i]->camToWorld.log().transpose()<<endl;
		cout<<"after  fh Toreference pose = "<<fh->shell->camToTrackingRef.log().transpose()<<endl;
        cout<<"after  fh reference id = "<<fh->shell->trackingRef->id<<endl;
        cout<<"after  fh reference pose = "<<fh->shell->trackingRef->camToWorld.log().transpose()<<endl;
		cout<<"after  fh pose = "<<fh->shell->camToWorld.log().transpose()<<endl;
    }
    //extract ORB in KeyFrame own
    orb->operator()(fh->shell->image_mat,cv::Mat(),fh->shell->mv_Keypoints,fh->shell->m_matDescriptor);

	//make immatureKeypoint
	fh->shell->mv_immatureKeypoints.resize(fh->shell->mv_Keypoints.size());
	//printf("size = %d\n",fh->shell->mv_immatureKeypoints.size());
	for(int i=0; i<fh->shell->mv_Keypoints.size();i++)
	{
		ImmaturePoint* impt = new ImmaturePoint(fh->shell->mv_Keypoints[i].pt.x,fh->shell->mv_Keypoints[i].pt.y,fh, 1.0, &Hcalib);
		fh->shell->mv_immatureKeypoints[i] = impt;
	}

	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh, true);


	boost::unique_lock<boost::mutex> lock(mapMutex);


    //=====================convert the descriptor to DBowVector and DataBases================
    m_vocabulary.transform(fh->shell->m_matDescriptor,fh->shell->m_BowVec);

    //=====================detect the loop candidates ======================================
	//FrameHessian* fhLast = fh;
    if(allKeyFramesHistory.size() > 10 && !hasLoop && KF_num>50) {
		detectLoop(fh);
	}
    m_dataBase.add(fh->shell->m_matDescriptor);


	// =========================== Flag Frames to be Marginalized. =========================
	flagFramesForMarginalization(fh);


	// =========================== add New Frame to Hessian Struct. =========================
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();

	//更新协方差矩阵
	updataCov(fh);

	//显示追踪点数量
//	for(FrameShell* fms : allKeyFramesHistory)
//	{
//		cout<<endl<<"id = "<<fms->id<<"current = "<<fms<<endl;
//		cout<<"map size = "<<fms->mm_pointsSize.size()<<endl;
//		for(auto i = fms->mm_pointsSize.begin();i!=fms->mm_pointsSize.end();++i)
//			cout<<"frame = "<<i->first<<"    pointSize = "<<i->second<<endl;
//	}


	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}


	// =========================== Activate Points (& flag for marginalization). =========================
	activatePointsMT();
	ef->makeIDX();

	// =========================== OPTIMIZE ALL =========================

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);


	// =========================== Figure Out if INITIALIZATION FAILED =========================
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}



    if(isLost) return;


	// =========================== REMOVE OUTLIER =========================
	removeOutliers();


	{//更新Tracker
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		coarseTracker_forNewKF->makeK(&Hcalib);
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}


	debugPlot("post Optimize");



	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();
	ef->dropPointsF();
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	ef->marginalizePointsF();



	// =========================== add new Immature points & new residuals =========================
	makeNewTraces(fh, 0);

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }

	//printf("+++++++pointSize = %d+++++++++++++++++++++++\n\n",(int)frameHessians.back()->pointHessians.size());

	// =========================== Marginalize Frames =========================

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
            {
				marginalizeFrame(frameHessians[i], covLog);
				i=0;
            }


	//printLogLine();
    //printEigenValLine();

}



g2o::Sim3 FullSystem::ComputeSim3RANSAC(cv::Mat& P1, cv::Mat& P2, int minInliers, int maxIterations,cv::Mat& K, std::vector<bool>& isInliersBest)//S12
{
    assert(P1.data && P2.data);
    int N = P1.cols;

    float epsilon = (float)minInliers/N;
    int IterNum = ceil(log(1-0.95)/log(1-pow(epsilon,3)));//迭代次数

    int maxIterationNum = maxIterations<IterNum?maxIterations:IterNum;

    int inliersNumBset = 0;
    double errorBest = 10000000.0;

    cv::Mat P1_3(3,3,CV_64F);
    cv::Mat P2_3(3,3,CV_64F);
    cv::RNG rng;
    int iterTimes = 0;
    /*迭代*/
    while(iterTimes<maxIterationNum)
    {
        iterTimes++;
        std::vector<bool> isInliers(N,false);
        int inliersNum = 0;
        double error = 0.0;
        //std::cout<<"iterTimes = "<<iterTimes<<std::endl;

        //选择最小集合
        for (int i = 0; i < 3; ++i) {
            double k = rng.uniform(0.0,(double)(N-1));
            //std::cout<<" k "<<i<<" = "<<k<<std::endl;
            P1.col((int)k).copyTo(P1_3.col(i));
            P2.col((int)k).copyTo(P2_3.col(i));
        }

        cv::Mat T12 = ComputeSim3(P1_3,P2_3);

        /*检查内点*/
        cv::Mat R12 = T12.colRange(0,3).rowRange(0,3);
        cv::Mat t12 = T12.colRange(3,4).rowRange(0,3);
        cv::resize(t12,t12,cv::Size(N,3));
        //std::cout<<"R12 = "<<R12<<std::endl;
        //std::cout<<"T12 = "<<t12<<std::endl;
        //std::cout<<"P1_3 = "<<P1_3<<std::endl;
        //std::cout<<"P2_3 = "<<P2_3<<std::endl;
        //std::cout<<"P1 size = "<<P1.size()<<"  K size = "<<K.size()<<"  R12 size = "<<R12.size()<<"  t12 size = "<<t12.size()<<std::endl;
        cv::Mat Pu1 = P1;
        cv::Mat Pu2to1 = (R12*P2+t12);//project P2 in P1
        //std::cout<<"Pu1 = "<<Pu1<<std::endl;
        //std::cout<<"Pu2to1 = "<<Pu2to1<<std::endl;
        //std::cout<<"Pu1 size = "<<Pu1.size()<<"  Pu2to1 size = "<<Pu2to1.size()<<std::endl;
        assert(Pu1.cols == Pu2to1.cols && Pu2to1.cols == N);
        for(int i=0; i<N; i++)
        {
//            std::cout<<"Pu1 = "<<i<<Pu1.col(i).t()<<std::endl;
//            std::cout<<"Pu2 = "<<i<<Pu2to1.col(i).t()<<std::endl;

            double e = cv::norm(Pu1.col(i)-Pu2to1.col(i),cv::NORM_L2);
            //std::cout<<"e->"<<i<<"  "<<e<<std::endl;
            if(e < 1.2)//threhold 0.5
            {
                isInliers[i] = true;
                inliersNum++;
                error += e;
            }
        }
        error /= inliersNum;
        printf("**  *iterTimes = %d, error = %f,  inliers num = %d*****************\n",iterTimes,error,inliersNum);

        if(inliersNum > inliersNumBset)//error < errorBest && inliersNum > 3
        {
            isInliersBest = isInliers;
            inliersNumBset = inliersNum;
            errorBest = error;
        }

        if(inliersNum > 0.8*N)
            iterTimes += 10;

    }

    /*用所有内点，重新计算Sim3*/
    if(inliersNumBset < 3)
    {
        printf("there are two few inliners\n");
        return g2o::Sim3();
    } else{

        cv::Mat P1best(3,inliersNumBset,CV_64F);
        cv::Mat P2best(3,inliersNumBset,CV_64F);
        for (int i = 0,j=0; i < N; ++i) {
            if(isInliersBest[i])
            {
                P1.col(i).copyTo(P1best.col(j));
                P2.col(i).copyTo(P2best.col(j));
                j++;
            }
        }
        cv::Mat bestP12 = ComputeSim3(P1best,P2best);
        //std::cout<<"bestP12 = "<<bestP12<<std::endl;

        cv::Mat sR = bestP12.colRange(0,3).rowRange(0,3);
        cv::Mat t = bestP12.colRange(3,4).rowRange(0,3);

        double scale = cv::norm(sR.col(0));//和算出来的scale不同
        sR = sR/scale;


        Mat33 R12;
        R12 <<  sR.at<double>(0,0),sR.at<double>(0,1),sR.at<double>(0,2),
                sR.at<double>(1,0),sR.at<double>(1,1),sR.at<double>(1,2),
                sR.at<double>(2,0),sR.at<double>(2,1),sR.at<double>(2,2);
        Eigen::Vector3d T12(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));

        printf("\n**********************************************************\n");
        printf("best Sim R det = %f,  scale =  %f,  error = %f,  inlinersNum = %d, allNum = %d \n",cv::determinant(sR),scale,errorBest,inliersNumBset,P1.cols);


        std::cout<<"T12 best points====================="<<std::endl;

        cv::resize(t,t,cv::Size(inliersNumBset,3));
        //cout<<"sR"<<sR<<endl<<"t"<<t<<endl;
        cout<<"R12"<<R12<<endl<<"T12"<<T12<<endl;
        cv::Mat P1_2 = sR*P2best + t;
        for(int i=0,j=0;i<P1.cols; i++)
        {
            if(isInliersBest[i])
            {
                std::cout<<i<<" P1   "<<P1best.col(j).t()<<std::endl;
                std::cout<<i<<" P1_2 "<<P1_2.col(j).t()<<std::endl<<std::endl;
                j++;
            }
        }

        return g2o::Sim3(R12,T12,scale);
    }
}


cv::Mat FullSystem::ComputeSim3(cv::Mat& P1, cv::Mat& P2)
{
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    //-1- compute centroid and relative coordinates
    assert(P1.type() == P2.type() && P1.cols>=3 && P2.cols>=3);

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

    cv::Mat R12(3,3,P1.type());
    cv::Rodrigues(RotationVec,R12);

    //-5- Rotate set 2
    cv::Mat P3 = R12*Pr2;

    //-6- compute scale
    double A = Pr1.dot(P3);
    cv::Mat Pr2Square = Pr2*Pr2.t();
    double B = cv::trace(Pr2Square)(0);
    double scale = A/B;


    //-7- compute translation
    cv::Mat t12(3,1,P2.type());
    t12 = O1 - scale*R12*O2;

    //-8- transformation
    cv::Mat T12 = cv::Mat::eye(4,4,P1.type());
    cv::Mat sR = scale*R12;


    sR.copyTo(T12.rowRange(0,3).colRange(0,3));
    t12.copyTo(T12.rowRange(0,3).col(3));


    printf("computeSim3 scale = %f  ,det R = %f\n",scale,cv::determinant(sR)/(pow(scale,3)));

//    std::cout<<"T12====================="<<std::endl;
//    cv::resize(t12,t12,cv::Size(P2.cols,3));
//    cv::Mat P1_ = sR*P2 + t12;
//    for(int i=0;i<P1.cols; i++)
//    {
//        std::cout<<i<<" P1 "<<P1.col(i).t()<<std::endl;
//        std::cout<<i<<" P1_2 "<<P1_.col(i).t()<<std::endl<<std::endl;
//    }



//    //-8- Sim3
//    Mat33 R;
//    R <<  R12.at<double>(0,0),R12.at<double>(0,1),R12.at<double>(0,2),
//          R12.at<double>(1,0),R12.at<double>(1,1),R12.at<double>(1,2),
//          R12.at<double>(2,0),R12.at<double>(2,1),R12.at<double>(2,2);
//    Eigen::Vector3d t(t12.at<double>(0,0),t12.at<double>(1,0),t12.at<double>(2,0));
//
//    g2o::Sim3 S12(R,t,scale1);
    //std::cout<<"T12"<<T12<<std::endl;
    return T12;

}

void FullSystem::ChangePoints(const cv::Mat& P, cv::Mat& Pr, const cv::Mat& O)
{
    assert(P.data && Pr.data && O.data);

    cv::reduce(P,O,1,CV_REDUCE_AVG);

    for(int i=0; i<P.cols; ++i)
    {
        Pr.col(i) = P.col(i) - O;
    }
}


void FullSystem::detectLoop(FrameHessian* fhLast)
{
	int size = allKeyFramesHistory.size();
	double score1 = m_vocabulary.score(fhLast->shell->m_BowVec,allKeyFramesHistory[size-2]->m_BowVec);
	double score2 = m_vocabulary.score(fhLast->shell->m_BowVec,allKeyFramesHistory[size-3]->m_BowVec);
	double score = score1<score2?score1:score2;
	printf("***************%d*************score = %f********************************\n",fhLast->shell->id,score);

	/*计算score*/
	DBoW3::QueryResults queryResults;
	m_dataBase.query(fhLast->shell->m_BowVec,queryResults,20);//max_results返回最大结果数量，max_id输出最大的结果数量
	//std::cout<<queryResults<<std::endl;

	/*回环候选图像筛选*/
	std::vector<int> keyFrameNum;
	for(int i=0;i<20;i++)
	{
		if(queryResults[i].Score < score)
			break;
		else
		{
			if( abs(queryResults[i].Id-size) > 10 )
				keyFrameNum.push_back(queryResults[i].Id);
		}
	}

	std::sort(keyFrameNum.begin(),keyFrameNum.end());

	int continueNum = 0;
	if(keyFrameNum.size() >= 3)
	{
		for(int i=1;i<keyFrameNum.size();++i)
		{
			if(abs(keyFrameNum[i]-keyFrameNum[i-1]) <= 2)
			{
				continueNum++;
				if(continueNum == 3)//连续几帧相似为回环
				{
					loopCandidateId = keyFrameNum[i];
					loopCurrentId = size;
                    hasLoop = true;
					break;
				}

			}
			else
				continueNum = 0;
		}
	}
	if(hasLoop)
	{
//			std::string m = "oldFrame"+std::to_string(loopCandidateId);
//            cv::imshow(m,allKeyFramesHistory[loopCandidateId]->image_mat);
//            cv::waitKey(20);
//			std::string n = "curFrame"+std::to_string(size);
//            cv::imshow(n,fhLast->shell->image_mat);
//            cv::waitKey(20);
		std::printf("loop detection*****loopCandidateId = %d*****currentId = %d*****\n",loopCandidateId,loopCurrentId);
	}
}


void FullSystem::computeMatches(cv::Mat& mat_query, cv::Mat& mat_train,std::vector<cv::DMatch>& goodMatches)
{
	assert(mat_query.data);
	assert(mat_train.data);
    //匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(mat_query,mat_train,matches);

	if(matches.size() == 0)
		return;
    int min = 1000;

    for(cv::DMatch i : matches)
    {
        if(i.distance < min)
            min = i.distance;
    }
    printf("描述子最小距离为%d\n",min);
	int threshold = 30<(3*min)?30:(3*min);
	//去除trainIdx中重复元素
	if(matches[0].distance < threshold)
		goodMatches.push_back(matches[0]);
    for(int i=1; i<matches.size(); ++i)
    {
        if(matches[i].distance < threshold && matches[i].trainIdx != matches[i-1].trainIdx)
            goodMatches.push_back(matches[i]);
    }
//    cv::Mat out;
//    cv::drawMatches(query->image_mat,query->mv_Keypoints,train->image_mat,train->mv_Keypoints,goodMatches,out);
//    cv::imshow("match",out);
//    cv::waitKey(20);
	return;
}

void FullSystem::saveAllKeyFrames()
{
	static int num = 0;
	std::ofstream* os= new std::ofstream();
	os->open("logs/keyframes.txt", std::ios::trunc | std::ios::out);
	if(os) {
		cout << endl << "success to open keyframes.txt file" << endl;
		os->clear();
	}
	os->precision(10);

	boost::unique_lock<boost::mutex> lock(mapMutex);
	for(FrameShell* fm : allKeyFramesHistory)
        if(fm && fm->trackingRef)
		    (*os)<<"第"<<num++<<"个  "<<"current id = "<<fm->id<<"   reference id = "<<fm->trackingRef->id<<endl;

	os->close();
	delete os;
}


//updata all keyframes‘ cov in frameHessians
void FullSystem::updataCov(FrameHessian* fh)
{
	//compute cov
	if (fh->shell->trackingRef != NULL) {

		(*covLog) << "current id = " << fh->shell->id << "   reference id = "
				  << fh->shell->trackingRef->id << endl;
		(*covLog) << "before reference cov = "
				  << fh->shell->trackingRef->m_cov(0, 0) << "  "
				  << fh->shell->trackingRef->m_cov(1, 1) << "  "
				  << fh->shell->trackingRef->m_cov(2, 2) << "  "
				  << fh->shell->trackingRef->m_cov(3, 3) << "  "
				  << fh->shell->trackingRef->m_cov(4, 4) << "  "
				  << fh->shell->trackingRef->m_cov(5, 5) << "  " << endl;

		Mat66 adj = fh->shell->camToTrackingRef.inverse().Adj();

		fh->shell->m_cov = Mat66::Identity() / 20. + adj * fh->shell->trackingRef->m_cov * adj.transpose();

		(*covLog) << "after cov = "
				  << fh->shell->m_cov(0, 0) << "  "
				  << fh->shell->m_cov(1, 1) << "  "
				  << fh->shell->m_cov(2, 2) << "  "
				  << fh->shell->m_cov(3, 3) << "  "
				  << fh->shell->m_cov(4, 4) << "  "
				  << fh->shell->m_cov(5, 5) << "  " << endl << endl << endl;



	} else
		(*covLog) << fh->shell->id << "  reference is null" << endl;

	return;
}


SE3 FullSystem::computeLastF_2_fhByFeatures(FrameHessian* fh, bool showDebug)
{
	if(fh == nullptr)
		return SE3();

	//extract ORB in current frame
	orb->operator()(fh->shell->image_mat,cv::Mat(),fh->shell->mv_Keypoints,fh->shell->m_matDescriptor);


	//make descriptor of all frameHessians
    assert(frameHessians.size() >= 1);
    std::vector<int> vIndex(frameHessians.size());
	cv::Mat all_frame_descriptor;
    frameHessians[0]->shell->m_matDescriptor.copyTo(all_frame_descriptor);
    vIndex[0] = frameHessians[0]->shell->m_matDescriptor.rows - 1;

	if(!all_frame_descriptor.data)
		return SE3();

	for(int i=1; i<frameHessians.size(); i++){
		cv::copyMakeBorder(all_frame_descriptor, all_frame_descriptor,0,frameHessians[i]->shell->m_matDescriptor.rows,0,0,cv::BORDER_CONSTANT);

        frameHessians[i]->shell->m_matDescriptor.copyTo(
                all_frame_descriptor.rowRange(
                        all_frame_descriptor.rows - frameHessians[i]->shell->m_matDescriptor.rows,
                        all_frame_descriptor.rows));
        vIndex[i] = vIndex[i-1] + frameHessians[i]->shell->m_matDescriptor.rows;

		if(showDebug)
        	cout<<"vIndex "<<i<<" = "<<vIndex[i]<<endl;
	}
	if(showDebug)
    	cout<<"all_frame_descriptor size = "<<all_frame_descriptor.rows<<"  "<<all_frame_descriptor.cols<<endl;

    std::vector<cv::DMatch> goodMatches;
    computeMatches(fh->shell->m_matDescriptor, all_frame_descriptor, goodMatches);

	if(showDebug) {
		cout << "find " << goodMatches.size() << " matches" << endl;
		for (auto i : goodMatches) {
			cout << "trainId = " << i.trainIdx << "  queryId = " << i.queryIdx << "  distance = " << i.distance << endl;
			cout << "train = " << all_frame_descriptor.row(i.trainIdx) << "  query = "
				 << fh->shell->m_matDescriptor.row(i.queryIdx) << endl;
		}
	}
    cout<<"*****************************************************************"<<endl;

	std::vector<cv::Point2f> vp2(goodMatches.size());
	std::vector<cv::Point3f> vp3(goodMatches.size());

    std::vector<std::vector<cv::DMatch>> all_matches(frameHessians.size());
    for(int i=0; i<goodMatches.size(); i++){
        int train_id = goodMatches[i].trainIdx - 1;
        int i_index = 0;
        while (train_id >= vIndex[i_index])
            i_index++;


		int temp_trainId = train_id;
		if(i_index > 0)
			temp_trainId = train_id - vIndex[i_index - 1];

		cout<<"resize queryId = "<<goodMatches[i].queryIdx<<"  trainId = "<<temp_trainId<<endl;
        all_matches[i_index].push_back(cv::DMatch(goodMatches[i].queryIdx, temp_trainId, goodMatches[i].distance));

		vp2[i] = fh->shell->mv_Keypoints[goodMatches[i].queryIdx].pt;
		cv::Point2f p2 = frameHessians[i_index]->shell->mv_Keypoints[temp_trainId].pt;
		double idepthL = (frameHessians[i_index]->shell->mv_immatureKeypoints[temp_trainId]->idepth_min +
				frameHessians[i_index]->shell->mv_immatureKeypoints[temp_trainId]->idepth_max)*0.5f;
		Eigen::Vector3d p3current((p2.x-Hcalib.cxl())/(Hcalib.fxl()*idepthL),
							      (p2.y-Hcalib.cyl())/(Hcalib.fyl()*idepthL),
							       1/idepthL);
		{
			//boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			Eigen::Vector3d p3world = frameHessians[i_index]->shell->camToWorld * p3current;
			vp3[i] = cv::Point3f(p3world[0], p3world[1], p3world[2]);
		}

		cout<<i<<"  p2 = "<<vp2[i].x<<"  "<<vp2[i].y<<"  "<<"  p3 = "<<vp3[i].x<<"  "<<vp3[i].y<<"  "<<vp3[i].z<<endl;
    }

    cout<<"have 3d points size = "<<vp3.size()<<"  2d points size = "<<vp2.size()<<endl;

    if(goodMatches.size() < 10)
        return SE3();

    cv::Mat K = (cv::Mat_<double>(3,3)<< Hcalib.fxl(), 0, Hcalib.cxl(),
                                        0,Hcalib.fyl(),Hcalib.cyl(),
                                        0,0,1);
    cv::Mat rvec, tvec;
    cv::solvePnPRansac(vp3, vp2, K, cv::Mat(), rvec, tvec);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Mat33 R_mat;
	R_mat << R.at<double>(0,0), R.at<double>(0,1),R.at<double>(0,2),
			 R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
			 R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);
    SE3 T_cw(R_mat, Eigen::Vector3d(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0)));
	cout<<"R = "<<R<<endl<<"R_mat"<<R_mat<<endl;

	if(showDebug) {
//		for (int i = 0; i < all_matches.size(); i++)
//			for (int j = 0; j < all_matches[i].size(); j++) {
//				cout << "i = " << i << "  trainId = " << all_matches[i][j].trainIdx << "  queryId = "
//					 << all_matches[i][j].queryIdx << "  distance = " << all_matches[i][j].distance << endl;
//				cout << "train = " << frameHessians[i]->shell->m_matDescriptor.row(all_matches[i][j].trainIdx)
//					 << "  query = "
//					 << fh->shell->m_matDescriptor.row(all_matches[i][j].queryIdx) << endl;
//			}
		//cout << 3 << endl << frameHessians[3]->shell->m_matDescriptor << endl;

		int id = 0;
		cv::namedWindow("match");
		for (auto f : frameHessians) {
			cv::Mat out;
			cv::drawMatches(fh->shell->image_mat, fh->shell->mv_Keypoints, f->shell->image_mat, f->shell->mv_Keypoints,
							all_matches[id++], out);
			cv::imshow("match", out);
			cv::waitKey();
		}
        cv::destroyWindow("match");
	}

	cout<<"T_cw = "<<T_cw.unit_quaternion().x()<<"  "
		<<T_cw.unit_quaternion().y()<<"  "
		<<T_cw.unit_quaternion().z()<<"  "
		<<T_cw.unit_quaternion().w()<<"  "
		<<T_cw.translation()[0]<<"  "
		<<T_cw.translation()[1]<<"  "
		<<T_cw.translation()[2]<<endl;

	return T_cw;
}

void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	firstFrame->idx = frameHessians.size();
	frameHessians.push_back(firstFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);


	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		if(rand()/(float)RAND_MAX > keepPercentage) continue;

		Pnt* point = coarseInitializer->points[0]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }


		pt->idepth_max=pt->idepth_min=1;
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

		ph->setIdepthScaled(point->iR*rescaleFactor);
		ph->setIdepthZero(ph->idepth);
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);

		firstFrame->pointHessians.push_back(ph);
		ef->insertPoint(ph);
	}



	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);


	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
	{
		int i = x+y*wG[0];
		if(selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
		if(!std::isfinite(impt->energyTH)) delete impt;
		else newFrame->immaturePoints.push_back(impt);

	}
	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}



void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
	}

	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}





}
