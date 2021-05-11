// Copyright (C) 2020 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen ;


namespace StOpt
{

/// \brief Calculate local sums as explained in  Langrené-Warin 2020
/// \param p_x particules (sample)  size : (dimension, nbSim)
/// \param p_z  rectilinear points
/// \param p_y  estimate for each p_x point
/// \param p_nbPtZ  helper to store index in local sum
/// \param p_nbPtZT  total number of local sum
ArrayXd localSumCalc(const ArrayXXd &p_x,
                     const vector< shared_ptr<ArrayXd> >    &p_z,
                     const ArrayXd &p_y,
                     const ArrayXi   &p_nbPtZ, const  int &p_nbPtZT)
{
    ArrayXXi iSort(p_x.rows(), p_x.cols());
    ArrayXXd sX(p_x.rows(), p_x.cols());
    for (int id = 0 ; id <  p_x.rows(); ++id)
    {
        vector<pair<double, int> > toSort(p_x.cols());
        for (int is = 0; is < p_x.cols(); ++is)
            toSort[is] = make_pair(p_x(id, is), is);
        sort(toSort.begin(), toSort.end());
        for (int is = 0; is < p_x.cols(); ++is)
        {
            iSort(id, is) = toSort[is].second;
            sX(id, is) = toSort[is].first;
        }
    }
    ArrayXXi idx(p_x.rows(), p_x.cols());
    // loop on dimension
    for (int id = 0; id < p_x.rows(); ++id)
    {
        int xidx = 0;
        int zidx = 0;
        // select only important
        while ((xidx < p_x.cols()) && (sX(id, xidx) <= (*p_z[id])(p_z[id]->size() - 1)))
        {
            if (sX(id, xidx) <= (*p_z[id])(zidx))
            {
                idx(id, iSort(id, xidx)) = zidx;
                xidx += 1;
            }
            else
            {
                zidx += 1;
            }
        }
        for (int ix = xidx; ix < p_x.cols(); ++ix)
        {
            idx(id, iSort(id, ix)) = -1;
        }
    }
    // return
    ArrayXd localSum = ArrayXd::Zero(p_nbPtZT);
    for (int is = 0; is < p_x.cols(); ++is)
    {
        // index
        if (idx.col(is).minCoeff() >= 0)
        {
            int iindex = idx(0, is) ;
            for (int id = 1; id < p_x.rows() ; ++id)
                iindex += idx(id, is) * p_nbPtZ(id);
            localSum(iindex) += p_y(is);
        }
    }
    localSum /= p_x.cols();
    return localSum;
}

/// \brief Calculate fast CDF  using Langrené Warin 2020 method
/// \param p_x  particules (sample)  size : (dimension, nbSim)
/// \param p_z  rectilinear points
/// \param p_y  estimate for each p_x point
/// \return  for each point of the grid return the CDF
ArrayXd fastCDF(const ArrayXXd &p_x, const vector< shared_ptr<ArrayXd> >    &p_z, const ArrayXd &p_y)
{
    // store nbpt per dimension before
    ArrayXi nbPtZ(p_z.size() + 1);
    nbPtZ(0) = 1;
    for (size_t id = 0; id < p_z.size(); ++id)
        nbPtZ(id + 1) = nbPtZ(id) * p_z[id]->size();
    // nb point
    int nbPtZT = nbPtZ(p_z.size());
    // helper
    ArrayXi nbPtZInv(p_z.size());
    nbPtZInv(p_z.size() - 1) = 1;
    for (int id = p_z.size() - 2; id >= 0 ; --id)
        nbPtZInv(id) = nbPtZInv(id + 1) * p_z[id + 1]->size();

    // calculate local sum
    ArrayXd localSum = localSumCalc(p_x,  p_z, p_y, nbPtZ, nbPtZT);

    // store index
    ArrayXi index = ArrayXi::Constant(p_z.size(), -1);
    // for return
    ArrayXd retCDF = ArrayXd::Zero(nbPtZT);
    // to store partial sum
    vector< shared_ptr< ArrayXd>  >  sumOfLocal(p_z.size() + 1);
    int isumOfLocal = nbPtZT;
    sumOfLocal[0] = make_shared< ArrayXd>(ArrayXd::Zero(isumOfLocal));
    for (size_t id  = 1; id <= p_z.size(); ++id)
    {
        isumOfLocal /= p_z[id - 1]->size();
        sumOfLocal[id] = make_shared< ArrayXd>(ArrayXd::Zero(isumOfLocal));
    }
    (*sumOfLocal[0]) = localSum;


    // previous index
    ArrayXi indexPrev(p_z.size());
    // nest on number of Z points
    for (int iloc = 0; iloc  < nbPtZT; ++iloc)
    {
        indexPrev = index;
        // index calculation for CDF
        int ipt = iloc;
        for (size_t id = 0; id  < p_z.size(); ++id)
        {
            index(id) = static_cast<int>(ipt / nbPtZInv(id));
            ipt -= index(id) * nbPtZInv(id);
        }
        for (size_t id = 0; id < p_z.size(); ++id)
        {
            if (index(id) != indexPrev(id))
            {
                // update sum of local
                for (int ipt = 0; ipt < sumOfLocal[id + 1]->size(); ++ipt)
                {
                    int iindexInSum = index(id) + ipt * p_z[id]->size();
                    // local index in sumOfLocal array
                    (*sumOfLocal[id + 1])(ipt) += (*sumOfLocal[id])(iindexInSum);
                }
                // update sum of local in other dimensions
                for (size_t idN = id + 1; idN <  p_z.size(); ++idN)
                {
                    // loop on points
                    for (int ipt = 0; ipt < sumOfLocal[idN + 1]->size(); ++ipt)
                    {
                        (*sumOfLocal[idN + 1])(ipt) = (*sumOfLocal[idN])(ipt * p_z[idN]->size());
                    }
                }
                break;
            }
        }
        // position in return array
        int illoc = index(0);
        for (size_t id = 1; id < p_z.size(); ++id)
            illoc += index(id) * nbPtZ(id);
        // store result
        retCDF(illoc) = (*sumOfLocal[p_z.size()])(0);
    }
    return retCDF;
}
}

