// Copyright (C) 2020 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <vector>
#include <memory>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>


using namespace std;
using namespace Eigen ;


//  To ease sorting of points in each dimension
double lesserPair(const pair<double, int > &c1, const pair<double, int> &c2)
{
    return c1.first < c2.first ;
}

namespace StOpt
{

///  p_pt           size (2,NnbSimul)
///  p_iSort1       first set of points of size (nbSimul,2)
///  p_iSort2       second set of points of size (nbSimul,2)
///  p_valToAdd     size (P, nbSimul)
///  p_fDominLoc    size (nbFunc, N)
///  merge the two set in first direction for the last dimension
void merge1D(const  ArrayXXd &p_pt,
             const  ArrayXXi &p_iSort1,
             const  ArrayXXi &p_iSort2,
             const ArrayXd   &p_valToAdd,
             ArrayXd   &p_fDominLoc)
{
    int i1 = p_iSort1.rows();
    int i2 =  p_iSort2.rows();
    int iloc = 0 ;
    double  sumToAdd = 0.;
    for (int i = 0; i < i2 ; ++i)
    {
        int ipt2 = p_iSort2(i, 0);
        int ipt1 = p_iSort1(iloc, 0);
        while (p_pt(0, ipt2) >=  p_pt(0, ipt1))
        {
            sumToAdd += p_valToAdd(ipt1);
            iloc += 1;
            if (iloc == i1)
                break;
            ipt1 =  p_iSort1(iloc, 0);
        }
        // add contribution
        p_fDominLoc(ipt2) += sumToAdd;
        // last points to treat
        if (iloc == i1)
        {
            for (int j = i + 1; j <  i2; ++j)
            {
                p_fDominLoc(p_iSort2(j, 0)) += sumToAdd;
            }
            break;
        }
    }
}




///  Merge nD procedure between two sets of point to calculate all summations
///  dimension above 2
///  Set A is dominated by set B in the current dimension
///  p_pt          size (d,nbSimul)
///  p_iSort1      first set of points A  size (nbSimul,d)
///  p_iSort2      second set of points B  size (nbSimul,d
///  p_idim        current dimension treated
///  p_valToAdd    arrays of size nbSimul
///  p_fDomin      arrays of size nbSimul
void mergeNDAlone(const  ArrayXXd &p_pt,
                  const  ArrayXXi &p_iSort1,
                  const  ArrayXXi &p_iSort2,
                  const int &p_idim,
                  const ArrayXd   &p_valToAdd,
                  ArrayXd    &p_fDomin)
{

    int i1 = p_iSort1.rows();
    int i2 =  p_iSort2.rows();
    // merge the two set to find the median point of the union
    int nbPoints = i1 + i2;
    int nbPtsDiv2 = nbPoints / 2;
    int iPos1 = 0; //position in array1
    int iPos2 = 0 ; // position in array 2
    int iPos = 0;
    int nDimMu = p_idim - 1;
    while ((iPos1 < i1) && (iPos2 < i2) && (iPos < nbPtsDiv2))
    {
        if (p_pt(nDimMu, p_iSort1(iPos1, nDimMu)) < p_pt(nDimMu, p_iSort2(iPos2, nDimMu)))
            iPos1++;
        else
            iPos2++;
        iPos++ ;
    }
    if (iPos1 == i1)
        iPos2 += nbPtsDiv2 - iPos;
    else if (iPos2 == i2)
        iPos1 += nbPtsDiv2 - iPos;
    double xMin = 0. ;
    if (iPos1 > 0)
    {
        if (iPos2 > 0)
            xMin = max(p_pt(nDimMu, p_iSort1(iPos1 - 1, nDimMu)), p_pt(nDimMu, p_iSort2(iPos2 - 1, nDimMu)));
        else
            xMin = p_pt(nDimMu, p_iSort1(iPos1 - 1, nDimMu));
    }
    else
        xMin = p_pt(nDimMu, p_iSort2(iPos2 - 1, nDimMu));

    double xMax = 0;
    if (iPos1 < i1)
    {
        if (iPos2 < i2)
            xMax = min(p_pt(nDimMu, p_iSort1(iPos1, nDimMu)), p_pt(nDimMu, p_iSort2(iPos2, nDimMu)));
        else
            xMax = p_pt(nDimMu, p_iSort1(iPos1, nDimMu));
    }
    else
        xMax =  p_pt(nDimMu, p_iSort2(iPos2, nDimMu));

    // xMed permist to seperate two sets with roughly the same number of particles
    double xMed = 0.5 * (xMin + xMax) ;

    // 4 sets
    ArrayXXi iSort11(iPos1, p_idim)  ; // set A below : A1
    iSort11.col(nDimMu) = p_iSort1.col(nDimMu).head(iPos1);
    ArrayXXi iSort21(iPos2, p_idim) ; // set B below: B1
    iSort21.col(nDimMu) = p_iSort2.col(nDimMu).head(iPos2);
    ArrayXXi iSort12(i1 - iPos1, p_idim); // set A above: A2
    iSort12.col(nDimMu) = p_iSort1.col(nDimMu).tail(i1 - iPos1);
    ArrayXXi iSort22(i2 - iPos2, p_idim) ; // set B above: B2
    iSort22.col(nDimMu) = p_iSort2.col(nDimMu).tail(i2 - iPos2);

    // now keep sorted point on dimension 0
    for (int id = 0 ; id < p_idim - 1; ++id)
    {
        int iloc11 = 0;
        int iloc12 = 0;
        int iloc21 = 0;
        int iloc22 = 0;
        // Set A
        for (int i = 0; i < i1; ++i)
        {
            int ipt = p_iSort1(i, id); // point number
            if ((p_pt(nDimMu, ipt) <= xMed) && (iloc11 < iPos1))
            {
                // under the median
                iSort11(iloc11++, id) = ipt;
            }
            else
            {
                iSort12(iloc12++, id) = ipt;
            }
        }
        // set B
        for (int i = 0; i < i2; ++i)
        {
            int ipt = p_iSort2(i, id); // point number
            if ((p_pt(nDimMu, ipt) <= xMed) && (iloc21 < iPos2))
            {
                // under the median
                iSort21(iloc21++, id) = ipt;
            }
            else
            {
                iSort22(iloc22++, id) = ipt;
            }
        }
    }
    // merge on the two set A1 and B1
    if ((iPos1 > 0) && (iPos2 > 0))
    {
        mergeNDAlone(p_pt, iSort11, iSort21, p_idim, p_valToAdd, p_fDomin);
    }

    // merge on teh two set A2 and B2
    if ((iPos1 < i1) && (iPos2 < i2))
    {
        mergeNDAlone(p_pt, iSort12, iSort22, p_idim, p_valToAdd, p_fDomin);
    }


    if (p_idim == 2)
    {
        // merge on inferior dimension : we know that point in iSort22 dominate iSort11 in dimension 2 and 3
        if ((iSort11.rows() > 0) && (iSort22.rows() > 0))
        {
            // merge 1D for the  direction  (x_j<x) (y_j<y)  (z_j< z) : 0
            merge1D(p_pt, iSort11, iSort22, p_valToAdd,  p_fDomin) ;
        }
    }
    else
    {
        /// merge in dimension below
        if ((iSort11.rows() > 0) && (iSort22.rows() > 0))
        {
            mergeNDAlone(p_pt, iSort11, iSort22, nDimMu, p_valToAdd, p_fDomin);

        }
    }
}



///  p_pt          size (d,nbSimul)
///  p_iSort      first set of points A  size (nbSimul,d)
///  p_valToAdd   vector of size  2^{d}  of arrays of size (P, nbSimul)
///  p_fDomin     vector of size  2^{d}  of arrays of size (P, nbSimul)
void recursiveCallNDAlone(const  ArrayXXd &p_pt,
                          const  ArrayXXi &p_iSort,
                          const ArrayXd    &p_valToAdd,
                          ArrayXd   &p_fDomin)
{
    if (p_iSort.cols() == 1)
    {
        for (int is = 1; is <  p_pt.cols(); ++is)
            p_fDomin(p_iSort(is, 0)) = p_fDomin(p_iSort(is - 1, 0)) + p_valToAdd(p_iSort(is - 1, 0)) ;
        for (int is = p_pt.cols() - 2; is >= 0; --is)
            p_fDomin(p_iSort(is, 0)) = p_fDomin(p_iSort(is + 1, 0)) + p_valToAdd(p_iSort(is + 1, 0)) ;
    }
    else if (p_iSort.rows() > 1)
    {
        // split into two part
        int iSize1 = p_iSort.rows() / 2 ;
        int iSize2 = p_iSort.rows() - iSize1  ;
        int nDimM1 = p_iSort.cols() - 1;
        // position valeu of splitting position
        double xMedium = 0.5 * (p_pt(nDimM1, p_iSort(iSize1 - 1, nDimM1)) + p_pt(nDimM1, p_iSort(iSize1, nDimM1)));
        // utilitary for sorted particles
        ArrayXXi iSort1(iSize1, p_iSort.cols());
        ArrayXXi iSort2(iSize2, p_iSort.cols());
        // copy last dimenson
        iSort1.col(nDimM1) = p_iSort.col(nDimM1).head(iSize1);
        iSort2.col(nDimM1) = p_iSort.col(nDimM1).tail(iSize2);
        // two  first dimensions
        for (int id = 0; id < nDimM1; ++id)
        {
            int iLoc1 = 0 ;
            int iLoc2 = 0 ;
            for (int i = 0 ; i < p_iSort.rows() ; ++i)
            {
                int iPoint = p_iSort(i, id) ; // get back point number
                // decide in which set to add the point
                if (p_pt(nDimM1, iPoint) < xMedium)
                    iSort1(iLoc1++, id) = iPoint;
                else
                    iSort2(iLoc2++, id) = iPoint;
            }
        }

        // call on the two set
        recursiveCallNDAlone(p_pt, iSort1, p_valToAdd,  p_fDomin);
        recursiveCallNDAlone(p_pt, iSort2, p_valToAdd,   p_fDomin);

        // merge nD for the 2 set
        if (p_iSort.cols() > 2)
            mergeNDAlone(p_pt, iSort1, iSort2, nDimM1,  p_valToAdd, p_fDomin);
        else
        {
            // 2D merge
            // merge 1D for direction 1
            merge1D(p_pt, iSort1, iSort2,   p_valToAdd,  p_fDomin) ;
        }

    }
}


/// \brief  Dominance  use in CDF
/// \param  p_pt        arrays of point coordinates  (d, nbSimul)
/// \param  p_valToAdd  terms to add  (exponentall in summation above) : vector of \f$2^d \f  kinds of terms  of size ( P, nbSimul)
/// \param  p_fDomin    result of summation  \f$2^d \f terms of size  ( P, nbSimul)
void nDDominanceAlone(const ArrayXXd &p_pt,
                      const ArrayXd    &p_valToAdd,
                      ArrayXd &p_fDomin)
{
    int nbSim = p_pt.cols();
    int nDim = p_pt.rows();
    // dimension 1
    ArrayXXi iSort(nbSim, nDim);
    for (int id = 0; id < nDim; ++id)
    {
        vector< std::pair< double, int> >   xSDim(nbSim);
        for (int i = 0; i < nbSim ; ++i)
        {
            xSDim[i] = make_pair(p_pt(id, i), i);
        }
        // sort
        sort(xSDim.begin(), xSDim.end(), lesserPair);
        for (int i = 0; i < nbSim ; ++i)
            iSort(i, id) = xSDim[i].second ;
    }
    p_fDomin.setConstant(0.);

    // recursive call with divide and conquer
    recursiveCallNDAlone(p_pt, iSort,  p_valToAdd,  p_fDomin);

}
}
