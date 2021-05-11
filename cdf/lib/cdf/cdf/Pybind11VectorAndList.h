// Copyright (C) 2019 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef PYBIND11VECTORANDLIST_H
#define PYBIND11VECTORANDLIST_H
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

/// \brief Defines converter from list to c++ and c++ to list
// Vector to List
template< typename T>
struct VecToList
{
    /// \brief Function
    /// \param p_vec vector to convert to PyObject
    /// \return a PyObject
    static PyObject *convert(const std::vector< T> &p_vec)
    {
        pybind11::list *l = new pybind11::list();
        for (size_t i = 0; i < p_vec.size(); i++)
        {
            (*l).append(p_vec[i]);
        }
        return l->ptr();
    }
};



template< typename T>
struct VecToListShPtr
{
    /// \brief Function
    /// \param p_vec vector of shared_ptr to convert to PyObject
    /// \return a PyObject
    static PyObject *convert(const std::vector< std::shared_ptr< T> > &p_vec)
    {
        pybind11::list *l = new pybind11::list();
        for (size_t i = 0; i < p_vec.size(); i++)
            (*l).append(*p_vec[i]);

        return l->ptr();
    }
};

template< typename T, typename TT >
struct VecToListShPtrTtoTT
{
    /// \brief Function
    /// \param p_vec vector of shared_ptr to convert to PyObject
    /// \return a PyObject
    static PyObject *convert(const std::vector< std::shared_ptr< T> > &p_vec)
    {
        pybind11::list *l = new pybind11::list();
        for (size_t i = 0; i < p_vec.size(); i++)
            (*l).append(* std::static_pointer_cast<TT>(p_vec[i]));

        return l->ptr();
    }
};


// list of objects to vector of shared_ptr
template< typename T>
std::vector< std::shared_ptr< T > > convertFromListShPtr(const pybind11::list &ns)
{
    std::vector< std::shared_ptr< T> > ret;
    ret.reserve(ns.size());
    for (auto item : ns)
    {
        T  local = item.cast<T>() ;
        ret.push_back(std::make_shared<T>(local));
    }
    return ret;
}


// converter list of objects to vector
template< typename T>
std::vector<T >  convertFromList(const pybind11::list &ns)
{
    std::vector< T > ret;
    ret.reserve(ns.size());
    for (auto item : ns)
    {
        T  local = item.cast<T>() ;
        ret.push_back(local);
    }
    return ret;
}

// same but send back a boost shared_ptr
template< typename T>
std::shared_ptr< std::vector<T > >  convertFromListToShared(const pybind11::list &ns)
{
    std::shared_ptr< std::vector<T > > ret = std::make_shared< std::vector<T> >() ;
    ret->reserve(ns.size());
    for (auto item : ns)
        ret->push_back(item.cast<T>());
    return ret;
}

#endif /* PYBIND11VECTORANDLIST_H */
