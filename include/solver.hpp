#pragma once

#include <matrix.hpp>
#include <iostream>
#include <cmath>
#include <mpi.h> 


class Solver
{
public: 
    Solver() = default;
    
    Solver(size_t M, size_t N, double (*boundary_condition)(double, double), int (*check_point_inside_region)(double, double), double x_min, double x_max, double y_min, double y_max);

    ~Solver();

    Matrix<double> solve(double precision, int& iteration_count);
    Matrix<double> solve_explicit(double precision, int& iteration_count);
    Matrix<double> solve_mpi(double precision, int& iteration_count);

private:
    double x_min;
    double h_x;
    size_t M;

    double y_min;
    double h_y;
    size_t N;

    int number_of_divisions_x;
    int number_of_divisions_y;
    void set_number_of_divisions(int number_of_processes);

    // double eps;

    Matrix<double> Q;
    // Matrix<double> F;
    Matrix<double> S;
    Matrix<double> B;
    Matrix<double> D;

    Matrix<double> operator_A(const Matrix<double>& iteration);

    double inline get_x(size_t i);
    double inline get_y(size_t j);

    int inline get_x_range_min(int proc_x_coord);
    int inline get_x_range_max(int proc_x_coord);
    int inline get_y_range_min(int proc_y_coord);
    int inline get_y_range_max(int proc_y_coord);
 
    // double (*density)(double, double) = nullptr;  // We have f = 1. Why bother?

    double (*boundary_condition)(double, double) = nullptr;
    int (*check_point_inside_region)(double, double) = nullptr;

    double scalar_product(const Matrix<double>& A, const Matrix<double>& B);
    double inline norm(const Matrix<double>& A);

    double dichotomy_solve_x(double x_start, double x_end, double y);
    double dichotomy_solve_y(double y_start, double y_end, double x);
    void init_S();
    void init_S_less_accurate();

    #ifdef DUMP_ERRS
    FILE* error_data_file;
    #endif

    #ifdef DUMP_DIFF_EVERY
    FILE* diff_data_file;
    #endif
};
