#include "solver.hpp"


Solver::Solver(size_t M, size_t N, double (*boundary_condition)(double, double), int (*check_point_inside_region)(double, double), double x_min, double x_max, double y_min, double y_max):
M(M), N(N), x_min(x_min), y_min(y_min), boundary_condition(boundary_condition), check_point_inside_region(check_point_inside_region)
{
    if ((boundary_condition == nullptr) || (check_point_inside_region == nullptr))
    {
        std::cout << "nullptr passed" << '\n';
        std::exit(1);
    }

    this->number_of_divisions_y = 1;
    this->number_of_divisions_x = 1;

    this->h_x = (x_max - x_min) / M;
    this->h_y = (y_max - y_min) / N;
    double eps = pow(std::max(this->h_x, this->h_y), 2);

    // this->init_S();
    this->init_S_less_accurate();

    this->Q = (Matrix<double>(M+1, N+1, this->h_x*this->h_y) - this->S) / (this->h_x*this->h_y * pow(eps, 2));
    // this->F = this->S / (this->h_x*this->h_y);
    // this->F = Matrix<double>(M+1, N+1, 1);  // F[i, j] == 1, why bother, again

    this->B = Matrix<double>(M+1, N+1);
    #pragma omp parallel for shared(B) collapse(2)
    for (size_t i = 1; i <= M - 1; i++)
    {
        for (size_t j = 1; j <= N - 1; j++)
        {
            // this->B[i, j] = F[i, j] + Q[i, j]*this->boundary_condition(this->get_x(i), this->get_y(j));
            this->B.set(i, j, 1 + Q.get(i, j)*this->boundary_condition(this->get_x(i), this->get_y(j)));
        }
    }

    this->D = this->Q + Matrix<double>(this->M+1, this->N+1, 2.0/pow(this->h_x, 2) + 2.0/pow(this->h_y, 2));

    #ifdef DUMP_ERRS
    const char* errs_filename = "./errors.txt";
    this->error_data_file = fopen(errs_filename, "w");
    #endif

    #ifdef DUMP_DIFF_EVERY
    const char* diif_filename = "./diffs.txt";
    this->diff_data_file = fopen(diif_filename, "w");
    std::fprintf(this->diff_data_file, "%zu %zu %f %f %f %f\n", this->M, this->N, x_min, x_max, y_min, y_max);
    #endif
}

Solver::~Solver()
{
    #ifdef DUMP_ERRS
    fclose(this->error_data_file);
    #endif

    #ifdef DUMP_DIFF_EVERY
    fclose(this->diff_data_file);
    #endif
}

Matrix<double> Solver::solve(double precision, int& iteration_count)
{
    double beta;
    Matrix<double> current_iteration(this->M+1, this->N+1);

    Matrix<double> next_r(this->M+1, this->N+1);
    auto current_r = this->B - this->operator_A(current_iteration);

    Matrix<double> next_z(this->M+1, this->N+1);
    auto current_z = current_r.hadamard_division(this->D);

    auto descent = current_z;

    auto step = this->scalar_product(current_z, current_r) / this->scalar_product(this->operator_A(descent), descent);

    auto next_iteration = current_iteration + step*descent;
    iteration_count = 1;

    while (this->norm(next_iteration - current_iteration) >= precision)
    {
        current_iteration = next_iteration;

        next_r = current_r - step*this->operator_A(descent);
        next_z = next_r.hadamard_division(this->D);
        beta = this->scalar_product(next_z, next_r) / this->scalar_product(current_z, current_r);

        current_r = next_r;
        current_z = next_z;

        descent = current_z + beta*descent;
        step = this->scalar_product(current_z, current_r) / this->scalar_product(this->operator_A(descent), descent);

        next_iteration = current_iteration + step*descent;
        iteration_count += 1;
    }

    return next_iteration;
}

Matrix<double> Solver::solve_explicit(double precision, int& iteration_count)
{
    double beta;
    Matrix<double> iteration(this->M+1, this->N+1);

    double error;

    auto r = this->B - this->operator_A(iteration);

    auto descent = r.hadamard_division(this->D);
    auto A_descent = this->operator_A(descent);

    double scalar_product_weight = this->h_x * this->h_y;
    double zr = this->scalar_product(descent, r) / scalar_product_weight;
    double zr_next;
    double descent_A_descent = this->scalar_product(A_descent, descent) / scalar_product_weight;
    double descent_sqr;

    double inv_h_x_sqr = 1.0 / pow(this->h_x, 2);
    double inv_h_y_sqr = 1.0 / pow(this->h_y, 2);

    auto step =  zr / descent_A_descent;

    int max_iterations = (this->N - 1) * (this->M - 1);
    // int max_iterations = 2;
    int iteration_number;
    for (iteration_number = 1; iteration_number <= max_iterations; iteration_number++)
    {
        zr_next = 0;

        #ifdef DUMP_DIFF_EVERY
        if (iteration_number % DUMP_DIFF_EVERY == 0)
        {
            for (size_t i = 0; i <= this->M; i++)
            {
                for (size_t j = 0; j <= this->N; j++)
                {
                    std::fprintf(this->diff_data_file, "%.12f ", step*descent.get(i, j));
                }
                std::fprintf(this->diff_data_file, "\n");
            }
            std::fprintf(this->diff_data_file, "\n");
        }
        #endif

        #pragma omp parallel for reduction(+ : zr_next)
        for (size_t i = 1; i <= this->M-1; i++)
        {
            for (size_t j = 1; j <= this->N-1; j++)
            {
                iteration.set(i, j, iteration.get(i, j) + step*descent.get(i, j));

                r.set(i, j, r.get(i, j) - step*A_descent.get(i, j));

                zr_next += r.get(i, j) * r.get(i, j) / this->D.get(i, j);
            }
        }

        beta = zr_next / zr;
        zr = zr_next;

        #pragma omp parallel for
        for (size_t i = 1; i <= this->M-1; i++)
        {
            for (size_t j = 1; j <= this->N-1; j++)
            {
                descent.set(i, j, r.get(i, j) / this->D.get(i, j) + beta * descent.get(i, j));
            }
        }
    
        descent_A_descent = 0;
        descent_sqr = 0;
        #pragma omp parallel for reduction(+ : descent_A_descent, descent_sqr)
        for (size_t i = 1; i <= this->M-1; i++)
        {
            for (size_t j = 1; j <= this->N-1; j++)
            {
                A_descent.set(i, j, -(descent.get(i+1, j) + descent.get(i-1, j)) * inv_h_x_sqr - (descent.get(i, j+1) + descent.get(i, j-1)) * inv_h_y_sqr + D.get(i, j)*descent.get(i, j));

                descent_A_descent += descent.get(i, j) * A_descent.get(i, j);
                descent_sqr += descent.get(i, j) * descent.get(i, j);
            }
        }

        step = zr / descent_A_descent;

        error = std::sqrt(step*step*descent_sqr*scalar_product_weight);

        #ifdef DUMP_ERRS
        std::fprintf(this->error_data_file, "%.12f\n", error);
        #endif

        if (error < precision)
        {
            #ifdef DUMP_DIFF_EVERY
            for (size_t i = 0; i <= this->M; i++)
            {
                for (size_t j = 0; j <= this->N; j++)
                {
                    std::fprintf(this->diff_data_file, "%.12f ", step*descent.get(i, j));
                }
                std::fprintf(this->diff_data_file, "\n");
            }
            std::fprintf(this->diff_data_file, "\n");
            #endif

            break;
        }
    }

    iteration_count = iteration_number;
    return iteration;
}

Matrix<double> Solver::solve_mpi(double precision, int &iteration_count)
{
    double all_error_sqr = 0;
    double inv_h_x_sqr = 1.0 / pow(this->h_x, 2);
    double inv_h_y_sqr = 1.0 / pow(this->h_y, 2);
    double scalar_product_weight = this->h_x * this->h_y;

    Matrix<double> result(this->M+1, this->N+1);

    double all_zr = 0;
    double all_descent_A_descent = 0;
    double all_zr_next = 0;

    int all_iterations = 0;

    int number_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    this->set_number_of_divisions(number_of_processes);

    MPI_Comm grid;
    int dims[2] = {this->number_of_divisions_x, this->number_of_divisions_y};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);

    int rank;
    int coordinates[2];

    MPI_Comm_rank(grid, &rank);
    MPI_Cart_coords(grid, rank, 2, coordinates);

    int neighbors[4];
    MPI_Cart_shift(grid, 1, 1, &neighbors[1], &neighbors[0]);
    MPI_Cart_shift(grid, 0, 1, &neighbors[2], &neighbors[3]);

    auto x_range_min = this->get_x_range_min(coordinates[0]);
    auto x_range_max = this->get_x_range_max(coordinates[0]);
    auto y_range_min = this->get_y_range_min(coordinates[1]);
    auto y_range_max = this->get_y_range_max(coordinates[1]);

    int local_x_dim = x_range_max - x_range_min + 1;
    int local_y_dim = y_range_max - y_range_min + 1;

    double beta;
    Matrix<double> local_iteration(local_x_dim, local_y_dim);

    double local_error_sqr;
    Matrix<double> r(local_x_dim, local_y_dim);
    Matrix<double> descent(local_x_dim, local_y_dim);
    Matrix<double> A_descent(local_x_dim, local_y_dim);

    double local_zr = 0;
    double local_zr_next = 0;
    double local_descent_A_descent = 0;
    double local_descent_sqr;

    for (int i = 0; i < local_x_dim; i++)
    {
        for (int j = 0; j < local_y_dim; j++)
        {
            r.set(i, j, this->B.get(i + x_range_min, j + y_range_min));
            descent.set(i, j, r.get(i, j) / this->D.get(i + x_range_min, j + y_range_min));
        
            local_zr += r.get(i, j) * descent.get(i, j);
        }
    }

    int buf_space = std::max(std::floor(this->M / this->number_of_divisions_x) + 1, std::floor(this->N / this->number_of_divisions_y) + 1);
    Matrix<double> send_buf_descent(4, buf_space);

    for (int i = 0; i < local_x_dim; i++)
    {
        send_buf_descent.set(1, i, descent.get(i, 0));
        send_buf_descent.set(0, i, descent.get(i, local_y_dim-1)); 
    }

    for (int j = 0; j < local_y_dim; j++)
    {
        send_buf_descent.set(2, j, descent.get(0, j));  
        send_buf_descent.set(3, j, descent.get(local_x_dim-1, j)); 
    }

    MPI_Request send_descent_requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    for (int i = 0; i < 4; i++)
    {
        MPI_Isend(send_buf_descent.get_ptr(i, 0), buf_space, MPI_DOUBLE, neighbors[i], 1, grid, &send_descent_requests[i]);
    }

    for (int i = 1; i < local_x_dim - 1; i++)
    {
        for (int j = 1; j < local_y_dim - 1; j++)
        {
            A_descent.set(i, j, -(descent.get(i+1, j) + descent.get(i-1, j)) * inv_h_x_sqr - (descent.get(i, j+1) + descent.get(i, j-1)) * inv_h_y_sqr + D.get(i + x_range_min, j + y_range_min)*descent.get(i, j));
            local_descent_A_descent += descent.get(i, j) * A_descent.get(i, j);
        }
    }

    Matrix<double> neighbors_descent(4, buf_space);
    MPI_Request recv_descent_requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    for (int i = 0; i < 4; i++)
    {
        MPI_Irecv(neighbors_descent.get_ptr(i, 0), buf_space, MPI_DOUBLE, neighbors[i], 1, grid, &recv_descent_requests[i]);
    }

    MPI_Waitall(4, recv_descent_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(4, send_descent_requests, MPI_STATUSES_IGNORE);

    for (int i = 1; i < local_x_dim-1; i++)
    {
        A_descent.set(i, 0, -(descent.get(i+1, 0) + descent.get(i-1, 0)) * inv_h_x_sqr - (descent.get(i, 1) + neighbors_descent.get(1, i)) * inv_h_y_sqr + D.get(i + x_range_min, y_range_min)*descent.get(i, 0));
        A_descent.set(i, local_y_dim-1, -(descent.get(i+1, local_y_dim-1) + descent.get(i-1, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, i) + descent.get(i, local_y_dim-1-1)) * inv_h_y_sqr + D.get(i + x_range_min, local_y_dim-1 + y_range_min)*descent.get(i, local_y_dim-1));
        local_descent_A_descent += descent.get(i, 0) * A_descent.get(i, 0);
        local_descent_A_descent += descent.get(i, local_y_dim-1) * A_descent.get(i, local_y_dim-1);
    }

    for (int j = 1; j < local_y_dim-1; j++)
    {
        A_descent.set(0, j, -(descent.get(1, j) + neighbors_descent.get(2, j)) * inv_h_x_sqr - (descent.get(0, j+1) + descent.get(0, j-1)) * inv_h_y_sqr + D.get(x_range_min, j + y_range_min)*descent.get(0, j));
        A_descent.set(local_x_dim-1, j, -(neighbors_descent.get(3, j) + descent.get(local_x_dim-1-1, j)) * inv_h_x_sqr - (descent.get(local_x_dim-1, j+1) + descent.get(local_x_dim-1, j-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, j + y_range_min)*descent.get(local_x_dim-1, j));
        local_descent_A_descent += descent.get(0, j) * A_descent.get(0, j);
        local_descent_A_descent += descent.get(local_x_dim-1, j) * A_descent.get(local_x_dim-1, j);
    }

    A_descent.set(0, 0, -(descent.get(1, 0) + neighbors_descent.get(2, 0)) * inv_h_x_sqr - (descent.get(0, 1) + neighbors_descent.get(1, 0)) * inv_h_y_sqr + D.get(x_range_min, y_range_min)*descent.get(0, 0));
    A_descent.set(local_x_dim-1, 0, -(neighbors_descent.get(3, 0) + descent.get(local_x_dim-1-1, 0)) * inv_h_x_sqr - (descent.get(local_x_dim-1, 1) + neighbors_descent.get(1, local_x_dim-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, y_range_min)*descent.get(local_x_dim-1, 0));
    A_descent.set(local_x_dim-1, local_y_dim-1, -(neighbors_descent.get(3, local_y_dim-1) + descent.get(local_x_dim-1-1, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, local_x_dim-1) + descent.get(local_x_dim-1, local_y_dim-1-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, local_y_dim-1 + y_range_min)*descent.get(local_x_dim-1, local_y_dim-1));
    A_descent.set(0, local_y_dim-1, -(descent.get(1, local_y_dim-1) + neighbors_descent.get(2, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, 0) + descent.get(0, local_y_dim-1-1)) * inv_h_y_sqr + D.get(x_range_min, local_y_dim-1 + y_range_min)*descent.get(0, local_y_dim-1));    

    local_descent_A_descent += descent.get(0, 0) * A_descent.get(0, 0);
    local_descent_A_descent += descent.get(local_x_dim-1, 0) * A_descent.get(local_x_dim-1, 0);
    local_descent_A_descent += descent.get(local_x_dim-1, local_y_dim-1) * A_descent.get(local_x_dim-1, local_y_dim-1);
    local_descent_A_descent += descent.get(0, local_y_dim-1) * A_descent.get(0, local_y_dim-1);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allreduce(&local_zr, &all_zr, 1, MPI_DOUBLE, MPI_SUM, grid);
    MPI_Allreduce(&local_descent_A_descent, &all_descent_A_descent, 1, MPI_DOUBLE, MPI_SUM, grid);

    double step = all_zr / all_descent_A_descent;

    int max_iterations = (this->N - 1) * (this->M - 1);
    // int max_iterations = 2;
    int iteration_number;
    for (iteration_number = 1; iteration_number <= max_iterations; iteration_number++)
    {
        local_zr_next = 0;
        for (int i = 0; i < local_x_dim; i++)
        {
            for (int j = 0; j < local_y_dim; j++)
            {
                local_iteration.set(i, j, local_iteration.get(i, j) + step*descent.get(i, j));

                r.set(i, j, r.get(i, j) - step*A_descent.get(i, j));

                local_zr_next += r.get(i, j) * r.get(i, j) / this->D.get(i + x_range_min, j + y_range_min);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&local_zr_next, &all_zr_next, 1, MPI_DOUBLE, MPI_SUM, grid);

        beta = all_zr_next / all_zr;
        local_zr = local_zr_next;

        local_descent_sqr = 0;
        for (size_t i = 0; i < local_x_dim; i++)
        {
            for (size_t j = 0; j < local_y_dim; j++)
            {
                descent.set(i, j, r.get(i, j) / this->D.get(i + x_range_min, j + y_range_min) + beta * descent.get(i, j));
                local_descent_sqr += descent.get(i, j) * descent.get(i, j);
            }
        }
    
        local_descent_A_descent = 0;
        
        for (int i = 0; i < local_x_dim; i++)
        {
            send_buf_descent.set(1, i, descent.get(i, 0));
            send_buf_descent.set(0, i, descent.get(i, local_y_dim-1));
        }

        for (int j = 0; j < local_y_dim; j++)
        {
            send_buf_descent.set(2, j, descent.get(0, j));
            send_buf_descent.set(3, j, descent.get(local_x_dim-1, j));
        }

        for (int i = 0; i < 4; i++)
        {
            MPI_Isend(send_buf_descent.get_ptr(i, 0), buf_space, MPI_DOUBLE, neighbors[i], 1, grid, &send_descent_requests[i]);
        }

        for (int i = 1; i < local_x_dim - 1; i++)
        {
            for (int j = 1; j < local_y_dim - 1; j++)
            {
                A_descent.set(i, j, -(descent.get(i+1, j) + descent.get(i-1, j)) * inv_h_x_sqr - (descent.get(i, j+1) + descent.get(i, j-1)) * inv_h_y_sqr + D.get(i + x_range_min, j + y_range_min)*descent.get(i, j));
                local_descent_A_descent += descent.get(i, j) * A_descent.get(i, j);
            }
        }

        for (int i = 0; i < 4; i++)
        {
            MPI_Irecv(neighbors_descent.get_ptr(i, 0), buf_space, MPI_DOUBLE, neighbors[i], 1, grid, &recv_descent_requests[i]);
        }

        MPI_Waitall(4, send_descent_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(4, recv_descent_requests, MPI_STATUSES_IGNORE);

        for (int i = 1; i < local_x_dim-1; i++)
        {
            A_descent.set(i, 0, -(descent.get(i+1, 0) + descent.get(i-1, 0)) * inv_h_x_sqr - (descent.get(i, 1) + neighbors_descent.get(1, i)) * inv_h_y_sqr + D.get(i + x_range_min, y_range_min)*descent.get(i, 0));
            A_descent.set(i, local_y_dim-1, -(descent.get(i+1, local_y_dim-1) + descent.get(i-1, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, i) + descent.get(i, local_y_dim-1-1)) * inv_h_y_sqr + D.get(i + x_range_min, local_y_dim-1 + y_range_min)*descent.get(i, local_y_dim-1));
            local_descent_A_descent += descent.get(i, 0) * A_descent.get(i, 0);
            local_descent_A_descent += descent.get(i, local_y_dim-1) * A_descent.get(i, local_y_dim-1);
        }

        for (int j = 1; j < local_y_dim-1; j++)
        {
            A_descent.set(0, j, -(descent.get(1, j) + neighbors_descent.get(2, j)) * inv_h_x_sqr - (descent.get(0, j+1) + descent.get(0, j-1)) * inv_h_y_sqr + D.get(x_range_min, j + y_range_min)*descent.get(0, j));
            A_descent.set(local_x_dim-1, j, -(neighbors_descent.get(3, j) + descent.get(local_x_dim-1-1, j)) * inv_h_x_sqr - (descent.get(local_x_dim-1, j+1) + descent.get(local_x_dim-1, j-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, j + y_range_min)*descent.get(local_x_dim-1, j));
            local_descent_A_descent += descent.get(0, j) * A_descent.get(0, j);
            local_descent_A_descent += descent.get(local_x_dim-1, j) * A_descent.get(local_x_dim-1, j);
        }

        A_descent.set(0, 0, -(descent.get(1, 0) + neighbors_descent.get(2, 0)) * inv_h_x_sqr - (descent.get(0, 1) + neighbors_descent.get(1, 0)) * inv_h_y_sqr + D.get(x_range_min, y_range_min)*descent.get(0, 0));
        A_descent.set(local_x_dim-1, 0, -(neighbors_descent.get(3, 0) + descent.get(local_x_dim-1-1, 0)) * inv_h_x_sqr - (descent.get(local_x_dim-1, 1) + neighbors_descent.get(1, local_x_dim-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, y_range_min)*descent.get(local_x_dim-1, 0));
        A_descent.set(local_x_dim-1, local_y_dim-1, -(neighbors_descent.get(3, local_y_dim-1) + descent.get(local_x_dim-1-1, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, local_x_dim-1) + descent.get(local_x_dim-1, local_y_dim-1-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, local_y_dim-1 + y_range_min)*descent.get(local_x_dim-1, local_y_dim-1));
        A_descent.set(0, local_y_dim-1, -(descent.get(1, local_y_dim-1) + neighbors_descent.get(2, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, 0) + descent.get(0, local_y_dim-1-1)) * inv_h_y_sqr + D.get(x_range_min, local_y_dim-1 + y_range_min)*descent.get(0, local_y_dim-1));    

        local_descent_A_descent += descent.get(0, 0) * A_descent.get(0, 0);
        local_descent_A_descent += descent.get(local_x_dim-1, 0) * A_descent.get(local_x_dim-1, 0);
        local_descent_A_descent += descent.get(local_x_dim-1, local_y_dim-1) * A_descent.get(local_x_dim-1, local_y_dim-1);
        local_descent_A_descent += descent.get(0, local_y_dim-1) * A_descent.get(0, local_y_dim-1);


        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&local_zr, &all_zr, 1, MPI_DOUBLE, MPI_SUM, grid);
        MPI_Allreduce(&local_descent_A_descent, &all_descent_A_descent, 1, MPI_DOUBLE, MPI_SUM, grid);

        step = all_zr / all_descent_A_descent;

        local_error_sqr = step*step*local_descent_sqr*scalar_product_weight;

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&local_error_sqr, &all_error_sqr, 1, MPI_DOUBLE, MPI_SUM, grid);

        #ifdef DUMP_ERRS
        if (rank == 0)
        {
            std::fprintf(this->error_data_file, "%.12f\n", std::sqrt(all_error_sqr));
        }
        #endif

        if (std::sqrt(all_error_sqr) < precision)
        {
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&iteration_number, &all_iterations, 1, MPI_INT, MPI_MAX, grid);

    if (rank != 0)
    {
        MPI_Send(local_iteration.get_ptr(0, 0), local_x_dim*local_y_dim, MPI_DOUBLE, 0, 3, grid);
    }

    if (rank == 0)
    {
        iteration_count = all_iterations;

        for (int i = 0; i < local_x_dim; i++)
        {
            for (int j = 0; j < local_y_dim; j++)
            {
                result.set(i + x_range_min, j + y_range_min, local_iteration.get(i, j));
            }
        }

        for (int other_rank = 1; other_rank < number_of_processes; other_rank++)
        {
            int other_coords[2];
            MPI_Cart_coords(grid, other_rank, 2, other_coords); 
            int other_x_min = this->get_x_range_min(other_coords[0]); 
            int other_x_max = this->get_x_range_max(other_coords[0]); 
            int other_y_min = this->get_y_range_min(other_coords[1]); 
            int other_y_max = this->get_y_range_max(other_coords[1]); 
            int other_x_dim = other_x_max - other_x_min + 1;
            int other_y_dim = other_y_max - other_y_min + 1;

            std::vector<double> other_iteration(other_x_dim*other_y_dim);
            MPI_Recv(&other_iteration[0], other_x_dim*other_y_dim, MPI_DOUBLE, other_rank, 3, grid, MPI_STATUS_IGNORE);

            for (int i = 0; i < other_x_dim; i++)
            {
                for (int j = 0; j < other_y_dim; j++)
                {
                    result.set(i + other_x_min, j + other_y_min, other_iteration[i*other_y_dim + j]);
                }
            }
        }
    }

    return result;
}

void Solver::set_number_of_divisions(int number_of_processes)
{
    // We assume number of processes is 2^n

    int rest = number_of_processes / 2;
    while (rest > 0)
    {
        this->number_of_divisions_x *= 2;
        rest /= 2;
        if (rest <= 0)
        {
            break;
        }
        this->number_of_divisions_y *= 2;
        rest /= 2;
    }
}

Matrix<double> Solver::operator_A(const Matrix<double> &iteration)
{
    auto result = Matrix<double>(iteration.get_rows(), iteration.get_columns());
    #pragma omp parallel for shared(result)
    for (size_t i = 1; i < result.get_rows()-1; i++)
    {
        for (size_t j = 1; j < result.get_columns()-1; j++)
        {
            // result[i, j] = -(iteration[i+1, j] - 2*iteration[i, j] + iteration[i-1, j]) / pow(this->h_x, 2) - (iteration[i, j+1] - 2*iteration[i, j] + iteration[i, j-1]) / pow(this->h_y, 2) + Q[i, j]*iteration[i, j];
            result.set(i, j, -(iteration.get(i+1, j) - 2*iteration.get(i, j) + iteration.get(i-1, j)) / pow(this->h_x, 2) - (iteration.get(i, j+1) - 2*iteration.get(i, j) + iteration.get(i, j-1)) / pow(this->h_y, 2) + Q.get(i, j)*iteration.get(i, j));
        }
    }

    return result;
}

double Solver::get_x(size_t i)
{
    return this->x_min + i*this->h_x;
}

double Solver::get_y(size_t j)
{
    return this->y_min + j*this->h_y;
}

inline int Solver::get_x_range_min(int proc_x_coord)
{
    return floor(proc_x_coord * (((float) (this->M-1)) / this->number_of_divisions_x))+1;
}

inline int Solver::get_x_range_max(int proc_x_coord)
{
    return this->get_x_range_min(proc_x_coord+1)-1;
}

inline int Solver::get_y_range_min(int proc_y_coord)
{
    return floor(proc_y_coord * (((float) (this->N-1)) / this->number_of_divisions_y))+1;
}

inline int Solver::get_y_range_max(int proc_y_coord)
{
    return this->get_y_range_min(proc_y_coord+1)-1;
}

double Solver::scalar_product(const Matrix<double>& A, const Matrix<double>& B)
{
    double result = 0;
    #pragma omp parallel for reduction(+ : result)
    for (size_t i = 1; i <= this->M - 1; i++)
    {
        for (size_t j = 1; j <= this->N - 1; j++)
        {
            result += this->h_x * this->h_y * A.get(i, j) * B.get(i, j);
        }
    }

    return result;
}

double Solver::norm(const Matrix<double>& A)
{
    return sqrt(this->scalar_product(A, A));
}

double Solver::dichotomy_solve_x(double x_start, double x_end, double y)
{
    double x_left = x_start;
    double x_right = x_end;
    
    int f_left = this->boundary_condition(x_start, y);
    if (f_left == 0) {return x_left;}

    int f_right = this->boundary_condition(x_end, y);
    if (f_right == 0) {return x_right;}

    double x_centre = (x_start + x_end) / 2.0;
    int f_centre;
    while (x_right - x_left >= 1e-6)
    {
        x_centre = (x_left + x_right) / 2.0;
        f_centre = this->boundary_condition(x_centre, y);
        if (f_centre == 0) return x_centre;

        if (f_left * f_centre < 0)
        {
            f_right = f_centre;
            x_right = x_centre;
        }

        if (f_centre * f_right < 0)
        {
            f_left = f_centre;
            x_left = x_centre;
        }
    }

    return x_centre;
}

double Solver::dichotomy_solve_y(double y_start, double y_end, double x)
{
    double y_left = y_start;
    double y_right = y_end;
    
    int f_left = this->boundary_condition(x, y_start);
    if (f_left == 0) {return y_left;}

    int f_right = this->boundary_condition(x, y_end);
    if (f_right == 0) {return y_right;}

    double y_centre = (y_start + y_end) / 2.0;
    int f_centre;
    while (y_right - y_left >= 1e-6)
    {
        y_centre = (y_left + y_right) / 2.0;
        f_centre = this->boundary_condition(x, y_centre);
        if (f_centre == 0) return y_centre;

        if (f_left * f_centre < 0)
        {
            f_right = f_centre;
            y_right = y_centre;
        }

        if (f_centre * f_right < 0)
        {
            f_left = f_centre;
            y_left = y_centre;
        }
    }

    return y_centre;
}

void Solver::init_S()
{
// This function checks for all 14 cases of (P_ij intersects D) 
    // double x_left, x_right, y_down, y_up, zero1, zero2, s_aux;
    // auto inside_region = Matrix<int>(2, 2);
    this->S = Matrix<double>(this->M+1, this->N+1);
    #pragma omp parallel for shared(S)
    for (size_t i = 1; i <= this->M - 1; i++)
    {
        for (size_t j = 1; j <= this->N - 1; j++)
        {
            double x_left, x_right, y_down, y_up, zero1, zero2, s_aux;
            auto inside_region = Matrix<int>(2, 2);

            x_left = this->get_x(i) - 0.5*this->h_x;
            x_right = this->get_x(i) + 0.5*this->h_x;

            y_down = this->get_y(j) - 0.5*this->h_y;
            y_up = this->get_y(j) + 0.5*this->h_y;

            inside_region.set(0, 0, this->check_point_inside_region(x_left, y_down));
            inside_region.set(0, 1, this->check_point_inside_region(x_left, y_up));
            inside_region.set(1, 1, this->check_point_inside_region(x_right, y_up));
            inside_region.set(1, 0, this->check_point_inside_region(x_right, y_down));

            if ((inside_region.get(0, 0) >= 0) && (inside_region.get(0, 1) >= 0) && (inside_region.get(1, 1) >= 0) && (inside_region.get(1, 0) >= 0))
            {
                this->S.set(i, j, this->h_x*this->h_y);
                continue;
            }

            if ((inside_region.get(0, 0) <= 0) && (inside_region.get(0, 1) <= 0) && (inside_region.get(1, 1) <= 0) && (inside_region.get(1, 0) <= 0))
            {
                this->S.set(i, j, 0);
                continue;
            }

            for (int i1 = 0; i1 <= 1; i1++)
            {
                for (int j1 = 0; j1 <= 1; j1++)
                {
                    inside_region.set(i1, j1, 1 - abs(inside_region.get(i1, j1))); // Turn all 0 into 1, dichotomy will deal with it automatically
                }
            }

            if ((inside_region.get(0, 0)*inside_region.get(1, 0) == -1) && (inside_region.get(0, 1)*inside_region.get(1, 1) == -1))
            {
                zero1 = this->dichotomy_solve_x(x_left, x_right, y_down);
                zero2 = this->dichotomy_solve_x(x_left, x_right, y_up);

                s_aux = this->h_y * (zero1 - x_left + zero2 - x_left)/2.0;

                if (inside_region.get(0, 0) == 1)
                {
                    this->S.set(i, j, s_aux);
                    continue;
                }

                this->S.set(i, j, this->h_x*this->h_y - s_aux);
                continue;
            }

            if ((inside_region.get(0, 0)*inside_region.get(0, 1) == -1) && (inside_region.get(1, 0)*inside_region.get(1, 1) == -1))
            {
                zero1 = this->dichotomy_solve_y(y_down, y_up, x_left);
                zero2 = this->dichotomy_solve_y(y_down, y_up, x_right);

                s_aux = this->h_x * (zero1 - y_down + zero2 - y_down)/2.0;

                if (inside_region.get(0, 0) == 1)
                {
                    this->S.set(i, j, s_aux);
                    continue;
                }

                this->S.set(i, j, this->h_x*this->h_y - s_aux);
                continue;
            }

            if ((inside_region.get(0, 0)*inside_region.get(0, 1) == -1) && (inside_region.get(0, 0)*inside_region.get(1, 0) == -1))
            {
                zero1 = this->dichotomy_solve_x(x_left, x_right, y_down);
                zero2 = this->dichotomy_solve_y(y_down, y_up, x_left);

                s_aux = 0.5 * (zero1 - x_left) * (zero2 - y_down);

                if (inside_region.get(0, 0) == 1)
                {
                    this->S.set(i, j, s_aux);
                    continue;
                }

                this->S.set(i, j, this->h_x*this->h_y - s_aux);
                continue;
            }

            if ((inside_region.get(0, 0)*inside_region.get(0, 1) == -1) && (inside_region.get(0, 1)*inside_region.get(1, 1) == -1))
            {
                zero1 = this->dichotomy_solve_x(x_left, x_right, y_up);
                zero2 = this->dichotomy_solve_y(y_down, y_up, x_left);

                s_aux = 0.5 * (zero1 - x_left) * (y_up - zero2);

                if (inside_region.get(0, 1) == 1)
                {
                    this->S.set(i, j, s_aux);
                    continue;
                }

                this->S.set(i, j, this->h_x*this->h_y - s_aux);
                continue;
            }

            if ((inside_region.get(1, 0)*inside_region.get(1, 1) == -1) && (inside_region.get(0, 1)*inside_region.get(1, 1) == -1))
            {
                zero1 = this->dichotomy_solve_x(x_left, x_right, y_up);
                zero2 = this->dichotomy_solve_y(y_down, y_up, x_right);

                s_aux = 0.5 * (x_right - zero1) * (y_up - zero2);

                if (inside_region.get(1, 1) == 1)
                {
                    this->S.set(i, j, s_aux);
                    continue;
                }

                this->S.set(i, j, this->h_x*this->h_y - s_aux);
                continue;
            }

            if ((inside_region.get(1, 0)*inside_region.get(1, 1) == -1) && (inside_region.get(1, 0)*inside_region.get(0, 0) == -1))
            {
                zero1 = this->dichotomy_solve_x(x_left, x_right, y_down);
                zero2 = this->dichotomy_solve_y(y_down, y_up, x_right);

                s_aux = 0.5 * (x_right - zero1) * (zero2 - y_down);

                if (inside_region.get(1, 0) == 1)
                {
                    this->S.set(i, j, s_aux);
                    continue;
                }

                this->S.set(i, j, this->h_x*this->h_y - s_aux);
                continue;
            }
        }
    }
}

void Solver::init_S_less_accurate()
{
    this->S = Matrix<double>(this->M+1, this->N+1);
    double area = this->h_x*this->h_y;
    #pragma omp parallel for shared(S, area) collapse(2)
    for (size_t i = 1; i <= this->M - 1; i++)
    {
        for (size_t j = 1; j <= this->N - 1; j++)
        {
            double x_left, x_right, y_down, y_up;
            int sum = 0;

            x_left = this->get_x(i) - 0.5*this->h_x;
            x_right = this->get_x(i) + 0.5*this->h_x;

            y_down = this->get_y(j) - 0.5*this->h_y;
            y_up = this->get_y(j) + 0.5*this->h_y;

            auto in_ld = this->check_point_inside_region(x_left, y_down);
            auto in_lu = this->check_point_inside_region(x_left, y_up);
            auto in_ru = this->check_point_inside_region(x_right, y_up);
            auto in_rd = this->check_point_inside_region(x_right, y_down);

            sum += in_ld * (1.0 + in_ld);  // 1->1, 0->0, -1->0
            sum += in_rd * (1.0 + in_rd);
            sum += in_lu * (1.0 + in_lu);
            sum += in_ru * (1.0 + in_ru);

            this->S.set(i, j, (area/8.0)*sum);
        }
    }
}
