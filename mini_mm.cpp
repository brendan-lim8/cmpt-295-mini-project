#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

inline size_t idx(size_t i, size_t j, size_t N) {
    return i * N + j;
}

void fill_matrix(std::vector<double> &M, size_t N, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N * N; ++i) {
        M[i] = dist(rng);
    }
}

void matmul_ijk(const std::vector<double> &A,
                const std::vector<double> &B,
                std::vector<double> &C,
                size_t N)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[idx(i, k, N)] * B[idx(k, j, N)];
            }
            C[idx(i, j, N)] = sum;
        }
    }
}

void matmul_ikj(const std::vector<double> &A,
                const std::vector<double> &B,
                std::vector<double> &C,
                size_t N)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < N; ++k) {
            double a_ik = A[idx(i, k, N)];
            const double *B_row = &B[idx(k, 0, N)];
            double *C_row = &C[idx(i, 0, N)];
            for (size_t j = 0; j < N; ++j) {
                C_row[j] += a_ik * B_row[j];
            }
        }
    }
}

void matmul_blocked(const std::vector<double> &A,
                    const std::vector<double> &B,
                    std::vector<double> &C,
                    size_t N,
                    size_t BS)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (size_t ii = 0; ii < N; ii += BS) {
        for (size_t kk = 0; kk < N; kk += BS) {
            for (size_t jj = 0; jj < N; jj += BS) {
                for (size_t i = ii; i < ii + BS; ++i) {
                    for (size_t k = kk; k < kk + BS; ++k) {
                        double a_ik = A[idx(i, k, N)];
                        const double *B_row = &B[idx(k, jj, N)];
                        double *C_row = &C[idx(i, jj, N)];
                        for (size_t j = jj; j < jj + BS; ++j) {
                            C_row[j - jj] += a_ik * B_row[j - jj];
                        }
                    }
                }
            }
        }
    }
}

double max_abs_diff(const std::vector<double> &X,
                    const std::vector<double> &Y)
{
    double max_d = 0.0;
    size_t n = X.size();
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs(X[i] - Y[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

template <typename Func>
double time_matmul(Func f,
                   const std::vector<double> &A,
                   const std::vector<double> &B,
                   std::vector<double> &C,
                   size_t N,
                   size_t reps)
{
    double best_sec = 1e100;
    for (size_t r = 0; r < reps; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        f(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        best_sec = std::min(best_sec, diff.count());
    }
    return best_sec;
}

template <typename Func>
double time_matmul_blocked(Func f,
                           const std::vector<double> &A,
                           const std::vector<double> &B,
                           std::vector<double> &C,
                           size_t N,
                           size_t BS,
                           size_t reps)
{
    double best_sec = 1e100;
    for (size_t r = 0; r < reps; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        f(A, B, C, N, BS);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        best_sec = std::min(best_sec, diff.count());
    }
    return best_sec;
}

int main(int argc, char **argv) {
    size_t N = 512;      
    size_t BS = 32;      
    size_t reps = 3;    

    if (argc >= 2) {
        N = std::stoul(argv[1]);
    }
    if (argc >= 3) {
        BS = std::stoul(argv[2]);
    }

    std::cout << "Matrix size N = " << N << " (N x N)\n";
    std::cout << "Block size (for blocked version) = " << BS << "\n";
    std::cout << "Repetitions per variant = " << reps << "\n\n";

    std::vector<double> A(N * N), B(N * N);
    std::vector<double> C_ijk(N * N), C_ikj(N * N), C_blk(N * N);

    fill_matrix(A, N, 12345);
    fill_matrix(B, N, 67890);

    const size_t N_test = 64;
    std::vector<double> A_t(N_test * N_test), B_t(N_test * N_test);
    std::vector<double> C_ijk_t(N_test * N_test),
                        C_ikj_t(N_test * N_test),
                        C_blk_t(N_test * N_test);

    fill_matrix(A_t, N_test, 111);
    fill_matrix(B_t, N_test, 222);

    matmul_ijk(A_t, B_t, C_ijk_t, N_test);
    matmul_ikj(A_t, B_t, C_ikj_t, N_test);
    matmul_blocked(A_t, B_t, C_blk_t, N_test, std::min(BS, N_test));

    double diff_ijk_ikj = max_abs_diff(C_ijk_t, C_ikj_t);
    double diff_ijk_blk = max_abs_diff(C_ijk_t, C_blk_t);

    std::cout << "Correctness check (N_test = " << N_test << "):\n";
    std::cout << "  max |C_ijk - C_ikj| = " << diff_ijk_ikj << "\n";
    std::cout << "  max |C_ijk - C_blk| = " << diff_ijk_blk << "\n\n";


    auto t0_start = std::chrono::high_resolution_clock::now();
    matmul_ijk(A, B, C_ijk, N);
    auto t0_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t0_diff = t0_end - t0_start;
    std::cout << "[Reference] matmul_ijk single run: " << t0_diff.count() << " s\n\n";

    double best_ijk = time_matmul(matmul_ijk, A, B, C_ijk, N, reps);
    std::cout << "matmul_ijk (i-j-k order):\n";
    std::cout << "  best time over " << reps << " runs = " << best_ijk << " s\n";

    double best_ikj = time_matmul(matmul_ikj, A, B, C_ikj, N, reps);
    double diff_ijk_ikj_main = max_abs_diff(C_ijk, C_ikj);
    std::cout << "  max |C_ref - C_this| = " << 0 << "\n\n";

    std::cout << "matmul_ikj (i-k-j order):\n";
    std::cout << "  best time over " << reps << " runs = " << best_ikj << " s\n";
    std::cout << "  max |C_ref - C_this| = " << diff_ijk_ikj_main << "\n\n";

    double best_blk = time_matmul_blocked(matmul_blocked, A, B, C_blk, N, BS, reps);
    double diff_ijk_blk_main = max_abs_diff(C_ijk, C_blk);
    std::cout << "matmul_blocked:\n";
    std::cout << "  best time over " << reps << " runs = " << best_blk << " s\n";
    std::cout << "  max |C_ref - C_this| = " << diff_ijk_blk_main << "\n\n";

    return 0;
}

