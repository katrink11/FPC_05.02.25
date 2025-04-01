#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <chrono>

#define TIMER_START(id) auto start_##id = std::chrono::high_resolution_clock::now();
#define TIMER_STOP(id) auto end_##id = std::chrono::high_resolution_clock::now();
#define TIMER_RESULT(id) std::chrono::duration_cast<std::chrono::microseconds>(end_##id - start_##id).count()

typedef std::vector<std::vector<int>> matrix_t;

class Matrix
{
public:
    Matrix() : dim(0), ref(nullptr), row_offset(0), col_offset(0) {}

    Matrix(int size) : Matrix()
    {
        dim = size;
        data.resize(dim, std::vector<int>(dim));
    }

    Matrix(Matrix &other) : Matrix(other.size())
    {
        for (int i = 0; i < size(); i++)
            for (int j = 0; j < size(); j++)
                data[i][j] = other[i][j];
    }

    Matrix(const Matrix *parent, int row, int col) : dim(parent->size() - 1), ref(parent), row_offset(row), col_offset(col) {}

    virtual ~Matrix() {}

    bool valid_position(int r, int c) const { return r >= 0 && c >= 0 && r < dim && c < dim; }

    int get(int r, int c) const
    {
        if (!valid_position(r, c))
            throw std::out_of_range("Invalid position");
        if (!ref)
            return data[r][c];
        if (r >= row_offset)
            r++;
        if (c >= col_offset)
            c++;
        return ref->get(r, c);
    }

    void set(int r, int c, int val)
    {
        if (!valid_position(r, c))
            throw std::out_of_range("Invalid position");
        if (!ref)
            data[r][c] = val;
        else
            throw std::runtime_error("Read-only matrix");
    }

    std::vector<int> &operator[](int i)
    {
        if (i < 0 || i >= dim)
            throw std::out_of_range("Invalid position");
        if (!ref)
            return data[i];
        throw std::runtime_error("Read-only matrix");
    }

    friend std::ostream &operator<<(std::ostream &out, const Matrix &m)
    {
        for (int i = 0; i < m.size(); i++)
        {
            for (int j = 0; j < m.size(); j++)
                out << m.get(i, j) << " ";
            out << "\n";
        }
        return out;
    }

    int size() const { return dim; }

    int determinant() const
    {
        if (dim == 1)
            return get(0, 0);

        int result = 0;
        for (int i = 0; i < dim; i++)
            result += (i % 2 ? -1 : 1) * get(i, 0) * minor_matrix(i, 0).determinant();
        return result;
    }

    int determinant_parallel() const
    {
        if (dim == 1)
            return get(0, 0);

        std::atomic<int> result = 0;
        std::vector<std::thread> tasks;
        for (int i = 0; i < dim; i++)
        {
            tasks.emplace_back(std::thread([this, i, &result]
                                           {
                                               int temp = (i % 2 ? -1 : 1) * get(i, 0) * minor_matrix(i, 0).determinant();
                                               result.fetch_add(temp); }));
        }
        for (auto &t : tasks)
            t.join();

        return result;
    }

    Matrix minor_matrix(int r, int c) const { return Matrix(this, r, c); }

private:
    int dim;
    matrix_t data;
    const Matrix *ref;
    int row_offset, col_offset;
};

int main()
{
    // Пример матрицы 10x10
    const int size = 10;
        int matrix_data[size][size] = {
            {10, -8, 23, -47, 81, 12, 5, 6, 32, -70},
            {97, 91, -43, -28, 66, 18, -3, 45, -51, 74},
            {83, -35, -52, 49, 57, 63, 81, 42, -11, 53},
            {50, 60, 72, -79, 61, -33, 84, -7, 35, -65},
            {74, -19, 45, -12, -18, -77, 41, -34, -21, 82},
            {-378, -11, -47, 64, -91, 39, -48, -2, -87, -44},
            {-29, -83, -72, 89, 69, 47, -52, 54, 38, 30},
            {99, 59, 82, -42, 88, -25, -71, 27, 7, -89},
            {42, 75, 6, 9, -72, -7, -92, 14, -94, 37},
            {29, -78, -92, 35, 91, -22, -38, -53, 10, 29}
        };
    Matrix mat(size);

    // Заполняем матрицу данными
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            mat[i][j] = matrix_data[i][j];

    std::cout << "Input Matrix (" << size << "x" << size << "):\n";
    std::cout << mat << std::endl;

    TIMER_START(seq)
    int det_seq = mat.determinant();
    TIMER_STOP(seq)

    TIMER_START(par)
    int det_par = mat.determinant_parallel();
    TIMER_STOP(par)

    assert(det_seq == det_par);

    std::cout << "Determinant (Sequential): " << det_seq << "\nTime: " << TIMER_RESULT(seq) << " us" << std::endl;
    std::cout << "Determinant (Parallel): " << det_par << "\nTime: " << TIMER_RESULT(par) << " us" << std::endl;

    return 0;
}
