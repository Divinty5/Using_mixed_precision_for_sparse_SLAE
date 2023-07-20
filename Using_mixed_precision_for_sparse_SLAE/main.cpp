#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <iomanip>
#include <sstream>
#include <chrono>

using namespace std;

// Структура для матричек в формате csr (compressed sparse row)
struct CRSArrays {
    MKL_INT m;  //< the dim of the matrix
    MKL_INT nnz;//< the number of nnz (== ia[m])
    double* a;  //< the values (of size nnz)
    MKL_INT* ia;//< the usual rowptr (of size m+1)
    MKL_INT* ja;//< the colidx of each NNZ (of size nnz)

    CRSArrays() {
        a = NULL;
        ia = NULL;
        ja = NULL;
    }

    CRSArrays(const CRSArrays& other)
    {
        if (other.a == NULL)
        {
            cout << "passing a null array!!!!";// Передается не заполненная структура!!!!!!
        }
        this->m = other.m;
        this->nnz = other.nnz;
        this->a = new double[this->nnz];
        this->ia = new MKL_INT[this->m + 1];
        this->ja = new MKL_INT[this->nnz];
        for (int i = 0; i < this->nnz; i++)
        {
            this->a[i] = other.a[i];
            this->ja[i] = other.ja[i];
        }
        for (int i = 0; i < (this->m + 1); i++)
        {
            this->ia[i] = other.ia[i];
        }
    }
    CRSArrays& operator =(const CRSArrays& other)
    {
        if (this->a != NULL)
        {
            delete[] this->a;
            delete[] this->ja;
            delete[] this->ia;
        }
        this->m = other.m;
        this->nnz = other.nnz;
        this->a = new double[this->nnz];
        this->ia = new MKL_INT[this->m + 1];
        this->ja = new MKL_INT[this->nnz];
        for (int i = 0; i < this->nnz; i++)
        {
            this->a[i] = other.a[i];
            this->ja[i] = other.ja[i];
        }
        for (int i = 0; i < (this->m + 1); i++)
        {
            this->ia[i] = other.ia[i];
        }
        return *this;
    }
    ~CRSArrays() {
        delete[] a;
        delete[] ia;
        delete[] ja;
    }
};

// Структура для матричек в формате COO (Coordinate)
struct COOArrays {
    MKL_INT m;      //< the dimension of the matrix
    MKL_INT nnz;    //< the number of nnz inside the matrix
    double* val;    //< the values (size = nnz)
    MKL_INT* rowind;//< the row indexes (size = nnz)
    MKL_INT* colind;//< the col indexes (size = nnz)

    //simply set ptr to null
    COOArrays() {
        val = NULL;
        rowind = NULL;
        colind = NULL;
    }

    //delete ptr
    ~COOArrays() {
        delete[] val;
        delete[] rowind;
        delete[] colind;
    }
};

void ex_csrsort(int n, double* a, int* ja) {
    int i, j;
    double tt;
    int it;
    for (j = 1; j < n; j++) {
        for (i = 1; i < n; i++) {
            if (ja[i] < ja[i - 1]) {
                tt = a[i];
                a[i] = a[i - 1];
                a[i - 1] = tt;

                it = ja[i];
                ja[i] = ja[i - 1];
                ja[i - 1] = it;
            }
        }
    }

}

void ex_convert_COO_2_CSR(int n, int m, int nnz, int* coo_ia, int* coo_ja, double* coo_a, int** csr_ia, int** csr_ja, double** csr_a) {

    int* pbegin;

    int* ia, * ja;
    double* a;
    int i;

    ia = (int*)malloc((n + 1) * sizeof(int));
    ja = (int*)malloc(nnz * sizeof(int));
    a = (double*)malloc(nnz * sizeof(double));
    if ((ia == NULL) || (ja == NULL) || (a == NULL)) {
        printf("ERROR   : Could not allocate arrays for CSR matrix \n");
        exit(1);
    }

    // Convert from COO -> CSR: Find array ia
    for (i = 0; i < n + 1; i++) {
        ia[i] = 0;
    }
    for (i = 0; i < nnz; i++) {
        ia[coo_ia[i] + 1]++;
    }
    for (i = 1; i < n + 1; i++) {
        ia[i] += ia[i - 1];
    }

    // Convert from COO -> CSR: Find arrays ja, a
    pbegin = (int*)malloc(n * sizeof(int));
    if (pbegin == NULL) {
        printf("ERROR   : Could not allocate arrays for converter COO->CSR \n");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        pbegin[i] = ia[i];
    }
    for (i = 0; i < nnz; i++) {
        ja[pbegin[coo_ia[i]]] = coo_ja[i];
        a[pbegin[coo_ia[i]]] = coo_a[i];
        pbegin[coo_ia[i]]++;
    }

    // Convert from COO -> CSR: modify lines to be align with CSR spesification
    for (i = 0; i < n; i++) {
        ex_csrsort(ia[i + 1] - ia[i], &(a[ia[i]]), &(ja[ia[i]]));
    }

    *csr_ia = ia;
    *csr_ja = ja;
    *csr_a = a;
}

void load_matrix(string Matrix_file_name, CRSArrays* crs)
{
    string line;
    stringstream x;
    char word[256] = {};
    COOArrays coo;

    //File.mtx -> COO
    fstream mtx(Matrix_file_name);
    if (mtx)
    {
        while (getline(mtx, line))
        {
            if ((int(line[0]) == int('%'))) {
                //cout << line << endl;
            }
            else break;
        }
        x << line;
        int i = 0;
        while (x >> word) {
            switch (i)
            {
            case 0:
                coo.m = stoi(word);
            case 2:
                coo.nnz = stoi(word);
            default:
                break;
            }

            i++;
        }
        double* val = new double[coo.nnz];    //< the values (size = nnz)
        MKL_INT* rowind = new MKL_INT[coo.nnz];  //< the row indexes (size = nnz)
        MKL_INT* colind = new MKL_INT[coo.nnz];  //< the col indexes (size = nnz)

        int j = 0;
        while (getline(mtx, line))
        {
            i = 0;
            x.clear();
            x << line;
            while (x >> word) {
                switch (i)
                {
                case 0:
                    rowind[j] = stoi(word) - 1;

                case 1:
                    colind[j] = stoi(word) - 1;
                case 2:
                    val[j] = stod(word);

                default:
                    break;
                }
                i++;
            }
            j++;
            coo.rowind = rowind;
            coo.colind = colind;
            coo.val = val;
        }

        mtx.close();

        std::cout << "Reading " << Matrix_file_name << " done" << endl;
    }
    else std::cout << "File " << Matrix_file_name << " don't exist" << endl;
    x.clear();

    //COO -> CRS
    ex_convert_COO_2_CSR(coo.m, coo.m, coo.nnz, coo.rowind, coo.colind, coo.val, &((*crs).ia), &((*crs).ja), &((*crs).a));
    (*crs).m = coo.m;
    (*crs).nnz = coo.nnz;

    std::cout << "Convert " << Matrix_file_name << " done" << endl << endl;
}

//процедура получает 3 массива (csr), хранящие сим. матрицу как нижнетреугольную, и выдает 3 массива, которые хранят сим. матрицу как полную матрицу
void sym_to_gen(int nrows, MKL_INT* ia, MKL_INT* ja, double* a, MKL_INT* ia1, MKL_INT* ja1, double* a1) {
    int z = 0; ia1[0] = 0;
    for (int i = 0; i < nrows; i++) {
        for (int j = ia[i]; j < ia[i + 1]; j++) {
            a1[z] = a[j]; ja1[z] = ja[j]; z++;
        }
        for (int k = i + 1; k < nrows; k++) {
            for (int j = ia[k]; j < ia[k + 1]; j++) {
                if (ja[j] > i) break;
                if (ja[j] == i) { a1[z] = a[j]; ja1[z] = k; z++; break; }
            }
        }
        ia1[i + 1] = z;
    }
    return;
}
//процедура ниже так и не пригодилась :(
void b_to_lu(int nrows, MKL_INT* ia, MKL_INT* ja, double* a, MKL_INT* ial, MKL_INT* jal, double* al, MKL_INT* iau, MKL_INT* jau, double* au) {
    int z1 = 0, z2 = 0; ial[0] = 0; iau[0] = 0;
    for (int i = 0; i < nrows; i++) {
        for (int j = ia[i]; j < ia[i + 1]; j++) {
            if (ja[j] < i) { jal[z1] = ja[j]; al[z1] = a[j]; z1++; }
            else if (ja[j] == i) { jal[z1] = i; al[z1] = 1; z1++; jau[z2] = ja[j]; au[z2] = a[j]; z2++; }
            else if (ja[j] > i) { jau[z2] = ja[j]; au[z2] = a[j]; z2++; }
        }
        ial[i + 1] = z1; iau[i + 1] = z2;
    }
    return;
}
//разложение Холецкого версия 0
void ICC(int nrows, MKL_INT* ia, MKL_INT* ja, double* a) {
    int s, z, z1, z2;
    for (int i = 1; i < nrows; i++) {
        s = ia[i] - 1; //номер диагонального элемента в строке (i-1)((считаем что индексация идет с 0!))
        for (int k = ia[i - 1]; k < s; k++) {
            a[s] -= (a[k]) * (a[k]); //k - номера (ненулевых) элементов, стоящих слева от диагонального на этой строке (строке с индексом (i-1))
        }
        if (a[s] < 0) cout << "ERROR" << '\t' << s << endl;
        a[s] = sqrt(a[s]); //ПОД КОРНЕМ МОГУТ ВОЗНИКНУТЬ ОТРИЦАТЕЛЬНЫЕ ЧИСЛА
        for (int j = i + 1; j < nrows; j++) {
            z = -1; //z - номер элемента (в дальнейшем), который в COO имеет индексы (i-1; j-1) (в случае если он ненулевой)
            for (int zz = ia[j - 1]; zz < ia[j]; zz++) { //в этом цикле проверяем наличие элемента (ненулевого) с индексами (j-1; i-1) и, если он есть, то пересчитываем его (соотв. если его нет - то идем дальше)
                if (ja[zz] > i - 1) break;
                if (ja[zz] == i - 1) { z = zz; break; }
            }
            if (z != -1) {
                for (int k = 1; k < i; k++) {
                    z1 = -1; z2 = -1; //z1, z2 - номера элементов (в дальнейшем), которые в COO имеет индексы (i-1; k-1) и (j-1; k-1) (в случае если они ненулевые)
                    for (int zz = ia[j - 1]; zz < ia[j]; zz++) {
                        if (ja[zz] > k - 1) break;
                        if (ja[zz] == k - 1) { z1 = zz; break; }
                    }
                    if (z1 != -1) {
                        for (int zz = ia[i - 1]; zz < ia[i]; zz++) {
                            if (ja[zz] > k - 1) break;
                            if (ja[zz] == k - 1) { z2 = zz; break; }
                        }
                        if (z2 != -1) a[z] -= a[z1] * a[z2];
                    }
                }
                a[z] = a[z] / a[s];
            }
        }
    }
    return;
}
//разложение Холецкого версия 1
void ICC_1(int nrows, MKL_INT* ia, MKL_INT* ja, double* a) {
    int s, z, z1, z2;
    for (int i = 1; i < nrows; i++) {
        z = ia[i] - 1; //номер дагонального элемента в (i-1) строке
        for (int j = ia[i - 1]; j < z; j++) {
            s = ja[j]; //индекс столбца j-го элемента 
            for (int k = 0; k < s; k++) {
                z1 = -1; z2 = -1;
                for (int zz = ia[s]; zz < ia[s + 1]; zz++) {
                    if (ja[zz] > k) break;
                    if (ja[zz] == k) { z1 = zz; break; }
                }
                if (z1 != -1) {
                    for (int zz = ia[i - 1]; zz < j; zz++) {
                        if (ja[zz] > k) break;
                        if (ja[zz] == k) { z2 = zz; break; }
                    }
                    if (z2 != -1) { a[j] -= a[z1] * a[z2]; }
                }
            }
            a[j] /= a[ia[s + 1] - 1];
            a[z] -= a[j] * a[j];
        }
        a[z] = sqrt(a[z]);
    }
    return;
}
//разложение Холецкого версия 2 (актуальная)
int ICC_2(int nrows, MKL_INT* ia, MKL_INT* ja, double* a) {
    int z, l, start, end;
    for (int i = 1; i < nrows; i++) {
        z = ia[i] - 1; //номер дагонального элемента в (i-1) строке
        for (int k = ia[i - 1]; k < z; k++) {
            l = ia[i - 1];
            start = ja[l];
            end = ja[k];
            for (int s = ia[end]; s < ia[end + 1] && ja[s] < end; s++) {
                while (ja[l] < ja[s] && start < end) { l++; start = ja[l]; }
                if (ja[l] > ja[s]) continue;
                if (ja[l] == ja[s]) { a[k] -= a[l] * a[s]; }
            }
            a[k] /= a[ia[end + 1] - 1];
            a[z] -= a[k] * a[k];
        }
        if (a[z] < 0) return -1;
        a[z] = sqrt(a[z]);
    }
    return 0;
}

void CG( // Метод сопряжённых градиентов (Conjugate gradients)
    MKL_INT nrows,                    //число строк в матрице
    sparse_matrix_t& A, //матрица A в формате CSR
    double* u,                   //вектор решения
    double* f,                    //правая часть
    double eps,                 //точность для критерия остановки
    double* res,
    int itermax,                  //максимальное число итераций для критерия остановки
    int* number_of_operations   //суммарное число проделанных операций
)
{
    double* Ap = new double[nrows];
    double* p = new double[nrows];
    double* r = new double[nrows];

    int iter = 0;
    double alpha = 0.0;
    double beta = 0.0;
    double rr = 0;

    matrix_descr descr = { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT };

    copy(f, f + nrows, r);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., A, descr, u, 1, r);

    eps *= fabs(f[cblas_idamax(nrows, f, 1)]);

    copy(r, r + nrows, p);
    rr = cblas_ddot(nrows, r, 1, r, 1);

    while (fabs(r[cblas_idamax(nrows, r, 1)]) > eps && iter < itermax)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, p, 0, Ap);

        alpha = rr / cblas_ddot(nrows, Ap, 1, p, 1);

        cblas_daxpy(nrows, alpha, p, 1, u, 1);
        cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);
        //cout << fabs(r[cblas_idamax(nrows, r, 1)]) << endl;
        beta = rr;
        rr = cblas_ddot(nrows, r, 1, r, 1);
        beta = rr / beta;

        cblas_daxpby(nrows, 1, r, 1, beta, p, 1);

        iter++;
        *number_of_operations = iter;
    }
    *res = fabs(r[cblas_idamax(nrows, r, 1)]);
    delete[] Ap;
    delete[] p;
    delete[] r;
}

void P_ICC_CG( // Метод предоб. сопряжённых градиентов (неполная факторизация Холесского)
    int nrows,                    //число строк в матрице
    sparse_matrix_t& A,           //матрица A в формате CSR
    sparse_matrix_t& C,            //матрица С в формате CSR
    double* u,                   //вектор решения
    double* f,                    //правая часть
    double eps,                 //точность для критерия остановки
    double* res,
    int itermax,                  //максимальное число итераций для критерия остановки
    int* number_of_operations   //суммарное число проделанных операций
)
{
    double* Ap = new double[nrows];
    double* p = new double[nrows];
    double* r = new double[nrows];
    double* z = new double[nrows];

    int iter = 0;
    double alpha = 0.0;
    double beta = 0.0;
    double rz = 0;

    matrix_descr descr = { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT };

    copy(f, f + nrows, r);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., A, descr, u, 1, r);

    eps *= fabs(f[cblas_idamax(nrows, f, 1)]);

    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1., C, descr, r, Ap);
    mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1., C, descr, Ap, z);

    copy(z, z + nrows, p);
    rz = cblas_ddot(nrows, r, 1, z, 1);

    while (fabs(r[cblas_idamax(nrows, r, 1)]) > eps && iter < itermax)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, p, 0, Ap);

        alpha = rz / cblas_ddot(nrows, Ap, 1, p, 1);

        cblas_daxpy(nrows, alpha, p, 1, u, 1);
        cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

        //cout << fabs(r[cblas_idamax(nrows, r, 1)]) << endl;

        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, C, descr, r, Ap);
        mkl_sparse_d_trsv(SPARSE_OPERATION_TRANSPOSE, 1, C, descr, Ap, z);

        beta = rz;
        rz = cblas_ddot(nrows, r, 1, z, 1);
        beta = rz / beta;

        cblas_daxpby(nrows, 1, z, 1, beta, p, 1);

        iter++;
        *number_of_operations = iter;
    }
    *res = fabs(r[cblas_idamax(nrows, r, 1)]);
    delete[] Ap;
    delete[] p;
    delete[] r;
    delete[] z;
}

void P_ILU0_CG( // Метод предоб. сопряжённых градиентов (неполное LU разложение)
    int nrows,                    //число строк в матрице
    sparse_matrix_t& A,           //матрица A в формате CSR
    sparse_matrix_t& B,
    double* u,                   //вектор решения
    double* f,                    //правая часть
    double eps,                 //точность для критерия остановки
    double* res,
    int itermax,                  //максимальное число итераций для критерия остановки
    int* number_of_operations   //суммарное число проделанных операций
)
{
    double* Ap = new double[nrows];
    double* p = new double[nrows];
    double* r = new double[nrows];
    double* z = new double[nrows];

    int iter = 0;
    double alpha = 0.0;
    double beta = 0.0;
    double rz = 0;

    matrix_descr descr = { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT };
    matrix_descr descr1 = { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT };
    matrix_descr descr2 = { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_UNIT };

    copy(f, f + nrows, r);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., A, descr, u, 1, r);
    eps *= fabs(f[cblas_idamax(nrows, f, 1)]);

    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1., B, descr2, r, Ap);
    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1., B, descr1, Ap, z);

    copy(z, z + nrows, p);

    rz = cblas_ddot(nrows, r, 1, z, 1);

    while (fabs(r[cblas_idamax(nrows, r, 1)]) > eps && iter < itermax)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, p, 0, Ap);

        alpha = rz / cblas_ddot(nrows, Ap, 1, p, 1);

        cblas_daxpy(nrows, alpha, p, 1, u, 1);
        cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

        //cout << fabs(r[cblas_idamax(nrows, r, 1)]) << endl;

        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, B, descr2, r, Ap);
        mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, B, descr1, Ap, z);

        beta = rz;
        rz = cblas_ddot(nrows, r, 1, z, 1);
        beta = rz / beta;

        cblas_daxpby(nrows, 1, z, 1, beta, p, 1);

        iter++;
        *number_of_operations = iter;
    }
    *res = fabs(r[cblas_idamax(nrows, r, 1)]);
    delete[] Ap;
    delete[] p;
    delete[] r;
    delete[] z;
}

int main() {
    int pred = 1; //0 - если без предоб., 1 - если предоб. Холецкого, 2 - если предоб. ILU0
    double eps = 1e-7;
    int nit = 0;
    double res;
    CRSArrays crs;
    load_matrix("parabolic_fem.mtx", &crs);
    sparse_matrix_t A;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, crs.m, crs.m, crs.ia, (crs.ia + 1), crs.ja, crs.a);

    double* exact = new double[crs.m]; //exact. solution
    for (int i = 0; i < crs.m; i++) { exact[i] = 1; }
    double* u = new double[crs.m];  //Initial approximation
    for (int i = 0; i < crs.m; i++) { u[i] = 0; }
    double* f = new double[crs.m]; //right part
    matrix_descr descr = { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT };
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descr, exact, 0, f);

    if (pred == 0) {
        auto start = chrono::high_resolution_clock::now();
        CG(crs.m, A, u, f, eps, &res, 100000, &nit);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cblas_daxpy(crs.m, -1., u, 1, exact, 1); //вектор exact становится вектором ошибки
        cout << "Conjugate Gradient" << endl;
        cout << "Size of matrix: " << crs.m << "\nNumber of iterations: " << nit << "\nNorm of residual vector: " << res << endl;
        cout << "Error: " << fabs(exact[cblas_idamax(crs.m, exact, 1)]) << endl;
        cout << "Time: " << duration.count() << endl << endl;
    }
    if (pred == 1) {
        auto start = chrono::high_resolution_clock::now();
        sparse_matrix_t C;
        CRSArrays crs2(crs);
        //процедура ICC_2 считает элементы матрицы Холецкого
        if (ICC_2(crs2.m, crs2.ia, crs2.ja, crs2.a) == -1) {
            cout << "ICC failed :(" << endl;
            return 0;
        }
        mkl_sparse_d_create_csr(&C, SPARSE_INDEX_BASE_ZERO, crs2.m, crs2.m, crs2.ia, (crs2.ia + 1), crs2.ja, crs2.a);

        P_ICC_CG(crs.m, A, C, u, f, eps, &res, 100000, &nit);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cblas_daxpy(crs.m, -1., u, 1, exact, 1);
        cout << "Preconditioned (ICC) Conjugate Gradient" << endl;
        cout << "Size of matrix: " << crs.m << "\nNumber of iterations: " << nit << "\nNorm of residual vector: " << res << endl;
        cout << "Error: " << fabs(exact[cblas_idamax(crs.m, exact, 1)]) << endl;
        cout << "Time: " << duration.count() << endl << endl;
    }
    if (pred == 2) {
        auto start = chrono::high_resolution_clock::now();
        CRSArrays crs1;
        crs1.m = crs.m;
        crs1.nnz = 2 * crs.nnz - crs1.m;
        crs1.a = new double[crs1.nnz];
        crs1.ja = new MKL_INT[crs1.nnz];
        crs1.ia = new MKL_INT[(crs1.m) + 1];
        sym_to_gen(crs.m, crs.ia, crs.ja, crs.a, crs1.ia, crs1.ja, crs1.a); //процедура составляет 3 массива, которые хранят симметричную марицу не как нижнетреугольную, а как полную матрицу (эл-ты есть как сверху, так и снизу ) 
        //меняем индексацию новой матрицы с 0 на 1 (только так работает процедура для подсчета ILU0)
        for (int i = 0; i < crs1.m + 1; i++) {
            crs1.ia[i] += 1;
            crs1.ja[i] += 1;
        }
        for (int i = crs1.m + 1; i < crs1.nnz; i++) {
            crs1.ja[i] += 1;
        }
        //делаем два нужных массива, и как-то их заполняем)
        MKL_INT ipar[128];
        double dpar[128];
        ipar[1] = 6; ipar[5] = 1; ipar[30] = 0; dpar[30] = 0; dpar[31] = 0; MKL_INT ierr = 0;
        //создаем массив bilu0, где будут храниться эл-ты матриц L и U, и запускаем процедуру по подсчету эл-тов L и U
        double* bilu0 = new double[crs1.nnz];
        dcsrilu0(&crs1.m, crs1.a, crs1.ia, crs1.ja, bilu0, ipar, dpar, &ierr);
        //возвращаем индексацию матрицы с 1 на 0
        for (int i = 0; i < crs1.m + 1; i++) {
            crs1.ia[i] -= 1;
            crs1.ja[i] -= 1;
        }
        for (int i = crs1.m + 1; i < crs1.nnz; i++) {
            crs1.ja[i] -= 1;
        }
        sparse_matrix_t B;
        mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, crs1.m, crs1.m, crs1.ia, (crs1.ia + 1), crs1.ja, bilu0);

        P_ILU0_CG(crs.m, A, B, u, f, eps, &res, 100000, &nit);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cblas_daxpy(crs.m, -1., u, 1, exact, 1);
        cout << "Preconditioned (ILU0) Conjugate Gradient" << endl;
        cout << "Size of matrix: " << crs.m << "\nNumber of iterations: " << nit << "\nNorm of residual vector: " << res << endl;
        cout << "Error: " << fabs(exact[cblas_idamax(crs.m, exact, 1)]) << endl;
        cout << "Time: " << duration.count() << endl << endl;
        delete[] bilu0;
    }

    delete[] exact;
    delete[] u;
    delete[] f;
    return 0;
}