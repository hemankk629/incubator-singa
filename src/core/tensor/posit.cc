#include "singa/core/common.h"
#include "singa/core/posit.h"

namespace singa {

#define IS_POSIT_ZERO(flags) ((flags & 2) != 0)
#define IS_POSIT_NAR(flags) ((flags & 4) != 0)

#define SET_POSIT_ZERO(flags) flags |= 2
#define SET_POSIT_NAR(flags) flags |= 4

#define GET_POSIT_SIGN(flags) ((flags & 1) == 0) ? 1 : -1
#define SET_POSIT_SIGN_POSITIVE(flags) flags &= 0xFE
#define SET_POSIT_SIGN_NEGATIVE(flags) flags |= 1

posit_t posit_zero;
posit_t posit_one;
posit_t posit_minimum;
posit_t posit_maximum;
posit_t posit_nar;

void posit_init() {
    SET_POSIT_ZERO(posit_zero.flags);
    SET_POSIT_NAR(posit_nar.flags);
    posit_one = posit_from_int(1);
    // TODO    
}

bool posit_is_smaller(posit_t a, posit_t b) {
    int sign_a = GET_POSIT_SIGN(a.flags);
    int sign_b = GET_POSIT_SIGN(b.flags);
    if (sign_a == -1 && sign_b == 1)
        return true;
    if (sign_a == 1 && sign_b == -1)
        return false;
    if (a.exp < b.exp)
        return true;
    if (a.exp > b.exp)
        return false;
    if (a.fraction < b.fraction)
        return true;    
    return false;
}

bool posit_is_equal(posit_t a, posit_t b) {
    return (a.ps == b.ps) && (a.es == b.es) && (a.flags == b.flags) && (a.exp == b.exp) && (a.fraction == b.fraction);
}

bool posit_is_smaller_equal(posit_t a, posit_t b) {
    return posit_is_equal(a, b) || posit_is_smaller(a, b);
}

bool posit_is_bigger(posit_t a, posit_t b) {
    return posit_is_smaller(b, a);
}

bool posit_is_bigger_equal(posit_t a, posit_t b) {
    return posit_is_equal(a, b) || posit_is_smaller(b, a);
}

double posit_to_double(posit_t x) {
    if (IS_POSIT_ZERO(x.flags))
		return 0.0;
	if (IS_POSIT_NAR(x.flags))
		return NAN;

    int sign = GET_POSIT_SIGN(x.flags);   
	double fexp = (x.exp < 0) ? 1.0 / (1 << (-1 * x.exp)) : 1.0 * (1 << x.exp);
	double val = sign * fexp * (1.0 + ((double)x.fraction)/(1 << x.fs));

    return val;
}

posit_t posit_from_double(double val, uint8_t ps, uint8_t es) {
    int sign = 0;

	// ival - initial double value (save it)
	double ival = val;

	if (val < 0.0) {
		sign = -1;
		val = -val;
	}
	// special case - deal fast
	if (val == 0.0)
		return posit_zero;
	if (val == NAN)
		return posit_nar;

	// exp - total exponent
	int exp = (int)log2(val);
	if (exp <= 0 && exp > log2(val))
		exp--;
	// exp_val - the maximum value of the exponent field (2^es)
	int exp_val = (1<<es);
	// k - value of regime field in posit
	int k = exp / exp_val;
	// e - value of exponent field in posit
	int e = exp - k * exp_val;
	if (e < 0) {
		k--;
		e = exp - k * exp_val;
	}
	if (e > exp_val) {
		k++;
		e = exp - k * exp_val;
	}
	if (k >= ps - 2) {
#ifdef PDEBUG
		printf("Supra-maximal %f!\n", val);
#endif        
		if (sign == -1) {
			return posit_neg(posit_maximum);
		}
		else
			return posit_maximum;
	}
	if (k <= -ps + 2) {
#ifdef PDEBUG
		printf("Sub-minimal!\n");
#endif		
		if (sign == -1)
			return posit_neg(posit_minimum);
		else
			return posit_minimum;
	}

	double dexp = (1 << exp);
	if (exp < 0)
		dexp = 1.0 / (1 << (-exp));
	int fs = (k < 0) ? ps + k - 2 - es : ps - k - 3 - es;
	int frac = 0;
	if (fs > 0)
		frac = (val / dexp - 1.0) * (1 << fs);
	else
		fs = 0;

    posit_t ret;
    if (sign == -1)
        SET_POSIT_SIGN_NEGATIVE(ret.flags);
    else
        SET_POSIT_SIGN_POSITIVE(ret.flags);
    ret.ps = ps;
    ret.es = es;
    ret.fs = fs;
    ret.k = k;
    ret.e = e;
    ret.exp = exp;
    ret.fraction = frac;

    return ret;
}

float posit_to_float(posit_t x) {
    return (float)posit_to_double(x);
}

float to_float(posit_t x) {
    return posit_to_float(x);
}

float to_float(int x) {
    return static_cast<float>(x);
}

posit_t posit_from_float(float x) {    
    return posit_from_double((double)x, DEFAULT_PS, DEFAULT_ES);
}

posit_t posit_from_float(float x, uint8_t ps, uint8_t es) {    
    return posit_from_double((double)x, ps, es);
}

// TODO
posit_t posit_from_int(int x) {
    posit_t y;
    return y;
}

posit_t posit_abs(posit_t x) {
    if (GET_POSIT_SIGN(x.flags) == -1)
        return posit_neg(x);
    return x;
}

// TODO
posit_t posit_neg(posit_t x) {
    return x;
}

posit_t posit_sign(posit_t x) {
    return (GET_POSIT_SIGN(x.flags) == 1) ? posit_one : posit_neg(posit_one);
}

posit_t posit_max(posit_t a, posit_t b) {
    if (posit_is_bigger(a, b))
        return a;
    return b;
}

posit_t posit_min(posit_t a, posit_t b) {
    if (posit_is_smaller(a, b))
        return a;
    return b;
}

// TODO
posit_t posit_add(posit_t a, posit_t b) {
    if (IS_POSIT_NAR(a.flags) || IS_POSIT_NAR(b.flags))
        return posit_nar;
    if (IS_POSIT_ZERO(a.flags))
        return b;
    if (IS_POSIT_ZERO(b.flags))
        return a;
        

    posit_t y;
    return y;
}

// TODO
posit_t posit_sub(posit_t a, posit_t b) {
    posit_t y;
    return y;
}

// TODO
posit_t posit_mul(posit_t a, posit_t b) {
    if (IS_POSIT_NAR(a.flags) || IS_POSIT_NAR(b.flags))
        return posit_nar;
    if (IS_POSIT_ZERO(a.flags) || IS_POSIT_ZERO(b.flags))
        return posit_zero;
    if (posit_is_equal(a, posit_one))
        return b;
    if (posit_is_equal(b, posit_one))
        return a;
    int sign = GET_POSIT_SIGN(a.flags) * GET_POSIT_SIGN(b.flags);
    int exp = a.exp + b.exp;
    int frac = a.fraction * b.fraction;
}

// TODO
posit_t posit_div(posit_t a, posit_t b) {
    if (IS_POSIT_ZERO(b.flags))
        return posit_nar;
    if (IS_POSIT_ZERO(a.flags))
        return posit_zero;
    if (posit_is_equal(b, posit_one))
        return a;
    return a;
}

posit_t posit_sqrt(posit_t x) {
    double dx = posit_to_double(x);    
    return posit_from_double(sqrt(dx), x.ps, x.es);
}

posit_t posit_exp(posit_t x) {
    double dx = posit_to_double(x);
    return posit_from_double(exp(dx), x.ps, x.es);
}

posit_t posit_log(posit_t x) {
    double dx = posit_to_double(x);
    return posit_from_double(log(dx), x.ps, x.es);
}

posit_t posit_pow(posit_t x, posit_t y) {
    double dx = posit_to_double(x);
    double dy = posit_to_double(y);
    return posit_from_double(pow(dx, dy), x.ps, x.es);
}

posit_t posit_tanh(posit_t x) {
    double dx = posit_to_double(x);
    return posit_from_double(tanh(dx), x.ps, x.es);
}

int posit_cblas_isamax(const int __N, const posit_t *__X, const int __incX) {
	posit_t vmax = posit_abs(__X[0]);
	int idx = 0;
	for (int i = __incX; i < __N; i += __incX) {
		if (posit_is_bigger(posit_abs(__X[i]), vmax)) {
			vmax = posit_abs(__X[i]);
			idx = i;
		}
	}
	return idx;
}

posit_t posit_cblas_sasum(const int __N, const posit_t *__X, const int __incX) {
	posit_t sum = posit_abs(__X[0]);
	for (int i = __incX; i < __N; i += __incX)
		sum = posit_add(sum, posit_abs(__X[i]));
	return sum;
}

void posit_cblas_saxpy(const int __N, const posit_t __alpha, const posit_t *__X, const int __incX, posit_t *__Y, const int __incY) {
	// we suppose the strides are identical
	CHECK_EQ(__incX, __incY);
#pragma omp parallel for shared(__N, __incY, __X, __Y, __alpha) private(i)
	for (int i = 0; i < __N; i += __incY)
		__Y[i] = posit_add(__Y[i], posit_mul(__alpha, __X[i]));
}

posit_t posit_cblas_sdot(const int __N, const posit_t *__X, const int __incX, const posit_t *__Y, const int __incY) {
	// we suppose the strides are identical
	CHECK_EQ(__incX, __incY);
	posit_t sum = posit_mul(__X[0], __Y[0]);
	for (int i = __incY; i < __N; i += __incY) {
		sum = posit_add(sum, posit_mul(__X[i], __Y[i]));
	}
	return sum;
}

void posit_cblas_sscal(const int __N, const posit_t __alpha, posit_t *__X, const int __incX) {
#pragma omp parallel for shared(__N, __incX, __X, __alpha) private(i)
	for (int i = 0; i < __N; i += __incX) {
		__X[i] = posit_mul(__X[i], __alpha);
	}
}

posit_t posit_cblas_snrm2(const int __N, const posit_t *__X, const int __incX) {
	posit_t norm = posit_mul(__X[0], __X[0]);
	for (int i = __incX; i < __N; i += __incX)
		norm = posit_add(norm, posit_mul(__X[i], __X[i]));
	return posit_sqrt(norm);
}

void posit_cblas_sgemv(const enum MY_CBLAS_ORDER __Order,
		const enum MY_CBLAS_TRANSPOSE __TransA,
		const int __M, const int __N,
		const posit_t __alpha,
		const posit_t *__A, const int __lda,
		const posit_t *__X, const int __incX,
		const posit_t __beta,
		posit_t *__Y, const int __incY) {

	// LOG(INFO) << "In my_cblas_sgemv " << (__TransA == CblasTrans);

	int n = (__TransA == MyCblasTrans) ? __M : __N;
	int m = (__TransA == MyCblasTrans) ? __N : __M;

	if (__TransA == MyCblasTrans) {
		for (int r = 0; r < m; r++) {
			posit_t sum = posit_zero;
			for (int c = 0; c < n; c++) {
				int idx = c * m + r;
				sum = posit_add(sum, posit_mul(__A[idx], __X[c]));
			}
			__Y[r] = posit_add(posit_mul(__alpha, sum), posit_mul(__beta, __Y[r]));
		}
	}
	else {
		for (int r = 0; r < m; r++) {
			posit_t sum = posit_zero;
			for (int c = 0; c < n; c++) {
				int idx = r * n + c;
				sum = posit_add(sum, posit_mul(__A[idx], __X[c]));
			}
			__Y[r] = posit_add(posit_mul(__alpha, sum), posit_mul(__beta, __Y[r]));
		}
	}
}

void CxMxM(const posit_t* A, const posit_t* B, posit_t* C, size_t M, size_t N, size_t K, posit_t alpha, char reset) {
	size_t i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			if (reset)
				C[i*N+j] = posit_zero;
			posit_t sum = posit_zero;
			for (k = 0; k < K; k++)
				sum = posit_add(sum, posit_mul(A[i*K+k], B[k*N+j]));
			C[i*N+j] = posit_add(C[i*N+j], posit_mul(alpha, sum));
		}
	}
}

void CxMTxM(const posit_t* A, const posit_t* B, posit_t* C, size_t M, size_t N, size_t K, posit_t alpha, char reset) {
	size_t i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			if (reset)
				C[i*N+j] = posit_zero;
			posit_t sum = posit_zero;
			for (k = 0; k < K; k++)
				sum = posit_add(sum, posit_mul(A[k*M+i], B[k*N+j]));
			C[i*N+j] = posit_add(C[i*N+j], posit_mul(alpha, sum));
		}
	}
}

void CxMxMT(const posit_t* A, const posit_t* B, posit_t* C, size_t M, size_t N, size_t K, posit_t alpha, char reset) {
	size_t i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			if (reset)
				C[i*N+j] = posit_zero;
			posit_t sum = posit_zero;
			for (k = 0; k < K; k++)
				sum = posit_add(sum, posit_mul(A[i*K+k], B[j*K+k]));
			C[i*N+j] = posit_add(C[i*N+j], posit_mul(alpha, sum));
		}
	}
}

void CxMTxMT(const posit_t* A, const posit_t* B, posit_t* C, size_t M, size_t N, size_t K, posit_t alpha, char reset) {
	size_t i, j, k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			if (reset)
				C[i*N+j] = posit_zero;
			posit_t sum = posit_zero;
			for (k = 0; k < K; k++)
				sum = posit_add(sum, posit_mul(A[k*M+i], B[j*K+k]));
			C[i*N+j] = posit_add(C[i*N+j], posit_mul(alpha, sum));
		}
	}
}

void MpM(posit_t* A, const posit_t* B, size_t M, size_t N) {
	size_t i, j;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			A[i*N+j] = posit_add(A[i*N+j], B[i*N+j]);
		}
	}
}

void CxM(posit_t* A, const posit_t alpha, size_t M, size_t N) {
	size_t i, j;
	for (i = 0; i < M; i++)
		for (j = 0; j < N; j++)
			A[i*N+j] = posit_mul(A[i*N+j], alpha);
}

void posit_cblas_sgemm(const enum MY_CBLAS_ORDER __Order,
		const enum MY_CBLAS_TRANSPOSE __TransA,
		const enum MY_CBLAS_TRANSPOSE __TransB,
		const int __M, const int __N, const int __K,
		const posit_t __alpha,
		const posit_t *__A, const int __lda,
		const posit_t *__B, const int __ldb,
		const posit_t __beta,
		posit_t *__C, const int __ldc) {

	// LOG(INFO) << "In my_cblas_sgemm";

	if (__TransA == MyCblasNoTrans) {
		if (__TransB == MyCblasNoTrans) {
			CxM(__C, __beta, __M, __N);
			CxMxM(__A, __B, __C, __M, __N, __K, __alpha, 0);
		}
		else {
			CxM(__C, __beta, __M, __N);
			CxMxMT(__A, __B, __C, __M, __N, __K, __alpha, 0);
		}
	}
	else {
		if (__TransB == MyCblasNoTrans) {
			CxM(__C, __beta, __N, __K);
			CxMTxM(__A, __B, __C, __M, __N, __K, __alpha, 0);
		}
		else {
			CxM(__C, __beta, __N, __N);
			CxMTxMT(__A, __B, __C, __M, __N, __K, __alpha, 0);
		}
	}
}

} // end namespace