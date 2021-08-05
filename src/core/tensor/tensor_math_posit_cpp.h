/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SINGA_CORE_TENSOR_TENSOR_MATH_POSIT_CPP_H_
#define SINGA_CORE_TENSOR_TENSOR_MATH_POSIT_CPP_H_

#include "singa/core/posit.h"
#include "./tensor_math.h"
//#include "./stacktrace.h"
#include "singa/core/common.h"
#include "singa/core/tensor.h"
//#include <math.h>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <iostream>

#ifdef USE_CBLAS
#undef USE_CBLAS
// #include <cblas.h>
#endif

namespace singa {

// ******************************************************************************************
// traversal operations end
// ******************************************************************************************

template <>
void Abs<posit_t, lang::Cpp>(const Tensor& in, Tensor* out, Context *ctx) {
  traverse_unary<posit_t>(in, out, [](posit_t x) {return posit_abs(x);});
}

template <>
void Add<posit_t, lang::Cpp>(const Tensor& in, const posit_t x, Tensor* out, Context *ctx) {
  auto add_lambda = [&x](posit_t a) {
    return posit_add(x, a);
  };
  traverse_unary<posit_t>(in, out, add_lambda);
}

template <>
void Add<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  auto add_lambda_binary = [](posit_t a, posit_t b) {
    return posit_add(a, b);
  };
  traverse_binary<posit_t>(in1, in2, out, add_lambda_binary);
}

template <>
void Clamp<posit_t, lang::Cpp>(const posit_t low, const posit_t high,
                             const Tensor& in, Tensor* out,
                             Context *ctx) {
  auto clamp_lambda = [&low, &high](posit_t a) {
    if (posit_is_smaller(a, low)) {return low;}
    else if (posit_is_bigger(a, high)) {return high;}
    else {return a;}
  };
  traverse_unary<posit_t>(in, out, clamp_lambda);
}

template <>
void Div<posit_t, lang::Cpp>(const posit_t x, const Tensor& in, Tensor* out,
                           Context *ctx) {
  auto const_div = [&x](posit_t a) {return posit_div(x, a);};
  traverse_unary<posit_t>(in, out, const_div);
}

template <>
void Div<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2,
                           Tensor* out, Context *ctx) {
  auto binary_div = [](posit_t a, posit_t b) {return posit_div(a, b);};
  traverse_binary<posit_t>(in1, in2, out, binary_div);
}

template <>
void EltwiseMult<posit_t, lang::Cpp>(const Tensor& in, const posit_t x, Tensor* out,
                                   Context *ctx) {
  auto eltwisemult_lambda = [&x](posit_t a) {
    return posit_mul(a, x);
  };
  traverse_unary<posit_t>(in, out, eltwisemult_lambda);
}

template <>
void EltwiseMult<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                                   Context *ctx) {
  auto eltwisemult_lambda_binary = [](posit_t a, posit_t b) {
    return posit_mul(a, b);
  };
  traverse_binary<posit_t>(in1, in2, out, eltwisemult_lambda_binary);
}

template <>
void Exp<posit_t, lang::Cpp>(const Tensor& in, Tensor *out, Context *ctx) {
  traverse_unary<posit_t>(in, out, [](posit_t x) {return posit_exp(x);});
}

template <>
void GE<posit_t, lang::Cpp>(const Tensor& in, const posit_t x, Tensor* out,
                          Context *ctx) {
  auto ge_lambda = [&x](posit_t a) {
    return (posit_is_bigger_equal(a, x)) ? posit_one : posit_zero;
  };
  traverse_unary<posit_t>(in, out, ge_lambda);
}

template <>
void GE<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto ge_lambda_binary = [](posit_t a, posit_t b) {
    return (posit_is_bigger_equal(a, b)) ? posit_one : posit_zero;
  };
  traverse_binary<posit_t>(in1, in2, out, ge_lambda_binary);
}

template <>
void GT<posit_t, lang::Cpp>(const Tensor& in, const posit_t x, Tensor* out,
                          Context *ctx) {
  auto gt_lambda = [&x](posit_t a) {
    return (posit_is_bigger(a, x)) ? posit_one : posit_zero;
  };
  traverse_unary<posit_t>(in, out, gt_lambda);
}

template <>
void GT<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto gt_lambda_binary = [](posit_t a, posit_t b) {
   return (posit_is_bigger(a, b)) ? posit_one : posit_zero;
  };
  traverse_binary<posit_t>(in1, in2, out, gt_lambda_binary);
}

template <>
void LE<posit_t, lang::Cpp>(const Tensor& in, const posit_t x, Tensor* out,
                          Context *ctx) {
  auto le_lambda = [&x](posit_t a) {
    return (posit_is_smaller_equal(a, x)) ? posit_one : posit_zero;
  };
  traverse_unary<posit_t>(in, out, le_lambda);
}

template <>
void LE<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto le_lambda_binary = [](posit_t a, posit_t b) {
    return (posit_is_smaller_equal(a, b)) ? posit_one : posit_zero;
  };
  traverse_binary<posit_t>(in1, in2, out, le_lambda_binary);
}

template <>
void Log<posit_t, lang::Cpp>(const Tensor& in, Tensor* out,
                           Context *ctx) {
  auto ulog = [](posit_t a) {return posit_log(a);};
  traverse_unary<posit_t>(in, out, ulog);
}

template <>
void LT<posit_t, lang::Cpp>(const Tensor& in, const posit_t x, Tensor* out,
                          Context *ctx) {
  auto lt_lambda = [&x](posit_t a) {
    return (posit_is_smaller(a, x)) ? posit_one : posit_zero;
  };
  traverse_unary<posit_t>(in, out, lt_lambda);
}

template <>
void LT<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                          Context *ctx) {
  auto lt_lambda_binary = [](posit_t a, posit_t b) {
    return (posit_is_smaller(a, b)) ? posit_one : posit_zero;
  };
  traverse_binary<posit_t>(in1, in2, out, lt_lambda_binary);
}

template <>
void Pow<posit_t, lang::Cpp>(const Tensor& in, const posit_t x, Tensor *out, Context *ctx) {
  traverse_unary<posit_t>(in, out, [x](posit_t y) {return posit_pow(y, x);});
}

template <>
void Pow<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2, Tensor* out,
                           Context *ctx) {
  auto pow_lambda_binary = [](posit_t a, posit_t b) {
    return posit_pow(a, b);
  };
  traverse_binary<posit_t>(in1, in2, out, pow_lambda_binary);
}

template <>
void ReLU<posit_t, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  auto relu_lambda = [](posit_t a) {
    return (posit_is_bigger_equal(a, posit_zero)) ? a : posit_zero;
  };
  traverse_unary<posit_t>(in, out, relu_lambda);
}

template <>
void Set<posit_t, lang::Cpp>(const posit_t x, Tensor* out,
                           Context *ctx) {
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) outPtr[i] = x;
}

template <>
void Sigmoid<posit_t, lang::Cpp>(const Tensor& in, Tensor* out,
                               Context *ctx) {
  auto sigmoid_lambda = [](posit_t a) {
    return posit_div(posit_one, posit_add(posit_one, posit_exp(posit_neg(a))));
  };
  traverse_unary<posit_t>(in, out, sigmoid_lambda);
}

template <>
void Sign<posit_t, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  auto sign_lambda = [](posit_t a) {
    return posit_sign(a);
  };
  traverse_unary<posit_t>(in, out, sign_lambda);
}

template <>
void Sqrt<posit_t, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  auto usqrt = [](posit_t a) {return posit_sqrt(a);};
  traverse_unary<posit_t>(in, out, usqrt);
}

template <>
void Sub<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2,
                           Tensor* out, Context *ctx) {
  // CHECK_EQ(ctx->stream, nullptr);
  auto sub_lambda_binary = [](posit_t a, posit_t b) {
    return posit_sub(a, b);
  };
  traverse_binary<posit_t>(in1, in2, out, sub_lambda_binary);
}

// sum all elements of input into out
// TODO(wangwei) optimize using omp
template <>
void Sum<posit_t, lang::Cpp>(const Tensor& in, posit_t *out,
                           Context *ctx) {
  posit_t s = posit_zero;
  const posit_t *inPtr = static_cast<const posit_t *>(in.block()->data());
  for (size_t i = 0; i < in.Size(); i++) {
    posit_add(s, inPtr[i]);
  }
  *out = s;
}

template <>
void Tanh<posit_t, lang::Cpp>(const Tensor& in, Tensor* out,
                            Context *ctx) {
  auto tanh_lambda = [](posit_t a) {
    return posit_tanh(a);
  };
  traverse_unary<posit_t>(in, out, tanh_lambda);
}

template <>
void Transform<posit_t, lang::Cpp>(const Tensor& in, Tensor* out,
                                 Context *ctx) {
  auto identity = [](posit_t a) {return a;};
  traverse_unary<posit_t>(in, out, identity);
}

template <>
void Bernoulli<posit_t, lang::Cpp>(const posit_t p, Tensor* out, Context *ctx) {
  float fp = posit_to_float(p);
  std::bernoulli_distribution distribution(fp);
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = distribution(ctx->random_generator) ? posit_one : posit_zero;
  }
}

// template <>
// void Gaussian<posit_t, lang::Cpp>(const posit_t mean, const posit_t std, Tensor* out, Context *ctx) { 
void Gaussian(const posit_t mean, const posit_t std, Tensor* out, Context *ctx) { 
  float fmean = posit_to_float(mean);
  float fstd = posit_to_float(std);
  std::normal_distribution<float> distribution(fmean, fstd);
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = static_cast<posit_t>(posit_from_float(distribution(ctx->random_generator)));
  }
}

// template <>
// void Uniform<posit_t, lang::Cpp>(const posit_t low, const posit_t high, Tensor* out, Context *ctx) {
void Uniform(const posit_t low, const posit_t high, Tensor* out, Context *ctx) {
  float flow = posit_to_float(low);
  float fhigh = posit_to_float(high);
  std::uniform_real_distribution<float> distribution(flow, fhigh);
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  for (size_t i = 0; i < out->Size(); i++) {
    outPtr[i] = static_cast<posit_t>(posit_from_float(distribution(ctx->random_generator)));
  }
}

// ====================Blas operations======================================

//warning, this function has block M overwritting to block M itself
template <>
void DGMM<posit_t, lang::Cpp>(const bool side_right,
                            const Tensor& M, const Tensor& v,
                            Tensor* out, Context *ctx) {
  const posit_t *MPtr = static_cast<const posit_t *>(M.block()->data());
  const posit_t *vPtr = static_cast<const posit_t *>(v.block()->data());
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  const size_t nrow = M.shape(0);
  const size_t ncol = M.shape(1);

  if (side_right) {
    for (size_t r = 0; r < nrow; r++) {
      size_t in_offset = M.stride()[0] * r, out_offset = out->stride()[0] * r;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[out_offset] = posit_mul(MPtr[in_offset], vPtr[c]);
        in_offset += M.stride()[1];
        out_offset += out->stride()[1];
      }
    }
  } else {
    for (size_t r = 0; r < nrow; r++) {
      size_t in_offset = M.stride()[0] * r, out_offset = out->stride()[0] * r;
      for (size_t c = 0; c < ncol; c++) {
        outPtr[out_offset] = posit_mul(MPtr[in_offset], vPtr[r]);
        in_offset += M.stride()[1];
        out_offset += out->stride()[1];
      }
    }
  }
}

template <>
void Amax<posit_t, lang::Cpp>(const Tensor& in, size_t *out,
                            Context *ctx) {
  const posit_t *inPtr = static_cast<const posit_t *>(in.block()->data());
  *out = posit_cblas_isamax(in.Size(), inPtr, 1); //not using strided traversal
}

template <>
void Asum<posit_t, lang::Cpp>(const Tensor& in, posit_t *out,
                            Context *ctx) {
  const posit_t *inPtr = static_cast<const posit_t *>(in.block()->data());
  *out = posit_cblas_sasum(in.Size(), inPtr, 1); //not using strided traversal
}

// template <>
// void Axpy<float, lang::Cpp>(const float alpha,
//                             const Tensor& in, Tensor *out, Context *ctx) {
//   //check input tensor for strides first
//   if (in.stride() == out->stride()) {
//     const float *inPtr = static_cast<const float *>(in.block()->data());
//     float *outPtr = static_cast<float *>(out->block()->mutable_data());
//     cblas_saxpy(in.Size(), alpha, inPtr, 1, outPtr, 1);
//   } else {
//     //LOG(FATAL) << "Axpy, input and output strides do not match." ;
//     EltwiseMult<float, lang::Cpp>(in, alpha, out, ctx);
//   }
// }

template <>
void Axpy<posit_t, lang::Cpp>(const posit_t alpha,
                            const Tensor& in, Tensor *out, Context *ctx) {
  //check input tensor for strides first
  const posit_t *inPtr = static_cast<const posit_t *>(in.block()->data());
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());

  if (in.stride() == out->stride()) {
    posit_cblas_saxpy(in.Size(), alpha, inPtr, 1, outPtr, 1);
  } else {
    //LOG(FATAL) << "Axpy, input and output strides do not match." ;
    Tensor t(in.shape(), in.device(), in.data_type());
    EltwiseMult<posit_t, lang::Cpp>(in, alpha, &t, ctx);
    posit_t* tPtr = static_cast<posit_t*>(t.block()->mutable_data());
    posit_cblas_saxpy(in.Size(), posit_one, tPtr, 1, outPtr, 1);
  }
}

// template <>
// void Axpy<float, lang::Cpp>(const float alpha,
//                            const Tensor& in, Tensor *out, Context *ctx) {
//  //check input tensor for strides first
//  if (in.stride() == out->stride()) {
//    const float *inPtr = static_cast<const float *>(in.block()->data());
//    float *outPtr = static_cast<float *>(out->block()->mutable_data());
//    cblas_saxpy(in.Size(), alpha, inPtr, 1, outPtr, 1);
//  } else if(out->transpose()) {
//    LOG(FATAL) << "output is already transposed." ;
//  } else {
//    LOG(FATAL) << "Axpy, input and output strides do not match." ;
//  }
// }

template <>
void Dot<posit_t, lang::Cpp>(const Tensor& in1, const Tensor& in2,
                           posit_t *out, Context *ctx) {
  //check input tensor for strides first
  if (!(in1.transpose()) && !(in2.transpose())) {
    const posit_t *in1Ptr = static_cast<const posit_t *>(in1.block()->data());
    const posit_t *in2Ptr = static_cast<const posit_t *>(in2.block()->data());
    *out = posit_cblas_sdot(in1.Size(), in1Ptr, 1, in2Ptr, 1);
  } else {
    LOG(FATAL) << "Dot, one of the input is tranposed. Not implemented yet." ;
  }
}

template <>
void Scale<posit_t, lang::Cpp>(const posit_t x, Tensor *out,
                             Context *ctx) {
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  posit_cblas_sscal(out->Size(), x, outPtr, 1); //not using strided traversal
}

template <>
void Nrm2<posit_t, lang::Cpp>(const Tensor& in, posit_t *out,
                            Context *ctx) {
  const posit_t *inPtr = static_cast<const posit_t *>(in.block()->data());
  *out = posit_cblas_snrm2(in.Size(), inPtr, 1); //not using strided traversal
}

template <>
void GEMV<posit_t, lang::Cpp>(const posit_t alpha, const Tensor& A, const Tensor& v,
                            const posit_t beta, Tensor *out, Context *ctx) {
  const posit_t *APtr = static_cast<const posit_t *>(A.block()->data());
  const posit_t *vPtr = static_cast<const posit_t *>(v.block()->data());
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  const size_t m = A.shape()[0];
  const size_t n = A.shape()[1];
  if (A.transpose()) {
    posit_cblas_sgemv(MyCblasRowMajor, MyCblasTrans, n, m, alpha, APtr, m, vPtr, 1, beta,
                    outPtr, 1);
  } else {
    posit_cblas_sgemv(MyCblasRowMajor, MyCblasNoTrans, m, n, alpha, APtr, n, vPtr, 1,
                    beta, outPtr, 1);
  }
}

template <>
void GEMM<posit_t, lang::Cpp>(const posit_t alpha,
                            const Tensor& A, const Tensor& B, const posit_t beta,
                            Tensor *C, Context *ctx) {
  auto transA = A.transpose();
  auto transa = transA ? MyCblasTrans : MyCblasNoTrans;
  auto transB = B.transpose();
  auto transb = transB ? MyCblasTrans : MyCblasNoTrans;
  const size_t nrowA = A.shape()[0];
  const size_t ncolA = A.shape()[1];
  const size_t ncolB = B.shape()[1];
  auto lda = transA ? nrowA : ncolA;
  auto ldb = transB ? ncolA : ncolB;
  auto ldc = ncolB;
  const posit_t *APtr = static_cast<const posit_t *>(A.block()->data());
  const posit_t *BPtr = static_cast<const posit_t *>(B.block()->data());
  posit_t *CPtr = static_cast<posit_t *>(C->block()->mutable_data());
  posit_cblas_sgemm(MyCblasRowMajor, transa, transb, nrowA, ncolB, ncolA, alpha, APtr,
                lda, BPtr, ldb, beta, CPtr, ldc);
}

template <>
void ComputeCrossEntropy<posit_t, lang::Cpp>(bool int_target,
    const size_t batchsize,
    const size_t dim, const Block *p,
    const Block *t, Block *loss,
    Context *ctx) {
  const posit_t *pPtr = static_cast<const posit_t *>(p->data());
  const int *tPtr = static_cast<const int *>(t->data());
  posit_t *lossPtr = static_cast<posit_t *>(loss->mutable_data());
  if (int_target) {
    for (size_t i = 0; i < batchsize; i++) {
      int truth_idx = tPtr[i];
      CHECK_GE(truth_idx, 0);
      posit_t prob_of_truth = pPtr[i * dim + truth_idx];
      lossPtr[i] = posit_neg(posit_log(posit_max(prob_of_truth, posit_minimum)));
    }
  } else {
    for (size_t i = 0; i < batchsize; i++) {
      posit_t sum = posit_zero;
      for (size_t j = 0; j < dim; j++) {
        sum = posit_add(sum, posit_from_int(tPtr[i * dim + j]));
      }
      posit_t loss = posit_zero;
      for (size_t j = 0, offset = i * dim; j < dim; j++, offset++) {
        loss = posit_sub(loss, posit_mul(posit_div(posit_from_int(tPtr[offset]), sum), posit_log(posit_max(pPtr[offset], posit_minimum))));
      }
      lossPtr[i] = loss;
    }
  }
}

template <>
void SoftmaxCrossEntropyBwd<posit_t, lang::Cpp>(bool int_target,
    const size_t batchsize,
    const size_t dim, const Block *p,
    const Block *t, Block *grad,
    Context *ctx) {
  CHECK_EQ(p, grad) << "Use the same pointer to optimize performance";  
  const int *tPtr = static_cast<const int *>(t->data());
  posit_t *gradPtr = static_cast<posit_t *>(grad->mutable_data());

  if (int_target) {
    for (size_t i = 0; i < batchsize; i++) {
      int truth_idx = static_cast<int>(tPtr[i]);
      CHECK_GE(truth_idx, 0);
      gradPtr[i * dim + truth_idx] = posit_sub(gradPtr[i * dim + truth_idx], posit_one);
    }
  } else {
    for (size_t i = 0; i < batchsize; i++) {
      posit_t sum = posit_zero;
      for (size_t j = 0; j < dim; j++) {
        sum = posit_add(sum, posit_from_int(tPtr[i * dim + j]));
      }
      for (size_t j = 0, offset = i * dim; j < dim; j++, offset++) {
        gradPtr[offset] = posit_sub(gradPtr[offset], posit_div(posit_from_int(tPtr[offset]), sum));
      }
    }
  }
}

template <>
void RowMax<posit_t, lang::Cpp>(const Tensor& in, Tensor *out, Context *ctx) {
  const posit_t *inPtr = static_cast<const posit_t *>(in.block()->data());
  posit_t *outPtr = static_cast<posit_t *>(out->block()->mutable_data());
  const size_t nrow = in.shape()[0];
  const size_t ncol = in.shape()[1];
  vector<int> traversal_info = generate_traversal_info(in);
  vector<int> shape_multipliers = generate_shape_multipliers(in);

  for (size_t r = 0; r < nrow; r++) {
    int counter_offset = (r * ncol);
    posit_t maxval = posit_zero;
    for (size_t c = 0; c < ncol; c++) {
      maxval = posit_max(maxval, inPtr[traversal_info[in.shape().size()]]);
      traverse_next(in, shape_multipliers, traversal_info, counter_offset + c + 1);
    }
    outPtr[r] = maxval;
  }
}

} // end

#endif  // SINGA_CORE_TENSOR_TENSOR_MATH_POSIT_CPP_H_