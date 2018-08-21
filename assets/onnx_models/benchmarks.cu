
namespace conv__7605911250412906459
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__7605911250412906459(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 3},        /* Input1 */
      {"input[2]", 224},      /* Input2 */
      {"input[3]", 224},      /* Input3 */
      {"filter_count", 64},   /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__7605911250412906459(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__7605911250412906459(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__7605911250412906459(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__7605911250412906459(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__7605911250412906459(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__7605911250412906459(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__7605911250412906459(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__7605911250412906459(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__7605911250412906459(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__7605911250412906459(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__7605911250412906459(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__7605911250412906459(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      3,   /* Input1 */         \
      224, /* Input2 */         \
      224, /* Input3 */         \
      64,  /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__7605911250412906459); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__7605911250412906459); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__7605911250412906459);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__7605911250412906459);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__7605911250412906459);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__7605911250412906459);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__7605911250412906459

namespace conv__14198308928769061651
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__14198308928769061651(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 64},       /* Input1 */
      {"input[2]", 224},      /* Input2 */
      {"input[3]", 224},      /* Input3 */
      {"filter_count", 64},   /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__14198308928769061651(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__14198308928769061651(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__14198308928769061651(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__14198308928769061651(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__14198308928769061651(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__14198308928769061651(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__14198308928769061651(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__14198308928769061651(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__14198308928769061651(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__14198308928769061651(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__14198308928769061651(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__14198308928769061651(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      64,  /* Input1 */         \
      224, /* Input2 */         \
      224, /* Input3 */         \
      64,  /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__14198308928769061651); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__14198308928769061651); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__14198308928769061651);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__14198308928769061651);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__14198308928769061651);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__14198308928769061651);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__14198308928769061651

namespace conv__16626576756657293610
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__16626576756657293610(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 64},       /* Input1 */
      {"input[2]", 112},      /* Input2 */
      {"input[3]", 112},      /* Input3 */
      {"filter_count", 128},  /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__16626576756657293610(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__16626576756657293610(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__16626576756657293610(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__16626576756657293610(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__16626576756657293610(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__16626576756657293610(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__16626576756657293610(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__16626576756657293610(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__16626576756657293610(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__16626576756657293610(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__16626576756657293610(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__16626576756657293610(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      64,  /* Input1 */         \
      112, /* Input2 */         \
      112, /* Input3 */         \
      128, /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__16626576756657293610); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__16626576756657293610); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__16626576756657293610);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__16626576756657293610);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__16626576756657293610);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__16626576756657293610);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__16626576756657293610

namespace conv__13337270216424104043
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__13337270216424104043(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 128},      /* Input1 */
      {"input[2]", 112},      /* Input2 */
      {"input[3]", 112},      /* Input3 */
      {"filter_count", 128},  /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__13337270216424104043(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__13337270216424104043(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__13337270216424104043(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__13337270216424104043(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__13337270216424104043(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__13337270216424104043(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__13337270216424104043(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__13337270216424104043(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__13337270216424104043(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__13337270216424104043(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__13337270216424104043(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__13337270216424104043(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      128, /* Input1 */         \
      112, /* Input2 */         \
      112, /* Input3 */         \
      128, /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__13337270216424104043); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__13337270216424104043); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__13337270216424104043);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__13337270216424104043);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__13337270216424104043);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__13337270216424104043);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__13337270216424104043

namespace conv__3280367009960007047
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3280367009960007047(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 128},      /* Input1 */
      {"input[2]", 56},       /* Input2 */
      {"input[3]", 56},       /* Input3 */
      {"filter_count", 256},  /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__3280367009960007047(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3280367009960007047(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__3280367009960007047(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3280367009960007047(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__3280367009960007047(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3280367009960007047(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__3280367009960007047(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3280367009960007047(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__3280367009960007047(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3280367009960007047(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__3280367009960007047(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3280367009960007047(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      128, /* Input1 */         \
      56,  /* Input2 */         \
      56,  /* Input3 */         \
      256, /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__3280367009960007047); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__3280367009960007047); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__3280367009960007047);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__3280367009960007047);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__3280367009960007047);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__3280367009960007047);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__3280367009960007047

namespace conv__18108158721138750823
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__18108158721138750823(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 256},      /* Input1 */
      {"input[2]", 56},       /* Input2 */
      {"input[3]", 56},       /* Input3 */
      {"filter_count", 256},  /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__18108158721138750823(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__18108158721138750823(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__18108158721138750823(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__18108158721138750823(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__18108158721138750823(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__18108158721138750823(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__18108158721138750823(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__18108158721138750823(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__18108158721138750823(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__18108158721138750823(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__18108158721138750823(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__18108158721138750823(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      256, /* Input1 */         \
      56,  /* Input2 */         \
      56,  /* Input3 */         \
      256, /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__18108158721138750823); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__18108158721138750823); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__18108158721138750823);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__18108158721138750823);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__18108158721138750823);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__18108158721138750823);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__18108158721138750823

namespace conv__2926869540711932357
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__2926869540711932357(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 256},      /* Input1 */
      {"input[2]", 28},       /* Input2 */
      {"input[3]", 28},       /* Input3 */
      {"filter_count", 512},  /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__2926869540711932357(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__2926869540711932357(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__2926869540711932357(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__2926869540711932357(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__2926869540711932357(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__2926869540711932357(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__2926869540711932357(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__2926869540711932357(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__2926869540711932357(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__2926869540711932357(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__2926869540711932357(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__2926869540711932357(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      256, /* Input1 */         \
      28,  /* Input2 */         \
      28,  /* Input3 */         \
      512, /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__2926869540711932357); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__2926869540711932357); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__2926869540711932357);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__2926869540711932357);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__2926869540711932357);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__2926869540711932357);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__2926869540711932357

namespace conv__4406346575509718427
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__4406346575509718427(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 512},      /* Input1 */
      {"input[2]", 28},       /* Input2 */
      {"input[3]", 28},       /* Input3 */
      {"filter_count", 512},  /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__4406346575509718427(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__4406346575509718427(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__4406346575509718427(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__4406346575509718427(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__4406346575509718427(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__4406346575509718427(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__4406346575509718427(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__4406346575509718427(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__4406346575509718427(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__4406346575509718427(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__4406346575509718427(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__4406346575509718427(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      512, /* Input1 */         \
      28,  /* Input2 */         \
      28,  /* Input3 */         \
      512, /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__4406346575509718427); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__4406346575509718427); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__4406346575509718427);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__4406346575509718427);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__4406346575509718427);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__4406346575509718427);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__4406346575509718427

namespace conv__3955600878335767527
{

static void LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3955600878335767527(benchmark::State &state)
{
  state.counters.insert({
      {"input[0]", 1},        /* Input0 */
      {"input[1]", 512},      /* Input1 */
      {"input[2]", 14},       /* Input2 */
      {"input[3]", 14},       /* Input3 */
      {"filter_count", 512},  /* FilterCount */
      {"filter_height", 3},   /* FilterHeight */
      {"filter_width", 3},    /* FilterWidth */
      {"pad_height", 1},      /* PadHeight */
      {"pad_width", 1},       /* PadWidth */
      {"stride_height", 1},   /* StrideHeight */
      {"stride_width", 1},    /* StrideWidth */
      {"dilation_height", 1}, /* DilationHeight */
      {"dilation_width", 1}   /* DilationWidth */
  });
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT8__3955600878335767527(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int8_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3955600878335767527(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_INT32__3955600878335767527(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<int32_t, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3955600878335767527(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF__3955600878335767527(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3955600878335767527(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__3955600878335767527(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3955600878335767527(state);
}
#endif

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_FLOAT__3955600878335767527(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<float, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3955600878335767527(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FWD_DOUBLE__3955600878335767527(benchmark::State &state)
{
  CUDNN_CONV_FWD_Impl<double, convolution_algorithm>(state);
  LAYER_CUDNN_CONV_FWD_ADD_COUNTERS__3955600878335767527(state);
}

#define InputArgs()             \
  Args({{                       \
      1,   /* Input0 */         \
      512, /* Input1 */         \
      14,  /* Input2 */         \
      14,  /* Input3 */         \
      512, /* FilterCount */    \
      3,   /* FilterHeight */   \
      3,   /* FilterWidth */    \
      1,   /* PadHeight */      \
      1,   /* PadWidth */       \
      1,   /* StrideHeight */   \
      1,   /* StrideWidth */    \
      1,   /* DilationHeight */ \
      1    /* DilationWidth */  \
  }})

#define BENCHMARK_LAYER_CUDNN_CONV_FWD(b)                                                                                                                                                                                                                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();         \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime(); \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->ArgNames({"input[0]", "input[1]", "input[2]", "input[3]", "filter_count", "filter_height", "filter_width", "pad_height", "pad_width", "stride_height", "stride_width", "dilation_height", "dilation_width"})->InputArgs()->UseManualTime()

/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT8__3955600878335767527); */
/* BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_INT32__3955600878335767527); */
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF__3955600878335767527);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_HALF_TENSOROP__3955600878335767527);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_FLOAT__3955600878335767527);
BENCHMARK_LAYER_CUDNN_CONV_FWD(LAYER_CUDNN_CONV_FWD_DOUBLE__3955600878335767527);

#undef InputArgs
#undef BENCHMARK_LAYER_CUDNN_CONV_FWD
} // namespace conv__3955600878335767527
