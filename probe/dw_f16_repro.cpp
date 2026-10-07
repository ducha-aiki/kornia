// Pure-oneDNN reproducer for kornia/kornia#5493: creating an f16 depthwise 1xK forward convolution primitive
// spins forever in jit_brdgmm_kernel_base_t::generate() on AVX512-FP16 CPUs (seen from PyTorch 2.9.1 / 2.14.0).
// usage: dw_f16_repro [train|infer] [plain|any] [f16|bf16|f32] [kw=17] [ow=96] [groups=3] [ih=96]
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

int main(int argc, char **argv) {
    const bool infer = argc > 1 && !std::strcmp(argv[1], "infer");
    const bool any = argc > 2 && !std::strcmp(argv[2], "any");
    memory::data_type dt = memory::data_type::f16;
    const char *dt_name = argc > 3 ? argv[3] : "f16";
    if (!std::strcmp(dt_name, "bf16")) dt = memory::data_type::bf16;
    if (!std::strcmp(dt_name, "f32")) dt = memory::data_type::f32;
    const memory::dim kw = argc > 4 ? std::atoi(argv[4]) : 17;
    const memory::dim ow = argc > 5 ? std::atoi(argv[5]) : 96;
    const memory::dim g = argc > 6 ? std::atoi(argv[6]) : 3;
    const memory::dim ih = argc > 7 ? std::atoi(argv[7]) : 96;
    const memory::dim iw = ow + kw - 1;

    const dnnl_version_t *v = dnnl_version();
    std::printf("oneDNN v%d.%d.%d (%s) prop=%s fmt=%s dt=%s g%ld mb1 ic%ld ih%ld iw%ld oc%ld oh%ld ow%ld kh1 kw%ld\n",
            v->major, v->minor, v->patch, v->hash, infer ? "forward_inference" : "forward_training",
            any ? "any" : "nchw/goihw", dt_name, (long)g, (long)g, (long)ih, (long)iw, (long)g, (long)ih, (long)ow,
            (long)kw);
    std::fflush(stdout);

    try {
        engine eng(engine::kind::cpu, 0);
        const auto act_tag = any ? memory::format_tag::any : memory::format_tag::nchw;
        const auto wei_tag = any ? memory::format_tag::any : memory::format_tag::goihw;
        memory::desc src({1, g, ih, iw}, dt, act_tag);
        memory::desc wei({g, 1, 1, 1, kw}, dt, wei_tag);
        memory::desc dst({1, g, ih, ow}, dt, act_tag);
        auto pd = convolution_forward::primitive_desc(eng,
                infer ? prop_kind::forward_inference : prop_kind::forward_training, algorithm::convolution_direct, src,
                wei, dst, {1, 1}, {0, 0}, {0, 0});
        std::printf("primitive_desc ok, impl: %s\n", pd.impl_info_str());
        std::fflush(stdout);

        auto prim = convolution_forward(pd);
        std::printf("primitive created\n");
        std::fflush(stdout);

        memory s(pd.src_desc(), eng), w(pd.weights_desc(), eng), d(pd.dst_desc(), eng);
        stream st(eng);
        prim.execute(st, {{DNNL_ARG_SRC, s}, {DNNL_ARG_WEIGHTS, w}, {DNNL_ARG_DST, d}});
        st.wait();
        std::printf("executed\n");
    } catch (const dnnl::error &e) {
        std::printf("dnnl::error: %s\n", e.what());
        return 2;
    }
    return 0;
}
