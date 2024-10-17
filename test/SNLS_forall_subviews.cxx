#include <cstdlib>
#include <iostream>
#include <random>

#include "SNLS_config.h"
#include "SNLS_kernels.h"
#if defined(SNLS_RAJA_ONLY) || defined(SNLS_RAJA_PORT_SUITE)
#include "RAJA/RAJA.hpp"
#include "SNLS_device_forall.h"
#include "SNLS_view_types.h"

int main() {

    const int npts = SNLS_GPU_BLOCKS * 4;
    const int row  = 3;

    //
    // Allocate and initialize vector data
    //
    RAJA::resources::Host host{};
#if defined(RAJA_ENABLE_CUDA)
    RAJA::resources::Cuda res_gpu{};
#elif defined(RAJA_ENABLE_HIP)
    RAJA::resources::Hip res_gpu{};
#elif defined(RAJA_ENABLE_SYCL)
    RAJA::resources::Sycl res_gpu{};
#endif

#if defined(__snls_gpu_active__)
    auto x   = res_gpu.allocate<double>(npts * row);
    auto x1d = res_gpu.allocate<double>(npts);
#else
    auto x   = host.allocate<double>(npts * row);
    auto x1d = host.allocate<double>(npts);
#endif

#if defined(__snls_gpu_active__)
    constexpr auto EXEC_STRAT = snls::ExecutionStrategy::GPU;
#else
    constexpr auto EXEC_STRAT = snls::ExecutionStrategy::CPU;
#endif
    snls::Device::GetInstance().SetBackend(EXEC_STRAT);

    snls::forall_strat<SNLS_GPU_BLOCKS>(0, npts, EXEC_STRAT, [=] __snls_hdev__(int i) {
        for (int j = 0; j < row; j++) {
            x[i * row + j] = 0;
        }
        x1d[i] = 0;
    });

    const snls::rview2d v2d(x, npts, row);
    const snls::rview1d v1d(x1d, npts);

    auto res = snls::Device::GetInstance().GetDefaultRAJAResource();

    auto event = snls::forall<SNLS_GPU_BLOCKS, true>(0, npts, res, [=] __snls_hdev__ (int i) {

    #if defined(__snls_device_only__)
        int xindex = threadIdx.x + blockIdx.x * blockDim.x;
        int yindex = threadIdx.y + blockIdx.y * blockDim.y;
        const int bindex = blockIdx.x;
        const int global_index = xindex + (gridDim.x * gridDim.y * yindex);
        const int thread_id = threadIdx.x;
    #else
        const int bindex = 0;
        const int global_index = i;
        const int thread_id = -10;
    #endif
        // We can still modify the internal data but the class itself is constant
        // which should hopefully let the compiler optimize more...
        snls::SubView sv(global_index, v2d);
        const snls::SubView svp(global_index, v2d);
        sv.set_offset(0);
        const snls::SubView sv1d(global_index, v1d);
        sv1d() = round(double(bindex));
        // This is a bit messed up though that even though the class and internal
        // member variable are constants we can still modify things...
        sv(0) = round(double(bindex));
        svp(1) = round(double(thread_id));
        sv(2) = round(double(bindex));
        double& ref = sv1d();
        ref = -1;

        // Testing that the assingment operator works on the GPU as well
        {
            auto v2ds0p = snls::SubView(i, 0, v2d);
            auto v2ds1p = snls::SubView(i, 1, v2d);
            snls::snls_swap(v2ds0p, v2ds1p);
        }
        // cool thing is this will fail to compile
        // sv(1, 2) = double(thread_id);
    });

    // Add a variation to snls::Device for how to do this...
    snls::Device::GetInstance().WaitFor(res, &event);

    snls::Device::GetInstance().SetBackend(snls::ExecutionStrategy::CPU);

    auto x_host   = host.allocate<double>(npts * row);
    auto x1d_host = host.allocate<double>(npts);

    const size_t nrows = 2;
    const size_t ncols = 2;


    auto a1 = host.allocate<double>(npts * nrows * ncols);
    auto a2 = host.allocate<double>(npts * nrows * ncols);
    auto a3 = host.allocate<double>(npts * nrows * ncols);

    for (size_t i = 0; i < (npts * nrows * ncols); i++) {
        a1[i] = double(i);
        a2[i] = double(2.0 * i);
        a3[i] = double(3.0 * i);
    }

    double* myarr[3] = {a1, a2, a3};
    // We need to make sure things are assigned like this to make sure things iterate the fastest.
    // left to the right is fastest but # of arrays and npts would swap places
    // given we're doing a SOA type thing here
    RAJA::MultiView<double, RAJA::Layout<3>> mview(myarr, npts, nrows, ncols);

#if defined(__snls_gpu_active__)
    res_gpu.memcpy(x_host, x, sizeof(double) * npts * row);
    res_gpu.memcpy(x1d_host, x1d, sizeof(double) * npts);
    res_gpu.wait();
#else
    host.memcpy(x_host, x, sizeof(double) * npts * row);
    host.memcpy(x1d_host, x1d, sizeof(double) * npts);
#endif

    snls::crview2d v2d_cpu(x_host, npts, row);
    snls::crview1d v1d_cpu(x1d_host, npts);

    RAJA::ReduceBitAnd<RAJA::seq_reduce, bool> output(true);

    snls::forall<SNLS_GPU_BLOCKS>(0, npts, [=] __snls_hdev__ (int i) {
        const snls::SubView v2ds(i, 0, v2d_cpu);
        const snls::SubView v2dsds(0, v2ds);
        const snls::SubView v2ds2(i, 1, v2d_cpu);

        auto v1ds = snls::SubView(i, v1d_cpu);

        auto v2ds0p = snls::SubView(i, 0, v2d_cpu);
        auto v2ds1p = snls::SubView(i, 1, v2d_cpu);

        snls::snls_swap(v2ds0p, v2ds1p);

        const snls::SubView mview_sv(i, mview);

        std::ignore = v2ds.get_data();
        std::ignore = mview_sv.get_data();
#ifdef __snls_host_only__
        // Our values are offset so check and make sure they're the same value
        output &= (v2ds(1) == v2ds2(0));

        const double val1 = double(i * nrows * ncols);
        const double val2 = double(2.0 * i * nrows * ncols) + 2.0;
        const double val3 = double(3.0 * i * nrows * ncols) + 9.0;

        output &= (mview_sv(0, 0, 0) == val1);
        output &= (mview_sv(1, 0, 1) == val2);
        output &= (mview_sv(2, 1, 1) == val3);

        // These next two cases deal with the fact that we swapped the views so
        // the offsets are now reversed from what they used to be
        output &= (v2ds(1) == v2ds0p(0));
        output &= (v2ds(0) == v2ds1p(0));
        // Finally checking to make sure that the SubView and view values are the same
        output &= (v1d_cpu(i) == v1ds());
        output &= snls::contains_data(v2ds);
        output &= snls::contains_data(v2dsds);
#endif
    });

#if defined(__snls_gpu_active__)
    res_gpu.deallocate(x);
    res_gpu.deallocate(x1d);
#else
    host.deallocate(x);
    host.deallocate(x1d);
#endif

    host.deallocate(x_host);
    host.deallocate(x1d_host);
    host.deallocate(a1);
    host.deallocate(a2);
    host.deallocate(a3);
    std::cout << "Test pass status: " << output.get() << std::endl;
    return !output.get();
}
#endif