#include <cstdlib>
#include <iostream>
#include <random>

#include "SNLS_config.h"
#include "SNLS_kernels.h"
#if defined(SNLS_RAJA_PORT_SUITE)
#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"
#include "SNLS_device_forall.h"
#include "SNLS_view_types.h"
#include "SNLS_memory_manager.h"

int main() {

    const int npts = SNLS_GPU_BLOCKS * 4;
    const int row  = 3;
    auto mm = snls::memoryManager::getInstance();
    auto x = mm.allocManagedArray<double>(npts * row);
    auto x1d = mm.allocManagedArray<double>(npts);

#if defined(__snls_gpu_active__)
    constexpr auto EXEC_STRAT = snls::ExecutionStrategy::GPU;
#else
    constexpr auto EXEC_STRAT = snls::ExecutionStrategy::CPU;
#endif
    snls::Device::GetInstance().SetBackend(EXEC_STRAT);

    snls::forall_strat<SNLS_GPU_BLOCKS>(0, npts, snls::ExecutionStrategy::CPU, [=] __snls_hdev__(int i) {
        for (int j = 0; j < row; j++) {
            x[i * row + j] = 0;
        }
        x1d[i] = 0;
    });

    const snls::rview2d v2d(x.data(snls::Device::GetInstance().GetCHAIES()), npts, row);
    const snls::rview1d v1d(x1d.data(snls::Device::GetInstance().GetCHAIES()), npts);

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

    snls::crview2d v2d_cpu(x.data(chai::ExecutionSpace::CPU), npts, row);
    snls::crview1d v1d_cpu(x1d.data(chai::ExecutionSpace::CPU), npts);

    RAJA::ReduceBitAnd<RAJA::seq_reduce, bool> output(true);

    snls::forall<SNLS_GPU_BLOCKS>(0, npts, [=] __snls_hdev__ (int i) {
        const snls::SubView v2ds(i, 0, v2d_cpu);
        const snls::SubView v2ds2(i, 1, v2d_cpu);

        auto v1ds = snls::SubView(i, v1d_cpu);

        auto v2ds0p = snls::SubView(i, 0, v2d_cpu);
        auto v2ds1p = snls::SubView(i, 1, v2d_cpu);

        snls::snls_swap(v2ds0p, v2ds1p);

#ifdef __snls_host_only__
        // Our values are offset so check and make sure they're the same value
        output &= (v2ds(1) == v2ds2(0));
        // These next two cases deal with the fact that we swapped the views so
        // the offsets are now reversed from what they used to be
        output &= (v2ds(1) == v2ds0p(0));
        output &= (v2ds(0) == v2ds1p(0));
        // Finally checking to make sure that the SubView and view values are the same
        output &= (v1d_cpu(i) == v1ds());
#endif
    });

    std::cout << "Test pass status: " << output.get() << std::endl;
    return !output.get();
}
#endif
