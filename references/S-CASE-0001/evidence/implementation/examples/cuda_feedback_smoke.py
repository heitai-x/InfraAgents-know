"""Tiny CUDA Driver API correctness/profile fixture, not a model benchmark.

Uses the installed NVIDIA driver; no CUDA toolkit or Python package is required.
Allocates 16 KiB and adds one to 4096 integers three times. Source code belongs
in engineering artifacts, never in the non-code knowledge base.
"""

import ctypes as ct
import json


def main() -> None:
    cuda = ct.CDLL("libcuda.so.1")

    def api(name, types, *arguments):
        function = getattr(cuda, name)
        function.argtypes = types
        function.restype = ct.c_int
        result = function(*arguments)
        if result:
            raise RuntimeError(f"{name} returned CUDA error {result}")

    pointer = ct.c_void_p
    device = ct.c_int()
    context = pointer()
    allocation = ct.c_uint64()
    module = pointer()
    count = 4096
    values = (ct.c_uint32 * count)(*range(count))
    ptx = b""".version 7.0
.target sm_70
.address_size 64
.visible .entry infra_add_one(.param .u64 data) {
 .reg .u32 %r<5>;
 .reg .u64 %rd<4>;
 ld.param.u64 %rd1, [data];
 mov.u32 %r1, %ctaid.x;
 mov.u32 %r2, %ntid.x;
 mov.u32 %r3, %tid.x;
 mad.lo.u32 %r4, %r1, %r2, %r3;
 mul.wide.u32 %rd2, %r4, 4;
 add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r1, [%rd3];
 add.u32 %r1, %r1, 1;
 st.global.u32 [%rd3], %r1;
 ret;
}
"""
    api("cuInit", [ct.c_uint], 0)
    api("cuDeviceGet", [pointer, ct.c_int], ct.byref(device), 0)
    api("cuCtxCreate_v2", [pointer, ct.c_uint, ct.c_int], ct.byref(context), 0, device)
    try:
        api("cuMemAlloc_v2", [pointer, ct.c_size_t], ct.byref(allocation), ct.sizeof(values))
        api("cuMemcpyHtoD_v2", [ct.c_uint64, pointer, ct.c_size_t], allocation, values, ct.sizeof(values))
        api("cuModuleLoadData", [pointer, ct.c_char_p], ct.byref(module), ptx)
        function = pointer()
        api("cuModuleGetFunction", [pointer, pointer, ct.c_char_p], ct.byref(function), module, b"infra_add_one")
        arguments = (pointer * 1)(ct.cast(ct.byref(allocation), pointer))
        for _ in range(3):
            api("cuLaunchKernel", [pointer, *([ct.c_uint] * 7), pointer, pointer, pointer],
                function, 16, 1, 1, 256, 1, 1, 0, None, arguments, None)
        api("cuCtxSynchronize", [])
        api("cuMemcpyDtoH_v2", [pointer, ct.c_uint64, ct.c_size_t], values, allocation, ct.sizeof(values))
        if any(value != index + 3 for index, value in enumerate(values)):
            raise AssertionError("CUDA smoke output differs from integer reference")
        print(json.dumps({"fixture": "cuda_feedback_smoke", "elements": count,
                          "kernel_launches": 3, "correctness": "passed",
                          "model_benchmark": False}))
    finally:
        if module:
            api("cuModuleUnload", [pointer], module)
        if allocation.value:
            api("cuMemFree_v2", [ct.c_uint64], allocation)
        api("cuCtxDestroy_v2", [pointer], context)


if __name__ == "__main__":
    main()
