#include "mnn/backend.hpp"
namespace mnn {
static Backend* g_backend = nullptr;
Backend& backend() { if (!g_backend) g_backend = &cpu_backend(); return *g_backend; }
void set_backend(Backend& b) { g_backend = &b; }
bool use_backend(const std::string& name) {
    if (name == "cpu") { set_backend(cpu_backend()); return true; }
#ifdef MNN_WITH_CUDA
    if (name == "cuda") { set_backend(cuda_backend()); return true; }
#endif
    return false;
}
}  // namespace mnn
