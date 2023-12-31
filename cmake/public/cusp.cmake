set(cusp_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(cusp_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
add_pkg(cusp
        VER 1.0
        HEAD_ONLY ./
        URL https://github.com/cusplibrary/cusplibrary/archive/refs/tags/v0.4.0.tar.gz
        MD5 41d8342631256935dcfb063d281b7dd4)
message("cusp_INC + ${cusp_INC}")
include_directories( ${cusp_INC})
add_library(spmm::cusp ALIAS cusp)


