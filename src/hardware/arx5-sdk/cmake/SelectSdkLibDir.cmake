include_guard(GLOBAL)

function(arx5_resolve_lib_dir repo_root out_var)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|arm|aarch64")
        set(_source_dir "${repo_root}/lib/aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
        set(_source_dir "${repo_root}/lib/x86_64")
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(_hardware_src "${_source_dir}/libhardware.so")
    set(_solver_src "${_source_dir}/libsolver.so")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(^i686)")
        execute_process(
            COMMAND getconf GNU_LIBC_VERSION
            OUTPUT_VARIABLE _glibc_version_raw
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "[0-9]+\\.[0-9]+" _glibc_version "${_glibc_version_raw}")

        if(
            _glibc_version
            AND _glibc_version VERSION_LESS "2.34"
            AND EXISTS "${_source_dir}/libhardware_20_04.so"
            AND EXISTS "${_source_dir}/libsolver_20_04.so"
        )
            message(STATUS "Using Ubuntu 20.04-compatible SDK libraries for glibc ${_glibc_version}")
            set(_hardware_src "${_source_dir}/libhardware_20_04.so")
            set(_solver_src "${_source_dir}/libsolver_20_04.so")
        endif()
    endif()

    set(_resolved_dir "${CMAKE_BINARY_DIR}/resolved_libs")
    file(MAKE_DIRECTORY "${_resolved_dir}")
    configure_file("${_hardware_src}" "${_resolved_dir}/libhardware.so" COPYONLY)
    configure_file("${_solver_src}" "${_resolved_dir}/libsolver.so" COPYONLY)

    set(${out_var} "${_resolved_dir}" PARENT_SCOPE)
endfunction()
