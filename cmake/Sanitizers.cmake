option(RTNEURAL_ENABLE_RADSAN "Enable RealtimeSanitizer (RADSan) checks (requires RADSan clang)" OFF)

function(rtneural_radsan_configure target)
    target_compile_definitions(${target} PUBLIC RTNEURAL_RADSAN_ENABLED)
    target_compile_options(${target} PUBLIC -fsanitize=realtime)
    target_link_options(${target} PUBLIC -fsanitize=realtime)
endfunction()
