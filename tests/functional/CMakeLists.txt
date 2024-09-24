rtneural_add_test(
    TARGET rtneural_test_functional
    SOURCES
        bad_model_test.cpp
        conv2d_model_test.cpp
        model_test.cpp
        sample_rate_rnn_test.cpp
        templated_tests.cpp
        torch_conv1d_test.cpp
        torch_conv1d_groups_test.cpp
        torch_lstm_test.cpp
        torch_gru_test.cpp
        torch_microtcn_test.cpp
        torch_convtranspose1d_test.cpp
        torch_conv1d_stride_test.cpp
    DEPENDENCIES PRIVATE RTNeural)
