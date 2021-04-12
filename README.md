WORK IN PROGRESS HERE


    git submodule update --init --recursive
    cd fastaugment
    mkdir -p build && cd build
    cmake .. && make
    cd ../../sigmoid_like_tf_op
    mkdir -p build && cd build
    cmake .. && make