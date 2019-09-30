|  LAYERNAME  |   LAYERTYPE   |         INPUTNAMES          |    OUTPUTNAMES    |             INPUTSHAPES             |  OUTPUTSHAPES   |
|-------------|---------------|-----------------------------|-------------------|-------------------------------------|-----------------|
| data_0      | ConstantInput |                             |                   | [[1,3,224,224]]                     | [[1,3,224,224]] |
| conv1_w_0   | ConstantInput |                             |                   | [[96,3,11,11]]                      | [[96,3,11,11]]  |
| conv1_b_0   | ConstantInput |                             |                   | [[96]]                              | [[96]]          |
| conv_1      | Conv          | data_0;conv1_w_0;conv1_b_0  | conv1_1           | [[1,3,224,224],[96,3,11,11],[96]]   | [[1,96,54,54]]  |
| relu_1      | Relu          | conv1_1                     | conv1_2           | [[1,96,54,54]]                      | [[1,96,54,54]]  |
| maxpool_1   | Pooling       | conv1_2                     | pool1_1           | [[1,96,54,54]]                      | [[1,96,27,27]]  |
| lrn_1       | LRN           | pool1_1                     | norm1_1           | [[1,96,27,27]]                      | [[1,96,27,27]]  |
| conv2_w_0   | ConstantInput |                             |                   | [[256,48,5,5]]                      | [[256,48,5,5]]  |
| conv2_b_0   | ConstantInput |                             |                   | [[256]]                             | [[256]]         |
| conv2_2     | Conv          | norm1_1;conv2_w_0;conv2_b_0 | conv2_1           | [[1,96,27,27],[256,48,5,5],[256]]   | [[1,256,27,27]] |
| relu_2      | Relu          | conv2_1                     | conv2_2           | [[1,256,27,27]]                     | [[1,256,27,27]] |
| maxpool_2   | Pooling       | conv2_2                     | pool2_1           | [[1,256,27,27]]                     | [[1,256,13,13]] |
| lrn_2       | LRN           | pool2_1                     | norm2_1           | [[1,256,13,13]]                     | [[1,256,13,13]] |
| conv3_w_0   | ConstantInput |                             |                   | [[384,256,3,3]]                     | [[384,256,3,3]] |
| conv3_b_0   | ConstantInput |                             |                   | [[384]]                             | [[384]]         |
| conv_3      | Conv          | norm2_1;conv3_w_0;conv3_b_0 | conv3_1           | [[1,256,13,13],[384,256,3,3],[384]] | [[1,384,13,13]] |
| relu_3      | Relu          | conv3_1                     | conv3_2           | [[1,384,13,13]]                     | [[1,384,13,13]] |
| conv4_w_0   | ConstantInput |                             |                   | [[384,192,3,3]]                     | [[384,192,3,3]] |
| conv4_b_0   | ConstantInput |                             |                   | [[384]]                             | [[384]]         |
| conv4_2     | Conv          | conv3_2;conv4_w_0;conv4_b_0 | conv4_1           | [[1,384,13,13],[384,192,3,3],[384]] | [[1,384,13,13]] |
| relu_4      | Relu          | conv4_1                     | conv4_2           | [[1,384,13,13]]                     | [[1,384,13,13]] |
| conv5_w_0   | ConstantInput |                             |                   | [[256,192,3,3]]                     | [[256,192,3,3]] |
| conv5_b_0   | ConstantInput |                             |                   | [[256]]                             | [[256]]         |
| conv5_2     | Conv          | conv4_2;conv5_w_0;conv5_b_0 | conv5_1           | [[1,384,13,13],[256,192,3,3],[256]] | [[1,256,13,13]] |
| relu_5      | Relu          | conv5_1                     | conv5_2           | [[1,256,13,13]]                     | [[1,256,13,13]] |
| maxpool_3   | Pooling       | conv5_2                     | pool5_1           | [[1,256,13,13]]                     | [[1,256,6,6]]   |
| OC2_DUMMY_1 | ConstantInput |                             |                   | [[1,9216]]                          | [[1,9216]]      |
| reshape_1   | Reshape       | pool5_1;OC2_DUMMY_1         | OC2_DUMMY_0       | [[1,256,6,6],[1,9216]]              | [[1,9216]]      |
| fc6_w_0     | ConstantInput |                             |                   | [[4096,9216]]                       | [[4096,9216]]   |
| fc6_b_0     | ConstantInput |                             |                   | [[4096]]                            | [[4096]]        |
| gemm_1      | Gemm          | OC2_DUMMY_0;fc6_w_0;fc6_b_0 | fc6_1             | [[1,9216],[4096,9216],[4096]]       | [[1,4096]]      |
| relu_6      | Relu          | fc6_1                       | fc6_2             | [[1,4096]]                          | [[1,4096]]      |
| dropout_1   | Dropout       | fc6_2                       | fc6_3;_fc6_mask_1 | [[1,4096]]                          | [[1,4096]]      |
| dropout_1   | Dropout       | fc6_2                       | fc6_3;_fc6_mask_1 | [[1,4096]]                          | [[1,4096]]      |
| fc7_w_0     | ConstantInput |                             |                   | [[4096,4096]]                       | [[4096,4096]]   |
| fc7_b_0     | ConstantInput |                             |                   | [[4096]]                            | [[4096]]        |
| gemm_2      | Gemm          | fc6_3;fc7_w_0;fc7_b_0       | fc7_1             | [[1,4096],[4096,4096],[4096]]       | [[1,4096]]      |
| relu_7      | Relu          | fc7_1                       | fc7_2             | [[1,4096]]                          | [[1,4096]]      |
| dropout_2   | Dropout       | fc7_2                       | fc7_3;_fc7_mask_1 | [[1,4096]]                          | [[1,4096]]      |
| dropout_2   | Dropout       | fc7_2                       | fc7_3;_fc7_mask_1 | [[1,4096]]                          | [[1,4096]]      |
| fc8_w_0     | ConstantInput |                             |                   | [[1000,4096]]                       | [[1000,4096]]   |
| fc8_b_0     | ConstantInput |                             |                   | [[1000]]                            | [[1000]]        |
| gemm_3      | Gemm          | fc7_3;fc8_w_0;fc8_b_0       | fc8_1             | [[1,4096],[1000,4096],[1000]]       | [[1,1000]]      |
| softmax_1   | Softmax       | fc8_1                       | prob_1            | [[1,1000]]                          | [[1,1000]]      |
