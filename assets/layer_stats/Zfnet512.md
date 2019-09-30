|    LAYERNAME    |   LAYERTYPE   |                  INPUTNAMES                   |   OUTPUTNAMES   |             INPUTSHAPES             |   OUTPUTSHAPES   |
|-----------------|---------------|-----------------------------------------------|-----------------|-------------------------------------|------------------|
| gpu_0/data_0    | ConstantInput |                                               |                 | [[1,3,224,224]]                     | [[1,3,224,224]]  |
| gpu_0/conv1_w_0 | ConstantInput |                                               |                 | [[96,3,7,7]]                        | [[96,3,7,7]]     |
| gpu_0/conv1_b_0 | ConstantInput |                                               |                 | [[96]]                              | [[96]]           |
| conv_1          | Conv          | gpu_0/data_0;gpu_0/conv1_w_0;gpu_0/conv1_b_0  | gpu_0/conv1_1   | [[1,3,224,224],[96,3,7,7],[96]]     | [[1,96,109,109]] |
| relu_1          | Relu          | gpu_0/conv1_1                                 | gpu_0/conv1_2   | [[1,96,109,109]]                    | [[1,96,109,109]] |
| lrn_1           | LRN           | gpu_0/conv1_2                                 | gpu_0/norm1_1   | [[1,96,109,109]]                    | [[1,96,109,109]] |
| maxpool_1       | Pooling       | gpu_0/norm1_1                                 | gpu_0/pool1_1   | [[1,96,109,109]]                    | [[1,96,54,54]]   |
| gpu_0/conv2_w_0 | ConstantInput |                                               |                 | [[256,96,5,5]]                      | [[256,96,5,5]]   |
| gpu_0/conv2_b_0 | ConstantInput |                                               |                 | [[256]]                             | [[256]]          |
| conv_2          | Conv          | gpu_0/pool1_1;gpu_0/conv2_w_0;gpu_0/conv2_b_0 | gpu_0/conv2_1   | [[1,96,54,54],[256,96,5,5],[256]]   | [[1,256,25,25]]  |
| relu_2          | Relu          | gpu_0/conv2_1                                 | gpu_0/conv2_2   | [[1,256,25,25]]                     | [[1,256,25,25]]  |
| lrn_2           | LRN           | gpu_0/conv2_2                                 | gpu_0/norm2_1   | [[1,256,25,25]]                     | [[1,256,25,25]]  |
| maxpool_2       | Pooling       | gpu_0/norm2_1                                 | gpu_0/pool2_1   | [[1,256,25,25]]                     | [[1,256,12,12]]  |
| gpu_0/conv3_w_0 | ConstantInput |                                               |                 | [[512,256,3,3]]                     | [[512,256,3,3]]  |
| gpu_0/conv3_b_0 | ConstantInput |                                               |                 | [[512]]                             | [[512]]          |
| conv_3          | Conv          | gpu_0/pool2_1;gpu_0/conv3_w_0;gpu_0/conv3_b_0 | gpu_0/conv3_1   | [[1,256,12,12],[512,256,3,3],[512]] | [[1,512,12,12]]  |
| relu_3          | Relu          | gpu_0/conv3_1                                 | gpu_0/conv3_2   | [[1,512,12,12]]                     | [[1,512,12,12]]  |
| gpu_0/conv4_w_0 | ConstantInput |                                               |                 | [[512,512,3,3]]                     | [[512,512,3,3]]  |
| gpu_0/conv4_b_0 | ConstantInput |                                               |                 | [[512]]                             | [[512]]          |
| conv_4          | Conv          | gpu_0/conv3_2;gpu_0/conv4_w_0;gpu_0/conv4_b_0 | gpu_0/conv4_1   | [[1,512,12,12],[512,512,3,3],[512]] | [[1,512,12,12]]  |
| relu_4          | Relu          | gpu_0/conv4_1                                 | gpu_0/conv4_2   | [[1,512,12,12]]                     | [[1,512,12,12]]  |
| gpu_0/conv5_w_0 | ConstantInput |                                               |                 | [[512,512,3,3]]                     | [[512,512,3,3]]  |
| gpu_0/conv5_b_0 | ConstantInput |                                               |                 | [[512]]                             | [[512]]          |
| conv_5          | Conv          | gpu_0/conv4_2;gpu_0/conv5_w_0;gpu_0/conv5_b_0 | gpu_0/conv5_1   | [[1,512,12,12],[512,512,3,3],[512]] | [[1,512,12,12]]  |
| relu_5          | Relu          | gpu_0/conv5_1                                 | gpu_0/conv5_2   | [[1,512,12,12]]                     | [[1,512,12,12]]  |
| maxpool_3       | Pooling       | gpu_0/conv5_2                                 | gpu_0/pool5_1   | [[1,512,12,12]]                     | [[1,512,6,6]]    |
| OC2_DUMMY_1     | ConstantInput |                                               |                 | [[1,18432]]                         | [[1,18432]]      |
| reshape_1       | Reshape       | gpu_0/pool5_1;OC2_DUMMY_1                     | OC2_DUMMY_0     | [[1,512,6,6],[1,18432]]             | [[1,18432]]      |
| gpu_0/fc6_w_0   | ConstantInput |                                               |                 | [[4096,18432]]                      | [[4096,18432]]   |
| gpu_0/fc6_b_0   | ConstantInput |                                               |                 | [[4096]]                            | [[4096]]         |
| gemm_1          | Gemm          | OC2_DUMMY_0;gpu_0/fc6_w_0;gpu_0/fc6_b_0       | gpu_0/fc6_1     | [[1,18432],[4096,18432],[4096]]     | [[1,4096]]       |
| relu_6          | Relu          | gpu_0/fc6_1                                   | gpu_0/fc6_2     | [[1,4096]]                          | [[1,4096]]       |
| gpu_0/fc7_w_0   | ConstantInput |                                               |                 | [[1024,4096]]                       | [[1024,4096]]    |
| gpu_0/fc7_b_0   | ConstantInput |                                               |                 | [[1024]]                            | [[1024]]         |
| gemm_2          | Gemm          | gpu_0/fc6_2;gpu_0/fc7_w_0;gpu_0/fc7_b_0       | gpu_0/fc7_1     | [[1,4096],[1024,4096],[1024]]       | [[1,1024]]       |
| relu_7          | Relu          | gpu_0/fc7_1                                   | gpu_0/fc7_2     | [[1,1024]]                          | [[1,1024]]       |
| gpu_0/fc8_w_0   | ConstantInput |                                               |                 | [[1000,1024]]                       | [[1000,1024]]    |
| gpu_0/fc8_b_0   | ConstantInput |                                               |                 | [[1000]]                            | [[1000]]         |
| gemm_3          | Gemm          | gpu_0/fc7_2;gpu_0/fc8_w_0;gpu_0/fc8_b_0       | gpu_0/pred_1    | [[1,1024],[1000,1024],[1000]]       | [[1,1000]]       |
| softmax_1       | Softmax       | gpu_0/pred_1                                  | gpu_0/softmax_1 | [[1,1000]]                          | [[1,1000]]       |
