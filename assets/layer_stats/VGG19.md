|      LAYERNAME       |   LAYERTYPE   |                     INPUTNAMES                      |     OUTPUTNAMES      |              INPUTSHAPES              |   OUTPUTSHAPES    |
|----------------------|---------------|-----------------------------------------------------|----------------------|---------------------------------------|-------------------|
| data                 | ConstantInput |                                                     |                      | [[1,3,224,224]]                       | [[1,3,224,224]]   |
| vgg0_conv0_weight    | ConstantInput |                                                     |                      | [[64,3,3,3]]                          | [[64,3,3,3]]      |
| vgg0_conv0_bias      | ConstantInput |                                                     |                      | [[64]]                                | [[64]]            |
| vgg0_conv0_fwd       | Conv          | data;vgg0_conv0_weight;vgg0_conv0_bias              | vgg0_conv0_fwd       | [[1,3,224,224],[64,3,3,3],[64]]       | [[1,64,224,224]]  |
| vgg0_relu0_fwd       | Relu          | vgg0_conv0_fwd                                      | vgg0_relu0_fwd       | [[1,64,224,224]]                      | [[1,64,224,224]]  |
| vgg0_conv1_weight    | ConstantInput |                                                     |                      | [[64,64,3,3]]                         | [[64,64,3,3]]     |
| vgg0_conv1_bias      | ConstantInput |                                                     |                      | [[64]]                                | [[64]]            |
| vgg0_conv1_fwd       | Conv          | vgg0_relu0_fwd;vgg0_conv1_weight;vgg0_conv1_bias    | vgg0_conv1_fwd       | [[1,64,224,224],[64,64,3,3],[64]]     | [[1,64,224,224]]  |
| vgg0_relu1_fwd       | Relu          | vgg0_conv1_fwd                                      | vgg0_relu1_fwd       | [[1,64,224,224]]                      | [[1,64,224,224]]  |
| vgg0_pool0_fwd       | Pooling       | vgg0_relu1_fwd                                      | vgg0_pool0_fwd       | [[1,64,224,224]]                      | [[1,64,112,112]]  |
| vgg0_conv2_weight    | ConstantInput |                                                     |                      | [[128,64,3,3]]                        | [[128,64,3,3]]    |
| vgg0_conv2_bias      | ConstantInput |                                                     |                      | [[128]]                               | [[128]]           |
| vgg0_conv2_fwd       | Conv          | vgg0_pool0_fwd;vgg0_conv2_weight;vgg0_conv2_bias    | vgg0_conv2_fwd       | [[1,64,112,112],[128,64,3,3],[128]]   | [[1,128,112,112]] |
| vgg0_relu2_fwd       | Relu          | vgg0_conv2_fwd                                      | vgg0_relu2_fwd       | [[1,128,112,112]]                     | [[1,128,112,112]] |
| vgg0_conv3_weight    | ConstantInput |                                                     |                      | [[128,128,3,3]]                       | [[128,128,3,3]]   |
| vgg0_conv3_bias      | ConstantInput |                                                     |                      | [[128]]                               | [[128]]           |
| vgg0_conv3_fwd       | Conv          | vgg0_relu2_fwd;vgg0_conv3_weight;vgg0_conv3_bias    | vgg0_conv3_fwd       | [[1,128,112,112],[128,128,3,3],[128]] | [[1,128,112,112]] |
| vgg0_relu3_fwd       | Relu          | vgg0_conv3_fwd                                      | vgg0_relu3_fwd       | [[1,128,112,112]]                     | [[1,128,112,112]] |
| vgg0_pool1_fwd       | Pooling       | vgg0_relu3_fwd                                      | vgg0_pool1_fwd       | [[1,128,112,112]]                     | [[1,128,56,56]]   |
| vgg0_conv4_weight    | ConstantInput |                                                     |                      | [[256,128,3,3]]                       | [[256,128,3,3]]   |
| vgg0_conv4_bias      | ConstantInput |                                                     |                      | [[256]]                               | [[256]]           |
| vgg0_conv4_fwd       | Conv          | vgg0_pool1_fwd;vgg0_conv4_weight;vgg0_conv4_bias    | vgg0_conv4_fwd       | [[1,128,56,56],[256,128,3,3],[256]]   | [[1,256,56,56]]   |
| vgg0_relu4_fwd       | Relu          | vgg0_conv4_fwd                                      | vgg0_relu4_fwd       | [[1,256,56,56]]                       | [[1,256,56,56]]   |
| vgg0_conv5_weight    | ConstantInput |                                                     |                      | [[256,256,3,3]]                       | [[256,256,3,3]]   |
| vgg0_conv5_bias      | ConstantInput |                                                     |                      | [[256]]                               | [[256]]           |
| vgg0_conv5_fwd       | Conv          | vgg0_relu4_fwd;vgg0_conv5_weight;vgg0_conv5_bias    | vgg0_conv5_fwd       | [[1,256,56,56],[256,256,3,3],[256]]   | [[1,256,56,56]]   |
| vgg0_relu5_fwd       | Relu          | vgg0_conv5_fwd                                      | vgg0_relu5_fwd       | [[1,256,56,56]]                       | [[1,256,56,56]]   |
| vgg0_conv6_weight    | ConstantInput |                                                     |                      | [[256,256,3,3]]                       | [[256,256,3,3]]   |
| vgg0_conv6_bias      | ConstantInput |                                                     |                      | [[256]]                               | [[256]]           |
| vgg0_conv6_fwd       | Conv          | vgg0_relu5_fwd;vgg0_conv6_weight;vgg0_conv6_bias    | vgg0_conv6_fwd       | [[1,256,56,56],[256,256,3,3],[256]]   | [[1,256,56,56]]   |
| vgg0_relu6_fwd       | Relu          | vgg0_conv6_fwd                                      | vgg0_relu6_fwd       | [[1,256,56,56]]                       | [[1,256,56,56]]   |
| vgg0_conv7_weight    | ConstantInput |                                                     |                      | [[256,256,3,3]]                       | [[256,256,3,3]]   |
| vgg0_conv7_bias      | ConstantInput |                                                     |                      | [[256]]                               | [[256]]           |
| vgg0_conv7_fwd       | Conv          | vgg0_relu6_fwd;vgg0_conv7_weight;vgg0_conv7_bias    | vgg0_conv7_fwd       | [[1,256,56,56],[256,256,3,3],[256]]   | [[1,256,56,56]]   |
| vgg0_relu7_fwd       | Relu          | vgg0_conv7_fwd                                      | vgg0_relu7_fwd       | [[1,256,56,56]]                       | [[1,256,56,56]]   |
| vgg0_pool2_fwd       | Pooling       | vgg0_relu7_fwd                                      | vgg0_pool2_fwd       | [[1,256,56,56]]                       | [[1,256,28,28]]   |
| vgg0_conv8_weight    | ConstantInput |                                                     |                      | [[512,256,3,3]]                       | [[512,256,3,3]]   |
| vgg0_conv8_bias      | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv8_fwd       | Conv          | vgg0_pool2_fwd;vgg0_conv8_weight;vgg0_conv8_bias    | vgg0_conv8_fwd       | [[1,256,28,28],[512,256,3,3],[512]]   | [[1,512,28,28]]   |
| vgg0_relu8_fwd       | Relu          | vgg0_conv8_fwd                                      | vgg0_relu8_fwd       | [[1,512,28,28]]                       | [[1,512,28,28]]   |
| vgg0_conv9_weight    | ConstantInput |                                                     |                      | [[512,512,3,3]]                       | [[512,512,3,3]]   |
| vgg0_conv9_bias      | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv9_fwd       | Conv          | vgg0_relu8_fwd;vgg0_conv9_weight;vgg0_conv9_bias    | vgg0_conv9_fwd       | [[1,512,28,28],[512,512,3,3],[512]]   | [[1,512,28,28]]   |
| vgg0_relu9_fwd       | Relu          | vgg0_conv9_fwd                                      | vgg0_relu9_fwd       | [[1,512,28,28]]                       | [[1,512,28,28]]   |
| vgg0_conv10_weight   | ConstantInput |                                                     |                      | [[512,512,3,3]]                       | [[512,512,3,3]]   |
| vgg0_conv10_bias     | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv10_fwd      | Conv          | vgg0_relu9_fwd;vgg0_conv10_weight;vgg0_conv10_bias  | vgg0_conv10_fwd      | [[1,512,28,28],[512,512,3,3],[512]]   | [[1,512,28,28]]   |
| vgg0_relu10_fwd      | Relu          | vgg0_conv10_fwd                                     | vgg0_relu10_fwd      | [[1,512,28,28]]                       | [[1,512,28,28]]   |
| vgg0_conv11_weight   | ConstantInput |                                                     |                      | [[512,512,3,3]]                       | [[512,512,3,3]]   |
| vgg0_conv11_bias     | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv11_fwd      | Conv          | vgg0_relu10_fwd;vgg0_conv11_weight;vgg0_conv11_bias | vgg0_conv11_fwd      | [[1,512,28,28],[512,512,3,3],[512]]   | [[1,512,28,28]]   |
| vgg0_relu11_fwd      | Relu          | vgg0_conv11_fwd                                     | vgg0_relu11_fwd      | [[1,512,28,28]]                       | [[1,512,28,28]]   |
| vgg0_pool3_fwd       | Pooling       | vgg0_relu11_fwd                                     | vgg0_pool3_fwd       | [[1,512,28,28]]                       | [[1,512,14,14]]   |
| vgg0_conv12_weight   | ConstantInput |                                                     |                      | [[512,512,3,3]]                       | [[512,512,3,3]]   |
| vgg0_conv12_bias     | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv12_fwd      | Conv          | vgg0_pool3_fwd;vgg0_conv12_weight;vgg0_conv12_bias  | vgg0_conv12_fwd      | [[1,512,14,14],[512,512,3,3],[512]]   | [[1,512,14,14]]   |
| vgg0_relu12_fwd      | Relu          | vgg0_conv12_fwd                                     | vgg0_relu12_fwd      | [[1,512,14,14]]                       | [[1,512,14,14]]   |
| vgg0_conv13_weight   | ConstantInput |                                                     |                      | [[512,512,3,3]]                       | [[512,512,3,3]]   |
| vgg0_conv13_bias     | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv13_fwd      | Conv          | vgg0_relu12_fwd;vgg0_conv13_weight;vgg0_conv13_bias | vgg0_conv13_fwd      | [[1,512,14,14],[512,512,3,3],[512]]   | [[1,512,14,14]]   |
| vgg0_relu13_fwd      | Relu          | vgg0_conv13_fwd                                     | vgg0_relu13_fwd      | [[1,512,14,14]]                       | [[1,512,14,14]]   |
| vgg0_conv14_weight   | ConstantInput |                                                     |                      | [[512,512,3,3]]                       | [[512,512,3,3]]   |
| vgg0_conv14_bias     | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv14_fwd      | Conv          | vgg0_relu13_fwd;vgg0_conv14_weight;vgg0_conv14_bias | vgg0_conv14_fwd      | [[1,512,14,14],[512,512,3,3],[512]]   | [[1,512,14,14]]   |
| vgg0_relu14_fwd      | Relu          | vgg0_conv14_fwd                                     | vgg0_relu14_fwd      | [[1,512,14,14]]                       | [[1,512,14,14]]   |
| vgg0_conv15_weight   | ConstantInput |                                                     |                      | [[512,512,3,3]]                       | [[512,512,3,3]]   |
| vgg0_conv15_bias     | ConstantInput |                                                     |                      | [[512]]                               | [[512]]           |
| vgg0_conv15_fwd      | Conv          | vgg0_relu14_fwd;vgg0_conv15_weight;vgg0_conv15_bias | vgg0_conv15_fwd      | [[1,512,14,14],[512,512,3,3],[512]]   | [[1,512,14,14]]   |
| vgg0_relu15_fwd      | Relu          | vgg0_conv15_fwd                                     | vgg0_relu15_fwd      | [[1,512,14,14]]                       | [[1,512,14,14]]   |
| vgg0_pool4_fwd       | Pooling       | vgg0_relu15_fwd                                     | vgg0_pool4_fwd       | [[1,512,14,14]]                       | [[1,512,7,7]]     |
| flatten_72           | Flatten       | vgg0_pool4_fwd                                      | flatten_72           | [[1,512,7,7]]                         | [[1,512,7,7]]     |
| vgg0_dense0_weight   | ConstantInput |                                                     |                      | [[4096,25088]]                        | [[4096,25088]]    |
| vgg0_dense0_bias     | ConstantInput |                                                     |                      | [[4096]]                              | [[4096]]          |
| vgg0_dense0_fwd      | Gemm          | flatten_72;vgg0_dense0_weight;vgg0_dense0_bias      | vgg0_dense0_fwd      | [[1,512,7,7],[4096,25088],[4096]]     | [[1,4096]]        |
| vgg0_dense0_relu_fwd | Relu          | vgg0_dense0_fwd                                     | vgg0_dense0_relu_fwd | [[1,4096]]                            | [[1,4096]]        |
| vgg0_dropout0_fwd    | Dropout       | vgg0_dense0_relu_fwd                                | vgg0_dropout0_fwd    | [[1,4096]]                            | [[1,4096]]        |
| flatten_77           | Flatten       | vgg0_dropout0_fwd                                   | flatten_77           | [[1,4096]]                            | [[1,4096]]        |
| vgg0_dense1_weight   | ConstantInput |                                                     |                      | [[4096,4096]]                         | [[4096,4096]]     |
| vgg0_dense1_bias     | ConstantInput |                                                     |                      | [[4096]]                              | [[4096]]          |
| vgg0_dense1_fwd      | Gemm          | flatten_77;vgg0_dense1_weight;vgg0_dense1_bias      | vgg0_dense1_fwd      | [[1,4096],[4096,4096],[4096]]         | [[1,4096]]        |
| vgg0_dense1_relu_fwd | Relu          | vgg0_dense1_fwd                                     | vgg0_dense1_relu_fwd | [[1,4096]]                            | [[1,4096]]        |
| vgg0_dropout1_fwd    | Dropout       | vgg0_dense1_relu_fwd                                | vgg0_dropout1_fwd    | [[1,4096]]                            | [[1,4096]]        |
| flatten_82           | Flatten       | vgg0_dropout1_fwd                                   | flatten_82           | [[1,4096]]                            | [[1,4096]]        |
| vgg0_dense2_weight   | ConstantInput |                                                     |                      | [[1000,4096]]                         | [[1000,4096]]     |
| vgg0_dense2_bias     | ConstantInput |                                                     |                      | [[1000]]                              | [[1000]]          |
| vgg0_dense2_fwd      | Gemm          | flatten_82;vgg0_dense2_weight;vgg0_dense2_bias      | vgg0_dense2_fwd      | [[1,4096],[1000,4096],[1000]]         | [[1,1000]]        |
