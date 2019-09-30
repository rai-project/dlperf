|             LAYERNAME              |   LAYERTYPE   |                       INPUTNAMES                       |         OUTPUTNAMES          |       INPUTSHAPES        |  OUTPUTSHAPES  |
|------------------------------------|---------------|--------------------------------------------------------|------------------------------|--------------------------|----------------|
| Parameter193                       | ConstantInput |                                                        |                              | [[16,4,4,10]]            | [[16,4,4,10]]  |
| Parameter193_reshape1_shape        | ConstantInput |                                                        |                              | [[256,10]]               | [[256,10]]     |
| Times212_reshape1                  | Reshape       | Parameter193;Parameter193_reshape1_shape               | Parameter193_reshape1        | [[16,4,4,10],[256,10]]   | [[256,10]]     |
| Input3                             | ConstantInput |                                                        |                              | [[1,1,28,28]]            | [[1,1,28,28]]  |
| Parameter5                         | ConstantInput |                                                        |                              | [[8,1,5,5]]              | [[8,1,5,5]]    |
| Convolution28                      | Conv          | Input3;Parameter5                                      | Convolution28_Output_0       | [[1,1,28,28],[8,1,5,5]]  | [[1,8,28,28]]  |
| Parameter6                         | ConstantInput |                                                        |                              | [[8,1,1]]                | [[8,1,1]]      |
| Plus30                             | ElementWise   | Convolution28_Output_0;Parameter6                      | Plus30_Output_0              | [[1,8,28,28],[8,1,1]]    | [[1,8,28,28]]  |
| ReLU32                             | Relu          | Plus30_Output_0                                        | ReLU32_Output_0              | [[1,8,28,28]]            | [[1,8,28,28]]  |
| Pooling66                          | Pooling       | ReLU32_Output_0                                        | Pooling66_Output_0           | [[1,8,28,28]]            | [[1,8,14,14]]  |
| Parameter87                        | ConstantInput |                                                        |                              | [[16,8,5,5]]             | [[16,8,5,5]]   |
| Convolution110                     | Conv          | Pooling66_Output_0;Parameter87                         | Convolution110_Output_0      | [[1,8,14,14],[16,8,5,5]] | [[1,16,14,14]] |
| Parameter88                        | ConstantInput |                                                        |                              | [[16,1,1]]               | [[16,1,1]]     |
| Plus112                            | ElementWise   | Convolution110_Output_0;Parameter88                    | Plus112_Output_0             | [[1,16,14,14],[16,1,1]]  | [[1,16,14,14]] |
| ReLU114                            | Relu          | Plus112_Output_0                                       | ReLU114_Output_0             | [[1,16,14,14]]           | [[1,16,14,14]] |
| Pooling160                         | Pooling       | ReLU114_Output_0                                       | Pooling160_Output_0          | [[1,16,14,14]]           | [[1,16,4,4]]   |
| Pooling160_Output_0_reshape0_shape | ConstantInput |                                                        |                              | [[1,256]]                | [[1,256]]      |
| Times212_reshape0                  | Reshape       | Pooling160_Output_0;Pooling160_Output_0_reshape0_shape | Pooling160_Output_0_reshape0 | [[1,16,4,4],[1,256]]     | [[1,256]]      |
| Times212                           | Gemm          | Pooling160_Output_0_reshape0;Parameter193_reshape1     | Times212_Output_0            | [[1,256],[256,10]]       | [[1,10]]       |
| Parameter194                       | ConstantInput |                                                        |                              | [[1,10]]                 | [[1,10]]       |
| Plus214                            | ElementWise   | Times212_Output_0;Parameter194                         | Plus214_Output_0             | [[1,10],[1,10]]          | [[1,10]]       |
