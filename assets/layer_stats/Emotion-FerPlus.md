|             LAYERNAME              |   LAYERTYPE   |                       INPUTNAMES                       |         OUTPUTNAMES          |          INPUTSHAPES          |   OUTPUTSHAPES   |
|------------------------------------|---------------|--------------------------------------------------------|------------------------------|-------------------------------|------------------|
| Parameter1367                      | ConstantInput |                                                        |                              | [[256,4,4,1024]]              | [[256,4,4,1024]] |
| Parameter1367_reshape1_shape       | ConstantInput |                                                        |                              | [[4096,1024]]                 | [[4096,1024]]    |
| Times622_reshape1                  | Reshape       | Parameter1367;Parameter1367_reshape1_shape             | Parameter1367_reshape1       | [[256,4,4,1024],[4096,1024]]  | [[4096,1024]]    |
| Input3                             | ConstantInput |                                                        |                              | [[1,1,64,64]]                 | [[1,1,64,64]]    |
| Constant339                        | ConstantInput |                                                        |                              | [[]]                          | [[]]             |
| Minus340                           | ElementWise   | Input3;Constant339                                     | Minus340_Output_0            | [[1,1,64,64],[]]              | [[1,1,64,64]]    |
| Constant343                        | ConstantInput |                                                        |                              | [[]]                          | [[]]             |
| Block352                           | ElementWise   | Minus340_Output_0;Constant343                          | Block352_Output_0            | [[1,1,64,64],[]]              | [[1,1,64,64]]    |
| Parameter3                         | ConstantInput |                                                        |                              | [[64,1,3,3]]                  | [[64,1,3,3]]     |
| Convolution362                     | Conv          | Block352_Output_0;Parameter3                           | Convolution362_Output_0      | [[1,1,64,64],[64,1,3,3]]      | [[1,64,64,64]]   |
| Parameter4                         | ConstantInput |                                                        |                              | [[64,1,1]]                    | [[64,1,1]]       |
| Plus364                            | ElementWise   | Convolution362_Output_0;Parameter4                     | Plus364_Output_0             | [[1,64,64,64],[64,1,1]]       | [[1,64,64,64]]   |
| ReLU366                            | Relu          | Plus364_Output_0                                       | ReLU366_Output_0             | [[1,64,64,64]]                | [[1,64,64,64]]   |
| Parameter23                        | ConstantInput |                                                        |                              | [[64,64,3,3]]                 | [[64,64,3,3]]    |
| Convolution380                     | Conv          | ReLU366_Output_0;Parameter23                           | Convolution380_Output_0      | [[1,64,64,64],[64,64,3,3]]    | [[1,64,64,64]]   |
| Parameter24                        | ConstantInput |                                                        |                              | [[64,1,1]]                    | [[64,1,1]]       |
| Plus382                            | ElementWise   | Convolution380_Output_0;Parameter24                    | Plus382_Output_0             | [[1,64,64,64],[64,1,1]]       | [[1,64,64,64]]   |
| ReLU384                            | Relu          | Plus382_Output_0                                       | ReLU384_Output_0             | [[1,64,64,64]]                | [[1,64,64,64]]   |
| Pooling398                         | Pooling       | ReLU384_Output_0                                       | Pooling398_Output_0          | [[1,64,64,64]]                | [[1,64,32,32]]   |
| Dropout408                         | Dropout       | Pooling398_Output_0                                    | Dropout408_Output_0          | [[1,64,32,32]]                | [[1,64,32,32]]   |
| Parameter63                        | ConstantInput |                                                        |                              | [[128,64,3,3]]                | [[128,64,3,3]]   |
| Convolution418                     | Conv          | Dropout408_Output_0;Parameter63                        | Convolution418_Output_0      | [[1,64,32,32],[128,64,3,3]]   | [[1,128,32,32]]  |
| Parameter64                        | ConstantInput |                                                        |                              | [[128,1,1]]                   | [[128,1,1]]      |
| Plus420                            | ElementWise   | Convolution418_Output_0;Parameter64                    | Plus420_Output_0             | [[1,128,32,32],[128,1,1]]     | [[1,128,32,32]]  |
| ReLU422                            | Relu          | Plus420_Output_0                                       | ReLU422_Output_0             | [[1,128,32,32]]               | [[1,128,32,32]]  |
| Parameter83                        | ConstantInput |                                                        |                              | [[128,128,3,3]]               | [[128,128,3,3]]  |
| Convolution436                     | Conv          | ReLU422_Output_0;Parameter83                           | Convolution436_Output_0      | [[1,128,32,32],[128,128,3,3]] | [[1,128,32,32]]  |
| Parameter84                        | ConstantInput |                                                        |                              | [[128,1,1]]                   | [[128,1,1]]      |
| Plus438                            | ElementWise   | Convolution436_Output_0;Parameter84                    | Plus438_Output_0             | [[1,128,32,32],[128,1,1]]     | [[1,128,32,32]]  |
| ReLU440                            | Relu          | Plus438_Output_0                                       | ReLU440_Output_0             | [[1,128,32,32]]               | [[1,128,32,32]]  |
| Pooling454                         | Pooling       | ReLU440_Output_0                                       | Pooling454_Output_0          | [[1,128,32,32]]               | [[1,128,16,16]]  |
| Dropout464                         | Dropout       | Pooling454_Output_0                                    | Dropout464_Output_0          | [[1,128,16,16]]               | [[1,128,16,16]]  |
| Parameter575                       | ConstantInput |                                                        |                              | [[256,128,3,3]]               | [[256,128,3,3]]  |
| Convolution474                     | Conv          | Dropout464_Output_0;Parameter575                       | Convolution474_Output_0      | [[1,128,16,16],[256,128,3,3]] | [[1,256,16,16]]  |
| Parameter576                       | ConstantInput |                                                        |                              | [[256,1,1]]                   | [[256,1,1]]      |
| Plus476                            | ElementWise   | Convolution474_Output_0;Parameter576                   | Plus476_Output_0             | [[1,256,16,16],[256,1,1]]     | [[1,256,16,16]]  |
| ReLU478                            | Relu          | Plus476_Output_0                                       | ReLU478_Output_0             | [[1,256,16,16]]               | [[1,256,16,16]]  |
| Parameter595                       | ConstantInput |                                                        |                              | [[256,256,3,3]]               | [[256,256,3,3]]  |
| Convolution492                     | Conv          | ReLU478_Output_0;Parameter595                          | Convolution492_Output_0      | [[1,256,16,16],[256,256,3,3]] | [[1,256,16,16]]  |
| Parameter596                       | ConstantInput |                                                        |                              | [[256,1,1]]                   | [[256,1,1]]      |
| Plus494                            | ElementWise   | Convolution492_Output_0;Parameter596                   | Plus494_Output_0             | [[1,256,16,16],[256,1,1]]     | [[1,256,16,16]]  |
| ReLU496                            | Relu          | Plus494_Output_0                                       | ReLU496_Output_0             | [[1,256,16,16]]               | [[1,256,16,16]]  |
| Parameter615                       | ConstantInput |                                                        |                              | [[256,256,3,3]]               | [[256,256,3,3]]  |
| Convolution510                     | Conv          | ReLU496_Output_0;Parameter615                          | Convolution510_Output_0      | [[1,256,16,16],[256,256,3,3]] | [[1,256,16,16]]  |
| Parameter616                       | ConstantInput |                                                        |                              | [[256,1,1]]                   | [[256,1,1]]      |
| Plus512                            | ElementWise   | Convolution510_Output_0;Parameter616                   | Plus512_Output_0             | [[1,256,16,16],[256,1,1]]     | [[1,256,16,16]]  |
| ReLU514                            | Relu          | Plus512_Output_0                                       | ReLU514_Output_0             | [[1,256,16,16]]               | [[1,256,16,16]]  |
| Pooling528                         | Pooling       | ReLU514_Output_0                                       | Pooling528_Output_0          | [[1,256,16,16]]               | [[1,256,8,8]]    |
| Dropout538                         | Dropout       | Pooling528_Output_0                                    | Dropout538_Output_0          | [[1,256,8,8]]                 | [[1,256,8,8]]    |
| Parameter655                       | ConstantInput |                                                        |                              | [[256,256,3,3]]               | [[256,256,3,3]]  |
| Convolution548                     | Conv          | Dropout538_Output_0;Parameter655                       | Convolution548_Output_0      | [[1,256,8,8],[256,256,3,3]]   | [[1,256,8,8]]    |
| Parameter656                       | ConstantInput |                                                        |                              | [[256,1,1]]                   | [[256,1,1]]      |
| Plus550                            | ElementWise   | Convolution548_Output_0;Parameter656                   | Plus550_Output_0             | [[1,256,8,8],[256,1,1]]       | [[1,256,8,8]]    |
| ReLU552                            | Relu          | Plus550_Output_0                                       | ReLU552_Output_0             | [[1,256,8,8]]                 | [[1,256,8,8]]    |
| Parameter675                       | ConstantInput |                                                        |                              | [[256,256,3,3]]               | [[256,256,3,3]]  |
| Convolution566                     | Conv          | ReLU552_Output_0;Parameter675                          | Convolution566_Output_0      | [[1,256,8,8],[256,256,3,3]]   | [[1,256,8,8]]    |
| Parameter676                       | ConstantInput |                                                        |                              | [[256,1,1]]                   | [[256,1,1]]      |
| Plus568                            | ElementWise   | Convolution566_Output_0;Parameter676                   | Plus568_Output_0             | [[1,256,8,8],[256,1,1]]       | [[1,256,8,8]]    |
| ReLU570                            | Relu          | Plus568_Output_0                                       | ReLU570_Output_0             | [[1,256,8,8]]                 | [[1,256,8,8]]    |
| Parameter695                       | ConstantInput |                                                        |                              | [[256,256,3,3]]               | [[256,256,3,3]]  |
| Convolution584                     | Conv          | ReLU570_Output_0;Parameter695                          | Convolution584_Output_0      | [[1,256,8,8],[256,256,3,3]]   | [[1,256,8,8]]    |
| Parameter696                       | ConstantInput |                                                        |                              | [[256,1,1]]                   | [[256,1,1]]      |
| Plus586                            | ElementWise   | Convolution584_Output_0;Parameter696                   | Plus586_Output_0             | [[1,256,8,8],[256,1,1]]       | [[1,256,8,8]]    |
| ReLU588                            | Relu          | Plus586_Output_0                                       | ReLU588_Output_0             | [[1,256,8,8]]                 | [[1,256,8,8]]    |
| Pooling602                         | Pooling       | ReLU588_Output_0                                       | Pooling602_Output_0          | [[1,256,8,8]]                 | [[1,256,4,4]]    |
| Dropout612                         | Dropout       | Pooling602_Output_0                                    | Dropout612_Output_0          | [[1,256,4,4]]                 | [[1,256,4,4]]    |
| Dropout612_Output_0_reshape0_shape | ConstantInput |                                                        |                              | [[1,4096]]                    | [[1,4096]]       |
| Times622_reshape0                  | Reshape       | Dropout612_Output_0;Dropout612_Output_0_reshape0_shape | Dropout612_Output_0_reshape0 | [[1,256,4,4],[1,4096]]        | [[1,4096]]       |
| Times622                           | Gemm          | Dropout612_Output_0_reshape0;Parameter1367_reshape1    | Times622_Output_0            | [[1,4096],[4096,1024]]        | [[1,1024]]       |
| Parameter1368                      | ConstantInput |                                                        |                              | [[1024]]                      | [[1024]]         |
| Plus624                            | ElementWise   | Times622_Output_0;Parameter1368                        | Plus624_Output_0             | [[1,1024],[1024]]             | [[1,1024]]       |
| ReLU636                            | Relu          | Plus624_Output_0                                       | ReLU636_Output_0             | [[1,1024]]                    | [[1,1024]]       |
| Dropout646                         | Dropout       | ReLU636_Output_0                                       | Dropout646_Output_0          | [[1,1024]]                    | [[1,1024]]       |
| Parameter1403                      | ConstantInput |                                                        |                              | [[1024,1024]]                 | [[1024,1024]]    |
| Times656                           | Gemm          | Dropout646_Output_0;Parameter1403                      | Times656_Output_0            | [[1,1024],[1024,1024]]        | [[1,1024]]       |
| Parameter1404                      | ConstantInput |                                                        |                              | [[1024]]                      | [[1024]]         |
| Plus658                            | ElementWise   | Times656_Output_0;Parameter1404                        | Plus658_Output_0             | [[1,1024],[1024]]             | [[1,1024]]       |
| ReLU670                            | Relu          | Plus658_Output_0                                       | ReLU670_Output_0             | [[1,1024]]                    | [[1,1024]]       |
| Dropout680                         | Dropout       | ReLU670_Output_0                                       | Dropout680_Output_0          | [[1,1024]]                    | [[1,1024]]       |
| Parameter1693                      | ConstantInput |                                                        |                              | [[1024,8]]                    | [[1024,8]]       |
| Times690                           | Gemm          | Dropout680_Output_0;Parameter1693                      | Times690_Output_0            | [[1,1024],[1024,8]]           | [[1,8]]          |
| Parameter1694                      | ConstantInput |                                                        |                              | [[8]]                         | [[8]]            |
| Plus692                            | ElementWise   | Times690_Output_0;Parameter1694                        | Plus692_Output_0             | [[1,8],[8]]                   | [[1,8]]          |
