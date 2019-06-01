
# Cloud Cost

## Get Data

see https://banzaicloud.com/cloudinfo/

```
curl -L -X GET 'https://banzaicloud.com/cloudinfo/api/v1/providers/amazon/services/compute/regions/us-east-1/products' > _fixtures/aws_instances.json
curl -L -X GET 'https://banzaicloud.com/cloudinfo/api/v1/providers/google/services/compute/regions/us-east4/products' > _fixtures/google_instances.json
curl -L -X GET 'https://banzaicloud.com/cloudinfo/api/v1/providers/azure/services/compute/regions/eastus/products' > _fixtures/azure_instances.json
```

## Resources


https://github.com/aws/aws-sdk-go/tree/master/service/costexplorer

https://github.com/hacker65536/awspricing

https://github.com/alexjguy/AWS-costs-markdown

https://github.com/banzaicloud/cloudinfo/blob/master/internal/cloudinfo/testdata/products.json
