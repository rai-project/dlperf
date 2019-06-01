import json
import requests
from decimal import Decimal

#Read EC2 JSON data into the variable
ec2_url = 'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.json'
ec2_filename = 'ComputeInstance.json'

if ec2_filename:
    try:
      with open(ec2_filename, 'r') as f:
        ec2_data = json.load(f)
    except:
      resp = requests.get(url=ec2_url)
      vpc_data = json.loads(resp.text)
else:
    resp = requests.get(url=ec2_url)
    ec2_data = json.loads(resp.text)

ec2_products = ec2_data['products']

#Read VPC JSON data into the  vaiable
vpc_url = 'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonVPC/current/index.json'
vpc_filename = 'VPC.json'

if vpc_filename:
    try:
      with open(vpc_filename, 'r') as f:
        vpc_data = json.load(f)
    except:
      resp = requests.get(url=vpc_url)
      vpc_data = json.loads(resp.text)
else:
    resp = requests.get(url=vpc_url)
    vpc_data = json.loads(resp.text)

vpc_products = vpc_data['products']

#Read S3 JSON data into the  vaiable
s3_url = 'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/S3/current/index.json'
s3_filename = 'S3.json'

if s3_filename:
    try:
      with open(s3_filename, 'r') as f:
        s3_data = json.load(f)
    except:
      resp = requests.get(url=s3_url)
      s3_data = json.loads(resp.text)
else:
    resp = requests.get(url=s3_url)
    s3_data = json.loads(resp.text)

s3_products = s3_data['products']

#Read CloudWatch JSON data into the  vaiable
cw_url = 'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonCloudWatch/current/index.json'
cw_filename = 'CloudWatch.json'

if cw_filename:
    try:
      with open(cw_filename, 'r') as f:
        cw_data = json.load(f)
    except:
      resp = requests.get(url=cw_url)
      cw_data = json.loads(resp.text)
else:
    resp = requests.get(url=cw_url)
    cw_data = json.loads(resp.text)

cw_products = cw_data['products']


#the datacenter that we need output for
location_filter = ["US East (Ohio)"]

#the list of OS that we need
operatingSystem_filter = ["RHEL", "Windows", "SUSE"]

#the list of instance types that we need output for
instanceType_filter = [
    "t2.nano",
    "t2.micro",
    "t2.small",
    "t2.medium",
    "t2.large",
    "t2.xlarge",
    "t2.2xlarge",

    "m5d.large",
    "m5d.xlarge",
    "m5d.2xlarge",
    "m5d.4xlarge",
    "m5d.12xlarge",
    "m5d.24xlarge",
  
    "m5.large",
    "m5.xlarge",
    "m5.2xlarge",
    "m5.4xlarge",
    "m5.12xlarge",
    "m5.24xlarge",  

    "m4.large",
    "m4.xlarge",
    "m4.2xlarge",
    "m4.4xlarge",
    "m4.10xlarge",
    "m4.16xlarge",

    "c5d.large",
    "c5d.xlarge",
    "c5d.2xlarge",
    "c5d.4xlarge",
    "c5d.9xlarge",
    "c5d.18xlarge",

    "c5.large",
    "c5.xlarge",
    "c5.2xlarge",
    "c5.4xlarge",
    "c5.9xlarge",
    "c5.18xlarge",

    "c4.large",
    "c4.xlarge",
    "c4.2xlarge",
    "c4.4xlarge",
    "c4.8xlarge",

    "g3.4xlarge",
    "g3.8xlarge",
    "g3.16xlarge",

    "p2.xlarge",
    "p2.8xlarge",
    "p2.16xlarge",

    "p3.2xlarge",
    "p3.8xlarge",
    "p3.16xlarge",

    "r4.large",
    "r4.xlarge",
    "r4.2xlarge",
    "r4.4xlarge",
    "r4.8xlarge",
    "r4.16xlarge",

    "x1.16xlarge"
    "x1.32xlarge"

    "d2.xlarge",
    "d2.2xlarge",
    "d2.4xlarge",
    "d2.8xlarge",

    "i2.xlarge",
    "i2.2xlarge",
    "i2.4xlarge",
    "i2.8xlarge",

    "h1.2xlarge",
    "h1.4xlarge",
    "h1.8xlarge",
    "h1.16xlarge",

    "i3.large",
    "i3.xlarge",
    "i3.2xlarge",
    "i3.4xlarge",
    "i3.8xlarge",
    "i3.16xlarge",
    "i3.metal"
]

#{"transferType" : "IntraRegion","endpointType" : "IPsec","transferType" : "AWS Inbound","transferType" : "AWS Outbound",}
vpc_productArr=["AWS Outbound", "IntraRegion", "ElasticIP:Address", "ElasticIP:AdditionalAddress", "CreateVpnConnection", "VPCE:VpcEndpoint", "NatGateway", "VPN Per Hour"]
s3_productArr=["Standard Access", "Standard-Infrequent Access", "One Zone-Infrequent Access", "Amazon Glacier Requests"]
usageType_filter = ["USE2-VPN-Usage-Hours:ipsec.1", "USE2-DataTransfer-In-Bytes", "USE2-DataTransfer-Out-Bytes","USE2-DataTransfer-Regional-Bytes"]
productFamily_filter = ["Data Transfer", "VpcEndpoint", "Cloud Connectivity", "IP Address", "NAT Gateway", "Load Balancer-Network","Storage", "API Request","Compute Instance" ]

config = {
  "services":{
    "AmazonCloudWatch" : {
      "url":"https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonCloudWatch/current/index.json",
      "file":"CloudWatch.json",
      "productFamilies": ["Data Payload","Storage Snapshot","Metric","Alarm","Dashboard", "API Request"]
    },
    "AmazonS3" : {
      "url" : "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/S3/current/index.json",
      "file" : "S3.json",
      "productFamilies" : ["Storage","API Request"]
    },
    "AWSDataTransfer": {
      "file" : "VPC.json",
      "url" : "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonVPC/current/index.json",
      "productFamilies" : ["Data Transfer"]
    },
    "AmazonVPC" : {
      "file" : "VPC.json",
      "url" : "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonVPC/current/index.json",
      "productFamilies" : ["Cloud Connectivity", "VpcEndpoint"]
    },
    "AmazonEC2":{
      "file" : "ComputeInstance.json",
      "url" : "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/index.json",
      "productFamilies" : ["IP Address","NAT Gateway", "Storage", "System Operation", "Storage Snapshot","Compute Instance"],
      "operatingSystems" : ["RHEL", "Windows", "SUSE"],
      "instanceTypes" : []

    }
  },
  "regions": ["US East (Ohio)"]
}
config['services']['AmazonEC2']['instanceTypes'] = instanceType_filter


def init_dict():
  global aws_costs
  aws_costs = {}
  aws_costs['Region']={}

  for location in config['regions']:
    aws_costs['Region'][location] = {}
    for service,v in config['services'].items():
      try:
        x = aws_costs['Region'][location]['services']
      except KeyError:
        aws_costs['Region'][location]['services'] = {}
      try:
        x = aws_costs['Region'][location]['services'][service]
      except KeyError:
        aws_costs['Region'][location]['services'][service] = {}
      
      try:
        x = aws_costs['Region'][location]['services'][service]['Product Family']
      except KeyError:
        aws_costs['Region'][location]['services'][service]['Product Family'] = {}

      for family in config['services'][service]['productFamilies']:
        try:
          x = aws_costs['Region'][location]['services'][service]['Product Family'][family]
        except KeyError:
          aws_costs['Region'][location]['services'][service]['Product Family'][family] = {}
      
          if (service == "AmazonEC2" and family == "Compute Instance"):
            try:
              x = aws_costs['Region'][location]['services'][service]['Product Family'][family]['Operating Systems']
            except KeyError:
              aws_costs['Region'][location]['services'][service]['Product Family'][family]['Operating Systems'] = {}
            for operatingSystem in config['services'][service]['operatingSystems']:
              try:
                x = aws_costs['Region'][location]['services'][service]['Product Family'][family]['Operating Systems'][operatingSystem]
              except KeyError:
                aws_costs['Region'][location]['services'][service]['Product Family'][family]['Operating Systems'][operatingSystem] = {}
              for instanceType in config['services']['AmazonEC2']['instanceTypes']:
                aws_costs['Region'][location]['services'][service]['Product Family'][family]['Operating Systems'][operatingSystem][instanceType]={}



####
hourlyTermCode = 'JRTCKXETXF'
hourlyRateCode = '6YS6EN2CT7'
upfrontFeeCode = '2TG2D8R56U'
### 1YR Standard
RI1YRNoUpfrontStdTermCode = '4NA7Y494T4'
RI1YRPartialUpfrontStdTermCode = 'HU7G6KETJZ'
RI1YRAllUpfrontStdTermCode = '6QCMYABX3D'
### 1 Year Convertible
RI1YRNoUpfrontConvTermCode = '7NE97W5U4E'
RI1YRPartialUpfrontConvTermCode = 'CUZHX8X6JH'
RI1YRAllUpfrontConvTermCode = 'VJWZNREJX2'
### 3 Year Standard
RI3YRNoUpfrontStdTermCode = 'BPH4J8HBKS'
RI3YRPartialUpfrontStdTermCode = '38NPMPTW36'
RI3YRAllUpfrontStdTermCode = 'NQ3QZPMQV9'
### 3 Year Convertible
RI3YRNoUpfrontConvTermCode = 'Z2E3P23VKM'
RI3YRPartialUpfrontConvTermCode = 'R5XV2EPZQZ'
RI3YRAllUpfrontConvTermCode = 'MZU6U2429S'
####



def return_aws_network_costs(vpc_products,ec2_products):
  for sku, product in ec2_products.items():

    #Data Transfer OUT From Amazon EC2 To Internet
    ##Up to 1 GB / Month 8EEUB22XNJ
    if (product['productFamily'] == "Data Transfer"):
      if (product['attributes']['transferType'] == "AWS Outbound" and
          product['attributes']['fromLocation'] in location_filter and
          product['attributes']['toLocation'] == "External"):
            aws_costs['Region'][product['attributes']['fromLocation']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Outgoing traffic cost per GB up to 1 GB / Month'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.8EEUB22XNJ']['pricePerUnit']['USD']))

      ##Next 9.999 TB / Month WVV8R9FH29
      if (product['attributes']['transferType'] == "AWS Outbound" and
          product['attributes']['fromLocation'] in location_filter and
          product['attributes']['toLocation'] == "External"):
            aws_costs['Region'][product['attributes']['fromLocation']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Outgoing traffic cost per GB up to 10 GB / Month'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.WVV8R9FH29']['pricePerUnit']['USD']))

      ##Next 40 TB / Month VF6T3GAUKQ
      if (product['attributes']['transferType'] == "AWS Outbound" and
          product['attributes']['fromLocation'] in location_filter and
          product['attributes']['toLocation'] == "External"):
            aws_costs['Region'][product['attributes']['fromLocation']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Outgoing traffic cost per GB up to 40 TB / Month'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.VF6T3GAUKQ']['pricePerUnit']['USD']))

      ##Next 100 TB / Month N9EW5UVVPA
      if (product['attributes']['transferType'] == "AWS Outbound" and
          product['attributes']['fromLocation'] in location_filter and
          product['attributes']['toLocation'] == "External"):
            aws_costs['Region'][product['attributes']['fromLocation']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Outgoing traffic cost per GB up to 100 TB / Month'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.N9EW5UVVPA']['pricePerUnit']['USD']))

      ##Greater than 150 TB / Month GPHXDESFBB
      if (product['attributes']['transferType'] == "AWS Outbound" and
          product['attributes']['fromLocation'] in location_filter and
          product['attributes']['toLocation'] == "External"):
            aws_costs['Region'][product['attributes']['fromLocation']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Outgoing traffic cost per GB grater than 150 TB / Month'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.GPHXDESFBB']['pricePerUnit']['USD']))

      #All data transfer in and out within the region	
      if (product['attributes']['transferType'] == "IntraRegion" and
          product['attributes']['fromLocation'] in location_filter and
          product['attributes']['toLocation'] in location_filter):
            aws_costs['Region'][product['attributes']['fromLocation']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['All data transfer in and out within the region per GB'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))

    if (product['productFamily'] == "IP Address"):

      #Elastic IP address
      if (product['attributes']['group'] == "ElasticIP:Address" and
          product['attributes']['location'] in location_filter):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Elastic IP address not attached to a running instance for the first hour'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.8EEUB22XNJ']['pricePerUnit']['USD']))
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Elastic IP address not attached to a running instance per hour (prorated)'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.JTU8TKNAMW']['pricePerUnit']['USD']))

      #Additional Elastic IP address    
      if (product['attributes']['group'] == "ElasticIP:AdditionalAddress" and
          product['attributes']['location'] in location_filter):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Additional Elastic IP address attached to a running instance per hour (prorated)'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))

    #NAT Gateway
    if (product['productFamily'] == "NAT Gateway" and
        product['attributes']['location'] in location_filter and
        product['attributes']['operation'] == "NatGateway"):
          if (product['attributes']['groupDescription'] ==  "Hourly charge for NAT Gateways"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['NAT Gateway per hour'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))
          if (product['attributes']['groupDescription'] ==  "Charge for per GB data processed by NatGateways"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Nat Gateway per GB'] = \
            str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))

  for sku, product in vpc_products.items():

    #VPN Connection
    if (product['attributes']['operation'] == "CreateVpnConnection" and 
        product['attributes']['location'] in location_filter):
          if (product['productFamily'] == "Cloud Connectivity"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['VPN Connection per hour'] = \
            str(float(vpc_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))

    #VPC Endpoint
    if (product['attributes']['operation'] == "VpcEndpoint" and 
        product['attributes']['location'] in location_filter):
      if (product['attributes']['group'] == "VPCE:VpcEndpoint" and
          product['productFamily'] == "VpcEndpoint" and
          product['attributes']['location'] in location_filter and
          product['attributes']['groupDescription'] == "Hourly charge for VPC Endpoints"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['VPC Endpoint per hour'] = \
            str(float(vpc_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))
      if (product['attributes']['location'] in location_filter and
          product['attributes']['groupDescription'] == "Charge for per GB data processed by VPC Endpoints"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Data Processed by VPC Endpoints per GB'] = \
            str(float(vpc_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))



## CLOUDWATCH ##
def return_cw_costs(cw_products):
  for sku, product in cw_products.items():

    # CloudWatch Metrics
    if (product['productFamily'] == "Metric" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "CloudWatch Custom Metrics"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['First 10.000 Metrics / Metric / Month'] = \
            str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'A6VD9GXV7W']['pricePerUnit']['USD'])
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Next 240.000 Metrics / Metric / Month'] = \
            str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'FG3KHZ79QN']['pricePerUnit']['USD'])
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Next 750.000 Metrics / Metric / Month'] = \
            str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '95SDKRWVHH']['pricePerUnit']['USD'])
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Over 1.000.000 Metrics / Metric / Month'] = \
            str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'VNJQCFJPS4']['pricePerUnit']['USD'])

    #CloudWatch APIs
    if (product['productFamily'] == "API Request" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "CloudWatch Bulk API Requests"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Metrics requested using GetMetricData / 1000'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

    #CloudWatch Dashboards
    if (product['productFamily'] == "Dashboard" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['groupDescription'] == "CloudWatch Dashboards" and
        product['attributes']['version'] == "Extended"):
          for location in config['regions']: 
            aws_costs['Region'][location]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Dashboard / month'] = \
            str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

    #CloudWatch Alarms
    if (product['productFamily'] == "Alarm" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "CloudWatch Alarms" and
        product['attributes']['alarmType'] == "Standard"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Regular (5min) / alarm'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])
    if (product['productFamily'] == "Alarm" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "CloudWatch Alarms" and
        product['attributes']['alarmType'] == "Standard"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['High Density (1 min) / alarm'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

    #CloudWatch Logs

    #Ingested
    if (product['productFamily'] == "Data Payload" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "Existing system, application, and custom log files" and
        product['attributes']['operation'] == "PutLogEvents"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Ingested logs / GB'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

    #Stored
    if (product['productFamily'] == "Storage Snapshot" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        "TimedStorage-ByteHrs" in product['attributes']['usagetype']):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Ingested logs / GB'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

    #Vended logs
    #Ingested
    if (product['productFamily'] == "Data Payload" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "Log files generated by AWS services" and
        product['attributes']['operation'] == "PutLogEvents"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Vended Ingested logs first 10TB / GB'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'Q3Z75P77EN']['pricePerUnit']['USD'])
    if (product['productFamily'] == "Data Payload" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "Log files generated by AWS services" and
        product['attributes']['operation'] == "PutLogEvents"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Vended Ingested logs first 20TB / GB'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'JM86AE7V7B']['pricePerUnit']['USD'])
    if (product['productFamily'] == "Data Payload" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "Log files generated by AWS services" and
        product['attributes']['operation'] == "PutLogEvents"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Vended Ingested logs next 20TB / GB'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'AMET9K77J3']['pricePerUnit']['USD'])
    if (product['productFamily'] == "Data Payload" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "Log files generated by AWS services" and
        product['attributes']['operation'] == "PutLogEvents"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Vended Ingested logs over 50TB / GB'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'AQTKZ3QBDA']['pricePerUnit']['USD'])
    if (product['productFamily'] == "Storage Snapshot" and
        product['attributes']['servicecode'] == "AmazonCloudWatch" and
        product['attributes']['location'] in location_filter and
        "TimedStorage-ByteHrs" in product['attributes']['usagetype']):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Vended ingested logs / GB'] = \
          str(cw_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

    #CloudWatch Events
    #did not find any


def return_s3_costs(s3_products):
    for sku, product in s3_products.items():
      if (product['productFamily'] == "Storage" and
          product['attributes']['location'] in location_filter and
          product['attributes']['storageClass'] == "General Purpose" and
          product['attributes']['volumeType'] == "Standard"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard Storage First 50 TB / Month / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'PGHJ3S3EYE']['pricePerUnit']['USD'])
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard Storage Next 450 TB / Month / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'D42MF2PVJS']['pricePerUnit']['USD'])
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard Storage Over 500 TB / Month / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + 'PXJDJ3YRG3']['pricePerUnit']['USD'])

      if (product['productFamily'] == "Storage" and
          product['attributes']['location'] in location_filter and
          product['attributes']['storageClass'] == "Infrequent Access" and
          product['attributes']['volumeType'] == "Standard - Infrequent Access"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard-Infrequent Access (S3 Standard-IA) Storage / Month / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "Storage" and
          product['attributes']['location'] in location_filter and
          product['attributes']['storageClass'] == "Infrequent Access" and
          product['attributes']['volumeType'] == "One Zone - Infrequent Access"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['One Zone-Infrequent Access (S3 One Zone-IA) Storage / Month / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "Storage" and
          product['attributes']['location'] in location_filter and
          product['attributes']['storageClass'] == "Archive" and
          product['attributes']['volumeType'] == "Amazon Glacier"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Amazon Glacier Storage Storage / Month / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-Select-Returned'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Data Returned by S3 Select / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])
     
      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == "S3-API-Select-Scanned"):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Data Scanned by S3 / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-Tier1'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['PUT, COPY, POST, or LIST Requests / 1000 Requests'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-Tier2'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['GET, SELECT and all other Requests / 1000 Requests'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-Tier2'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Lifecycle Transition Requests into Standard – Infrequent Access or One Zone - Infrequent Access / 1000 Requests'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      #S3 Standard-Infrequent Access
      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-SIA-Retrieval'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard-Infrequent Data Retrieved / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-SIA-Select-Returned'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard-Infrequent Data Returned by S3 Select / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-SIA-Select-Returned'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard-Infrequent Data Scanned by S3 / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-SIA-Tier1'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard-Infrequent PUT, COPY, POST, or LIST Requests / 1000 Requests'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-SIA-Tier2'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Standard-Infrequent GET, SELECT and all other Requests / 1000 Requests'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])


      #S3 One Zone-Infrequent Access
      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-ZIA-Retrieval'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['One Zone-Infrequent Data Retrieved / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-ZIA-Select-Returned'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['One Zone-Infrequent Data Returned by S3 Select / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-ZIA-Select-Scanned'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['One Zone-Infrequent Data Scanned by S3 / GB'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-ZIA-Tier1'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['One Zone-Infrequent PUT, COPY, POST, or LIST Requests / 1000 Requests'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

      if (product['productFamily'] == "API Request" and
          product['attributes']['location'] in location_filter and
          product['attributes']['group'] == 'S3-API-ZIA-Tier2'):
            aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['One Zone-Infrequent GET, SELECT and all other Requests / 1000 Requests'] = \
            str(s3_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])

def return_ebs_costs(ec2_products):
  for sku, product in ec2_products.items():
    if (product['productFamily'] == 'Storage' and
        product['attributes']['location'] in location_filter and
        product['attributes']['storageMedia'] == "SSD-backed" and
        product['attributes']['servicecode'] == "AmazonEC2" and
        product['attributes']['volumeType'] == "General Purpose"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Amazon EBS General Purpose SSD (gp2) volumes per GB-month of provisioned storage'] = \
          str(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])
    if (product['productFamily'] == 'Storage' and
        product['attributes']['location'] in location_filter and
        product['attributes']['storageMedia'] == "SSD-backed" and
        product['attributes']['servicecode'] == "AmazonEC2" and
        product['attributes']['volumeType'] == "Provisioned IOPS"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Amazon EBS Provisioned IOPS SSD (io1) volumes per GB-month of provisioned storage'] = \
          str(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])
    if (product['productFamily'] == 'System Operation' and
        product['attributes']['location'] in location_filter and
        product['attributes']['groupDescription'] == "IOPS" and
        product['attributes']['servicecode'] == "AmazonEC2"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Amazon EBS Provisioned IOPS SSD (io1) volumes per provisioned IOPS-month'] = \
          str(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])
    if (product['productFamily'] == 'Storage' and
        product['attributes']['location'] in location_filter and
        product['attributes']['storageMedia'] == "HDD-backed" and
        product['attributes']['servicecode'] == "AmazonEC2" and
        product['attributes']['volumeType'] == "Throughput Optimized HDD"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Amazon EBS Throughput Optimized HDD (st1) volumes per GB-month of provisioned storage'] = \
          str(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])
    if (product['productFamily'] == 'Storage' and
        product['attributes']['location'] in location_filter and
        product['attributes']['storageMedia'] == "HDD-backed" and
        product['attributes']['servicecode'] == "AmazonEC2" and
        product['attributes']['volumeType'] == "Cold HDD"):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Amazon EBS Cold HDD (sc1) volumes per GB-month of provisioned storage'] = \
          str(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])
    if (product['productFamily'] == 'Storage Snapshot' and
        product['attributes']['location'] in location_filter and
        product['attributes']['storageMedia'] == "Amazon S3" and
        "EBS:SnapshotUsage" in product['attributes']['usagetype']):
          aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Amazon EBS Snapshots to Amazon S3 per GB-month of data stored'] = \
          str(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + '6YS6EN2CT7']['pricePerUnit']['USD'])


def return_aws_costs(ec2_products):
  for sku, product in ec2_products.items():
    if (#sku == '7EDGJEQJFK5PT95F' and
        product['productFamily'] == 'Compute Instance' and
        product['attributes']['tenancy'] == 'Shared' and
        product['attributes']['preInstalledSw'] == 'NA' and
        product['attributes']['locationType'] == 'AWS Region' and
        product['attributes']['licenseModel'] != 'Bring your own license' and
        product['attributes']['currentGeneration'] == 'Yes' and
        product['attributes']['location'] in location_filter and
        product['attributes']['operatingSystem'] in operatingSystem_filter and
        product['attributes']['instanceType'] in instanceType_filter):

        aws_costs['Region'][product['attributes']['location']]['services'][product['attributes']['servicecode']]['Product Family'][product['productFamily']]['Operating Systems'][product['attributes']['operatingSystem']][product['attributes']['instanceType']][sku]=ec2_return_prices(sku)
  return aws_costs

def ec2_compose_locations(location_filter):
  loc_dict = {}
  for location in location_filter:
    loc_dict[location] = {[]}
  return loc_dict

## ec2_return_prices returns the prices for a ec2 for a give SKU
def ec2_return_prices(sku):
  prices = {}
  prices['hourly_price'] = str(float(ec2_data['terms']['OnDemand'][sku][sku + '.' + hourlyTermCode]['priceDimensions'][sku + '.' + hourlyTermCode + '.' + hourlyRateCode]['pricePerUnit']['USD']))
  prices['hourly_1y_std_nu'] = ec2_url_for_ri(sku,RI1YRNoUpfrontStdTermCode,hourlyRateCode)
  prices['hourly_1y_std_pu'] = ec2_url_for_ri(sku,RI1YRPartialUpfrontStdTermCode,hourlyRateCode)
  prices['fee_1y_std_pu'] = ec2_url_for_ri(sku,RI1YRPartialUpfrontStdTermCode,upfrontFeeCode)
  prices['hourly_1y_std_au'] = ec2_url_for_ri(sku,RI1YRAllUpfrontStdTermCode,hourlyRateCode)
  prices['fee_1y_std_au'] = ec2_url_for_ri(sku,RI1YRAllUpfrontStdTermCode,upfrontFeeCode)
  prices['hourly_1y_conv_nu'] = ec2_url_for_ri(sku,RI1YRNoUpfrontConvTermCode,hourlyRateCode)
  prices['hourly_1y_conv_pu'] = ec2_url_for_ri(sku,RI1YRPartialUpfrontConvTermCode,hourlyRateCode)
  prices['fee_1y_conv_pu'] = ec2_url_for_ri(sku,RI1YRPartialUpfrontConvTermCode,upfrontFeeCode)
  prices['hourly_1y_conv_au'] = ec2_url_for_ri(sku,RI1YRAllUpfrontConvTermCode,hourlyRateCode)
  prices['fee_1y_conv_au'] = ec2_url_for_ri(sku,RI1YRAllUpfrontConvTermCode,upfrontFeeCode)

  prices['hourly_3y_std_nu'] = ec2_url_for_ri(sku,RI3YRNoUpfrontStdTermCode,hourlyRateCode)
  prices['hourly_3y_std_pu'] = ec2_url_for_ri(sku,RI3YRPartialUpfrontStdTermCode,hourlyRateCode)
  prices['fee_3y_std_pu'] = ec2_url_for_ri(sku,RI3YRPartialUpfrontStdTermCode,upfrontFeeCode)
  prices['hourly_3y_std_au'] = ec2_url_for_ri(sku,RI3YRAllUpfrontStdTermCode,hourlyRateCode)
  prices['fee_3y_std_au'] = ec2_url_for_ri(sku,RI3YRAllUpfrontStdTermCode,upfrontFeeCode)
  prices['hourly_3y_conv_nu'] = ec2_url_for_ri(sku,RI3YRNoUpfrontConvTermCode,hourlyRateCode)
  prices['hourly_3y_conv_pu'] = ec2_url_for_ri(sku,RI3YRPartialUpfrontConvTermCode,hourlyRateCode)
  prices['fee_3y_conv_pu'] = ec2_url_for_ri(sku,RI3YRPartialUpfrontConvTermCode,upfrontFeeCode)
  prices['hourly_3y_conv_au'] = ec2_url_for_ri(sku,RI3YRAllUpfrontConvTermCode,hourlyRateCode)
  prices['fee_3y_conv_au'] = ec2_url_for_ri(sku,RI3YRAllUpfrontConvTermCode,upfrontFeeCode)
  
  return prices


## This function was created to avoid doing this concatenation several times in ec2_return_prices
def ec2_url_for_ri(sku,termcode,costcode):
  try:
    cost = str(float(ec2_data['terms']['Reserved'][sku][sku + '.' + termcode]['priceDimensions'][sku + '.' + termcode + '.' + costcode]['pricePerUnit']['USD']))
  except KeyError:
    cost = "NA"
  return cost

#aws_costs['Region'][location]['Product Family'][productFamily][vpc_product]
#aws_costs['Region'][location]['Product Family']['Compute Instance']['Operating Systems'][operatingSystem][instanceType]

def ec2_pretty_print(aws_costs):
  open('aws_costs.md','w').close()
  with open('aws_costs.md','w') as file:
    file.write('# Amazon Web Services Costs'+'\n')

    file.write('# Table of contents\n')
    for region,v in aws_costs.items(): #['Region']
      for region_value,v in v.items(): #region value
        file.write('[**'+region_value+'**](#'+region_value+') \n\n')
        for service_value,v in v['services'].items(): #Service name
          #print(service_value)
          file.write('[**Service : '+service_value+'**](#'+service_value+') \n\n')
          for pf_value,v in v['Product Family'].items(): #Service name

            if (pf_value == "Compute Instance"):
              for os_value,v in v['Operating Systems'].items():
                count = 0
                file.write('\n\n[**Operating System : '+os_value+'**](#'+os_value+') \n\n')
                #print("    "+os_value)
                for instance_value,v in v.items():
                  if v:
                    file.write('['+instance_value+'](#'+instance_value+') ')
                    #print("      "+instance_value)
                    if (count == 4):
                      count = 0
                      file.write('\n\n')
                    elif (count == 3):
                      count+=1
                    elif (count == 2):
                      count+=1
                    elif (count == 1):
                      count+=1
                    elif (count == 0):
                      count+=1
            else:
              #print("  "+pf_value)
              file.write('[**Product Family : '+pf_value+'**](#'+pf_value+') \n\n')
            # if (pf_value == 'Data Transfer'):
            #   file.write('[**Product Family : '+pf_value+'**](#'+pf_value+') \n\n')
            # if (pf_value == 'IP Address'):
            #   file.write('[**Product Family : '+pf_value+'**](#'+pf_value+') \n\n')
            # if (pf_value == 'VpcEndpoint'):
            #   file.write('[**Product Family : '+pf_value+'**](#'+pf_value+') \n\n')
            # if (pf_value == 'Cloud Connectivity'):
            #   file.write('[**Product Family : '+pf_value+'**](#'+pf_value+') \n\n')
            # if (pf_value == 'NAT Gateway'):
            #   file.write('[**Product Family : '+pf_value+'**](#'+pf_value+') \n\n')
            # if (pf_value == 'Storage'):
            #   file.write('[**Product Family : '+pf_value+' (S3)**](#'+pf_value+') \n\n')
            # if (pf_value == 'API Request'):
            #   file.write('[**Product Family : '+pf_value+' (S3)**](#'+pf_value+') \n\n')

                  
    for region,v in aws_costs.items(): #['Region']
      for region_value,v in v.items(): #region value
        file.write('\n# <a name="'+region_value+'"></a> Region: '+region_value+'\n')
        for service_value,v in v['services'].items(): #Service name
          file.write('\n# <a name="'+service_value+'"></a> Service: '+service_value+'\n')
          for pf_value,v in v['Product Family'].items(): #product family value
            if (pf_value == 'Compute Instance'):
              file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n')
              for os_value,v in v['Operating Systems'].items():
                #print(os_value)
                file.write('#### <a name="'+os_value+'"></a> OS: '+os_value+"\n")
                for instance_value,v in v.items():
                  if v:
                    file.write('##### <a name="'+instance_value+'"></a> Instance Type: '+instance_value+"\n")
                    #print (instance_value)
                    for k,v in v.items():
                      
                      file.write("###### Standard 1 Year Term: \n")
                      file.write("Payment Option  |Upfront                  |Hourly                    |On-Demand Hourly\n")
                      file.write("---             | ---                     |  ---                     | ----\n")
                      file.write("No Upfront      |0                        |"+v['hourly_1y_std_nu']+" |"+v['hourly_price']+"\n")
                      file.write("Partial Upfront |"+v['fee_1y_std_pu']+"   |"+v['hourly_1y_std_pu']+" |^  \n")
                      file.write("Full Upfront    |"+v['fee_1y_std_au']+"   |"+v['hourly_1y_std_au']+" |^ \n")

                      file.write("###### Convertible 1 Year Term: \n")
                      file.write("Payment Option  |Upfront                  |Hourly                    |On-Demand Hourly\n")
                      file.write("---             | ---                     |  ---                     | ----\n")
                      file.write("No Upfront      |0                        |"+v['hourly_1y_conv_nu']+" |"+v['hourly_price']+"\n")
                      file.write("Partial Upfront |"+v['fee_1y_conv_pu']+"   |"+v['hourly_1y_conv_pu']+" |^  \n")
                      file.write("Full Upfront    |"+v['fee_1y_conv_au']+"   |"+v['hourly_1y_conv_au']+" |^ \n")

                      file.write("###### Standard 3 Year Term: \n")
                      file.write("Payment Option  |Upfront                  |Hourly                    |On-Demand Hourly\n")
                      file.write("---             | ---                     |  ---                     | ----\n")
                      file.write("No Upfront      |0                        |"+v['hourly_3y_std_nu']+" |"+v['hourly_price']+"\n")
                      file.write("Partial Upfront |"+v['fee_3y_std_pu']+"   |"+v['hourly_3y_std_pu']+" |^  \n")
                      file.write("Full Upfront    |"+v['fee_3y_std_au']+"   |"+v['hourly_3y_std_au']+" |^ \n")

                      file.write("###### Convertible 3 Year Term: \n")
                      file.write("Payment Option  |Upfront                   |Hourly                    |On-Demand Hourly\n")
                      file.write("---             | ---                      |  ---                     | ----\n")
                      file.write("No Upfront      |0                         |"+v['hourly_3y_conv_nu']+" |"+v['hourly_price']+"\n")
                      file.write("Partial Upfront |"+v['fee_3y_conv_pu']+"   |"+v['hourly_3y_conv_pu']+" |^  \n")
                      file.write("Full Upfront    |"+v['fee_3y_conv_au']+"   |"+v['hourly_3y_conv_au']+" |^ \n")
                      #print(v['hourly_price'])

#aws_costs['Region'][location]['Product Family'][productFamily][vpc_product]
            elif (pf_value == 'Data Transfer'):
              #print(v)
              file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
              file.write('#### Data Transfer IN To Amazon EC2 From Internet\n\n')
              file.write('`All data transfer in per GB 0.00 `\n\n')
              file.write('#### Data Transfer OUT From Amazon EC2 From Internet\n\n')
              print(pf_value)
              for k,v in v.items():
                print(k)
                file.write("`"+k+" "+v+'`\n\n')
            elif (pf_value == 'IntraRegion'):
              file.write('#### Data Transfer Across AZ within this Region\n')
              for k,v in v.items():
                #print(k)
                file.write("`"+k+" "+v+'`\n\n')


            # elif (pf_value == 'IP Address'):
            #   file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
            #   for k,v in v.items():
            #     file.write("`"+k+" "+v+'`\n\n')
          
            # elif (pf_value == 'Cloud Connectivity'):
            #   file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
            #   for k,v in v.items():
            #     file.write("`"+k+" "+v+'`\n\n')

            # elif (pf_value == 'VpcEndpoint'):
            #   file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
            #   for k,v in v.items():
            #     file.write("`"+k+" "+v+'`\n\n')

            # elif (pf_value == 'NAT Gateway'):
            #   file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
            #   for k,v in v.items():
            #     file.write("`"+k+" "+v+'`\n\n')

            # elif (pf_value == 'Storage'):
            #   file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
            #   for k,v in v.items():
            #     file.write("`"+k+" "+v+'`\n\n')

            # elif (pf_value == 'API Request'):
            #   file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
            #   for k,v in v.items():
            #     file.write("`"+k+" "+v+'`\n\n')

            else:
              file.write('### <a name="'+pf_value+'"></a> Product Family : '+pf_value+'\n\n')
              #print(service_value)
              #print(pf_value)
              for k,v in v.items():
                #print("else "+k+" "+v)
                file.write("`"+k+" "+v+'`\n\n')



def main():
  init_dict()
  return_aws_costs(ec2_products)
  # return_aws_network_costs(vpc_products,ec2_products)
  # return_s3_costs(s3_products)
  # return_cw_costs(cw_products)
  # return_ebs_costs(ec2_products)
  ec2_pretty_print(aws_costs)

if __name__ == '__main__':
  main()