import datetime

import boto3
import time
import os
from time import strftime

from configparser import ConfigParser

#Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")


def read_config(title, id):
    userinfo = config_object[title]
    print(id + " is {}".format(userinfo[id]))
    return format(userinfo[id])


sqs = boto3.resource('sqs')
#Get the queue
#queue = sqs.get_queue_by_name(QueueName='test')


client_s3 =boto3.client(
     's3'
)
bucket_name = "stock-big-data-v2"
filename_to_upload = "data.csv"

print("file + " + filename_to_upload)
dest_filename = "data.csv"

client_s3.upload_file(
         "./" + filename_to_upload,
         bucket_name,
         dest_filename
)

