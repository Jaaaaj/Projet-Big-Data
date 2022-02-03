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
bucket_name =read_config("BUCKET","bucketname")
filename_to_upload =read_config("FILE","filename")

print("file + " + filename_to_upload)
dest_filename = str(read_config("FILE","filename"))

client_s3.upload_file(
         "D:/FISE 3/Projet BIG DATA/script/"+filename_to_upload,
         bucket_name,
         dest_filename
)

