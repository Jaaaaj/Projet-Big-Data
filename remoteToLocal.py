# importing the package
from paramiko.client import SSHClient
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





hadoopServer = read_config["HADOOP","hadoopServer"]
hadoopPort = read_config["HADOOP","hadoopPort"]
hadoopUser = read_config["HADOOP","hadoopUser"]
hadoopPassword = read_config["HADOOP","hadoopPassword"]
remotePathHDFS = read_config["HADOOP","remotePathHDFS"]
remotePathHadoop = read_config["HADOOP","remotePathHadoop"]
localPath = read_config["HADOOP","localPath"]

client = SSHClient()
client.load_system_host_keys()
client.connect(hadoopServer,hadoopPort,hadoopUser,hadoopPassword)
client.exec_command('hadoop fs -get '+remotePathHDFS+' .')
sftp = client.open_sftp()
sftp.get(remotePathHadoop,localPath)
sftp.close()