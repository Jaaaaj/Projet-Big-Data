# importing the package
from paramiko.client import SSHClient
import json

conf = json.load(open('conf.json'))
hadoopServer = conf["hadoopServer"]
hadoopPort = conf["hadoopPort"]
hadoopUser = conf["hadoopUser"]
hadoopPassword = conf["hadoopPassword"]
remotePathHDFS = conf["remotePathHDFS"]
remotePathHadoop = conf["remotePathHadoop"]
localPath = conf["localPath"]

client = SSHClient()
client.load_system_host_keys()
client.connect(hadoopServer,hadoopPort,hadoopUser,hadoopPassword)
client.exec_command('hadoop fs -get '+remotePathHDFS+' .')
sftp = client.open_sftp()
sftp.get(remotePathHadoop,localPath)
sftp.close()

