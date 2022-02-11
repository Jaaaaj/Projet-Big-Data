from contextlib import nullcontext
from genericpath import exists
from pip import main
import boto3
import os
import sys
import paramiko
from configparser import ConfigParser
from botocore.exceptions import ClientError
import time
import csv, json, datetime
from pymongo import MongoClient


CONFIG_PATH = "config.ini"
config_object = ConfigParser()
config_object.read(CONFIG_PATH)


def read_config(title, id):
    userinfo = config_object[title]
    return format(userinfo[id])


# Verification si l'instance existe
def instance_exists():
    list_instances = EC2_CLIENT.describe_instances()

    if not read_config("EC2", "id"):
        print("Pas d'instance détecté dans le fichier de configuration")
        return False

    for instance in list_instances["Reservations"]:
        if instance["Instances"][0]["InstanceId"] == read_config("EC2", "id"):
            print("L'instance ec2 " + read_config("EC2", "id") + " existe déjà")
            return True
    
    print("L'instance ec2 n'existe pas")
    return False


# Creation d'une paire de clés ssh
def create_key_pair():
    EC2_CLIENT.delete_key_pair(KeyName=read_config("EC2", "keyname"))
    key_pair = EC2_CLIENT.create_key_pair(KeyName=read_config("EC2", "keyname"))

    private_key = key_pair["KeyMaterial"]

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    # write private key to file with 400 permissions
    with os.fdopen(os.open("./tmp/aws_ec2_key.pem", os.O_WRONLY | os.O_CREAT, 0o400), "w+") as handle:
        handle.write(private_key)


# Creation de l'instance ec2
def create_instance():
    EC2_RESOURCE = boto3.resource('ec2', region_name="us-east-1")
    instances = EC2_RESOURCE.create_instances(
        ImageId="ami-0a8b4cd432b1c3063",
        MinCount=1,
        MaxCount=1,
        InstanceType="t2.micro",
        KeyName=read_config("EC2", "keyname")
    )

    print(instances)

    instances[0].modify_attribute(
        Groups=[
            read_config("EC2", "sg")
        ]
    )

    config_object["EC2"]["id"] = instances[0].id    # Id de la nouvelle machine ec2

    # Ecriture de l'id de l'instance dans le fichier de config
    with open(CONFIG_PATH, 'w') as configfile:
        config_object.write(configfile)

    print(read_config("EC2", "id"))


# Récupération de l'IP de l'instance ec2
def get_public_ip():
    reservations = EC2_CLIENT.describe_instances(InstanceIds=[read_config("EC2", "id")]).get("Reservations")

    print(reservations[0]["Instances"][0].get("PublicDnsName"))

    config_object["EC2"]["ipaddr"] = format(reservations[0]["Instances"][0].get("PublicDnsName"))    # Adresse IP de la nouvelle machine ec2

    # Ecriture de l'IP de l'instance dans le fichier de config
    with open(CONFIG_PATH, 'w') as configfile:
        config_object.write(configfile)


# Creation d'un security group qui autorise les connections ssh
def create_security_groups():
    EC2_RESOURCE = boto3.resource('ec2', region_name=read_config("EC2", "region"))

    security_groups = EC2_RESOURCE.security_groups.all()

    sg_exist = False

    print('Security Groups:')
    for security_group in security_groups:
        print(f'  - Security Group {security_group.id}')
        if security_group.id == read_config("EC2", "sg"):
            sg_exist = True

    if sg_exist == False:
        sg = EC2_RESOURCE.create_security_group(GroupName='Allow remote ssh access', Description = 'ssh-access', VpcId='vpc-00c7270d7cece3692') 
        sg.authorize_ingress(
            CidrIp='0.0.0.0/0',
            FromPort=22,
            ToPort=22,
            IpProtocol='tcp',
        )
        print("Nouveau security group créé")
        print(sg.id)
    else:
        print("Le security group existe déjà")


# Connection en SSH
def ssh_connection():
    k = paramiko.RSAKey.from_private_key_file("./tmp/aws_ec2_key.pem")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(hostname=read_config("EC2", "ipaddr"), username="ec2-user", pkey=k)

    commands = [
        "sudo yum update -y",
        "curl \"https://bootstrap.pypa.io/get-pip.py\" -o \"get-pip.py\"",
        "rm -r /path/to/dir/*",
        "mv ./credentials ./.aws/credentials",
        "mv ./config ./.aws/config",
        "python3 get-pip.py",
        "./.local/bin/pip install langdetect seaborn nltk sklearn pandas numpy lime geopy transformers",
        "python3 ./send_data.py",
        "python3 ./BOGDOTO-Lite.py",
        "rm ./predict.csv"
    ]
    for command in commands:
        print("running command: {}".format(command))
        stdin , stdout, stderr = c.exec_command(command)
        print(stdout.read())
        print(stderr.read())

    c.close


# Sauvegarde du csv des résultats dans MongoDB
def sauvegarde_mongo():
    csvPathFile = './tmp/predict.csv'

    # On convertit le fichier csv au format json
    data2 = []
    with open(csvPathFile) as csvFile:
        csvReader = csv.DictReader(csvFile)
        
        for row in csvReader:
            data2.append(row)

    # Envoi des données dans notre base Mongo
    client = MongoClient('localhost', 27017)
    db = client[read_config("MONGODB", "database")]
    collection = db[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

    collection.create_index("predictions")
    collection.insert_many(data2)

    client.close()


# MAIN
def main():
    print("main: ")

    if not read_config("EC2", "keyname") or not read_config("EC2", "region"):
        print("Erreur: le fichier de configuration est incomplet.")
        return

    global EC2_CLIENT
    EC2_CLIENT = boto3.client("ec2", region_name=read_config("EC2", "region"))

    if not instance_exists():
        print("Création d'une nouvelle instance ec2...")
        create_key_pair()   # Create ssh keys
        create_security_groups()
        create_instance()   # Create ec2 instance

    else:
        # Start ec2 instance
        print("Démarrage de l'instance")
        try:
            EC2_CLIENT.start_instances(InstanceIds=[read_config("EC2", "id")], DryRun=False)
        except ClientError as e:
            print(e)
            return

    # On attend que l'instance soit lancée pour continuer    
    sys.stdout.write("Initialisation de l'instance..")
    sys.stdout.flush()
    
    reservations = EC2_CLIENT.describe_instances(InstanceIds=[read_config("EC2", "id")]).get("Reservations")
    while not reservations[0]["Instances"][0]["State"].get("Name") == "running":
        reservations = EC2_CLIENT.describe_instances(InstanceIds=[read_config("EC2", "id")]).get("Reservations")
        sys.stdout.write(".")
        sys.stdout.flush()
    
    print()
    get_public_ip()     # On récupère l'ip de l'instance maintenant qu'elle est lancée

    # Test paramiko
    time.sleep(10)

    os.system("scp -i %s %s ec2-user@%s:~/" % ("./tmp/aws_ec2_key.pem", "./BOGDOTO-Lite.py", read_config("EC2", "ipaddr")))  # Envoi du script de machine learning
    os.system("scp -i %s %s ec2-user@%s:~/" % ("./tmp/aws_ec2_key.pem", "./send_data.py", read_config("EC2", "ipaddr")))  # Envoi du script de machine learning
    os.system("scp -i %s %s ec2-user@%s:~/" % ("./tmp/aws_ec2_key.pem", "../33000-BORDEAUX_nettoye.csv", read_config("EC2", "ipaddr")))   # Envoi des données csv
    os.system("scp -i %s %s ec2-user@%s:~/" % ("./tmp/aws_ec2_key.pem", read_config("EC2", "awspath")+"/credentials", read_config("EC2", "ipaddr")))   # Envoi des crédentials
    os.system("scp -i %s %s ec2-user@%s:~/" % ("./tmp/aws_ec2_key.pem", read_config("EC2", "awspath")+"/config", read_config("EC2", "ipaddr")))   # Envoi du conf
    ssh_connection()
    os.system("scp -i %s ec2-user@%s:%s ./tmp/predict.csv" % ("./tmp/aws_ec2_key.pem", read_config("EC2", "ipaddr"), "~/predict.csv"))

    #  Wait
    input("Instance en cours d'exécution, appuyer sur une touche pour l'arrêter ")

    # Stop ec2 instance
    try:
        EC2_CLIENT.stop_instances(InstanceIds=[read_config("EC2", "id")], DryRun=False)
        print("L'instance " + read_config("EC2", "id") + " a bien été arrêtée")
    except ClientError as e:
        print(e)

    sauvegarde_mongo()
    os.remove("./tmp/predict.csv")
    print("Résultats sauvegardés dans MongoDB")

    print("Fin du script")


if __name__ == "__main__":
    main()