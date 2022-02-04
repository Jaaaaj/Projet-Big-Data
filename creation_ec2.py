from contextlib import nullcontext
from pip import main
import boto3
import os
import sys
from configparser import ConfigParser
from botocore.exceptions import ClientError


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
    with os.fdopen(os.open("./tmp/aws_ec2_key.pem", os.O_WRONLY | os.O_CREAT, 0o777), "w+") as handle:
        handle.write(private_key)


# Creation de l'instance ec2
def create_instance():
    instances = EC2_CLIENT.run_instances(
        ImageId="ami-0a8b4cd432b1c3063",
        MinCount=1,
        MaxCount=1,
        InstanceType="t2.micro",
        KeyName=read_config("EC2", "keyname")
    )

    config_object["EC2"]["id"] = instances["Instances"][0]["InstanceId"]    # Id de la nouvelle machine ec2

    # Ecriture de l'id de l'instance dans le fichier de config
    with open(CONFIG_PATH, 'w') as configfile:
        config_object.write(configfile)

    print(read_config("EC2", "id"))


# Récupération de l'IP de l'instance ec2
def get_public_ip():
    reservations = EC2_CLIENT.describe_instances(InstanceIds=[read_config("EC2", "id")]).get("Reservations")

    print(reservations[0]["Instances"][0].get("PublicIpAddress"))

    config_object["EC2"]["ipaddr"] = format(reservations[0]["Instances"][0].get("PublicIpAddress"))    # Adresse IP de la nouvelle machine ec2

    # Ecriture de l'IP de l'instance dans le fichier de config
    with open(CONFIG_PATH, 'w') as configfile:
        config_object.write(configfile)


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
        create_instance()   # Create ec2 instance

    else:
        # Start ec2 instance
        print("Démarrage de l'instance")
        try:
            EC2_CLIENT.start_instances(InstanceIds=[read_config("EC2", "id")], DryRun=False)
        except ClientError as e:
            print(e)

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

    #  Wait
    input("Instance en cours d'exécution, appuyer sur une touche pour l'arrêter ")

    # Stop ec2 instance
    try:
        EC2_CLIENT.stop_instances(InstanceIds=[read_config("EC2", "id")], DryRun=False)
        print("L'instance " + read_config("EC2", "id") + " a bien été arrêtée")
    except ClientError as e:
        print(e)

    print("Fin du script")


if __name__ == "__main__":
    main()