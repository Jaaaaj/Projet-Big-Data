# pip install pymongo

import json, csv
from pymongo import MongoClient

csvPathFile = 'test.csv'

# On convertit le fichier csv au format json
data2 = []
with open(csvPathFile) as csvFile:
    csvReader = csv.DictReader(csvFile)
    
    for row in csvReader:
        data2.append(row)

# Envoi des données dans notre base Mongo
client = MongoClient('localhost', 27017)
db = client['Big']
collection = db['Data']

collection.insert_many(data2)

client.close()