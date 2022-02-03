
import boto3
import time
sqs = boto3.resource('sqs')
# Get the queue
sqs.create_queue(QueueName='send_file')
sqs.create_queue(QueueName='reponse')
queue = sqs.get_queue_by_name(QueueName='send_file')
queue2 = sqs.get_queue_by_name(QueueName='reponse')
queue.purge()
queue2.purge()


