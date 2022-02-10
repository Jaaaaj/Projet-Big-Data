#!/bin/sh

#how it work : scp -i ./key-pair.pem ./path/to/files/ <username>@<public-ip>:/pathwhere/you/need/to/copy
scp -i path/to/key file/to/copy user@ec2-xx-xx-xxx-xxx.compute-1.amazonaws.com:path/to/file
