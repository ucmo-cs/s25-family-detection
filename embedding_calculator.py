import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

#connect to aiven vector database
conn = psycopg2.connect()