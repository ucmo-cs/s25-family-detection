import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

def embedding_calculate():
    print("Doing embedding calculation...")
    # connect to aiven vector database
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    # go through all face img files in stored-faces directory
    for filename in os.listdir("stored-faces"):
        # open image
        img = Image.open("stored-faces/" + filename)
        # load imgbedding
        ibed = imgbeddings()
        # calculate embeddings
        embedding = ibed.to_embeddings(img)
        print(embedding[0].tolist())
        curr = conn.cursor()
        curr.execute("INSERT INTO pictures values (%s, %s)", (filename, embedding[0].tolist()))
        conn.commit()
        print(filename)
    print("Done embedding calculation!")
embedding_calculate()