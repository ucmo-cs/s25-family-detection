import os

from PIL import Image
from imgbeddings import imgbeddings
import psycopg2
import ast
def calculate_new_pic_embedding():
    print("This is calculating new pic embedding")
    file_name = "test-photos/schizoid.jpg"
    img = Image.open(file_name)
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    print(embedding[0].tolist())
    print(type(embedding[0]))
    #print(type(embedding))
    #print(type(embedding[0]))

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    #find similar images querying postgresql database
    curr = conn.cursor()
    string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) + "]"
    #establish a threhold to compare against photos in database, if photo is under threshold then display face we think its most like
    #if photo is over threshold then tell user no faces close to that photo could be detected
    thereshold = 15
    curr.execute("SELECT * FROM pictures WHERE embedding <-> %s < %s ORDER BY embedding <-> %s LIMIT 1;", (string_representation, thereshold, string_representation))
    rows = curr.fetchall()
    if not rows:
        print("No photo of that exact face found in database, calculating closest possible relative.")
        return
    for row in rows:
        print(f"This is vector result of db query: \n{row}")
        print(f"This is data type result of db query[1], column 1: \n{type(row[1])}")

        image_path = "stored-faces/" + row[0]
        image = Image.open(image_path)
        image.show()
    conn.close()
calculate_new_pic_embedding()