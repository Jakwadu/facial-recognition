import os
import io
import sqlite3
import pandas as pd
import pickle
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from face_dataset_encoder import read_and_encode_images
from argparse import ArgumentParser
from shutil import copy, rmtree

REFERENCE_DIR = 'references'


def build_parser():
    p = ArgumentParser()
    p.add_argument('-p', '--person', type=str, dest='person', help='The name of the person to be added/removed')
    p.add_argument('-a', '--add', type=str, dest='add', help='Add person to database')
    p.add_argument('-r', '--remove', dest='remove', action='store_true', help='Remove person from database')
    p.add_argument('-b', '--build-database', dest='build_database', action='store_false')
    return p


def create_new_reference(name, source_directory):
    files = os.listdir(source_directory)
    img_files = [os.path.join(source_directory, f) for f in files if f[-4:] == '.jpg']
    assert len(img_files) > 0, 'No .jpg files found in the specified directory'
    new_directory = os.path.join(REFERENCE_DIR, name)
    os.makedirs(new_directory)
    for img_file in img_files:
        f = os.path.basename(img_file)
        copy(img_file, os.path.join(new_directory, f))
    print(f'{name} successfully added to the database')


def add_person_to_database(name, source_directory):
    if os.path.exists(os.path.join(REFERENCE_DIR, name)):
        answer = input(f'{name} already exists in the database. Do yo want to replace the old entry?[y/n]:')
        check = [answer == char for char in ['y', 'Y', 'n', 'N']]
        if not any(check):
            add_person_to_database(name, source_directory)
        else:
            if answer == 'n' or answer == 'N':
                exit()
            else:
                remove_person_from_references(name)
                create_new_reference(name, source_directory)
    else:
        create_new_reference(name, source_directory)


def remove_person_from_references(name):
    path = os.path.join(REFERENCE_DIR, name)
    assert os.path.exists(path), f'{name} was not found. It may have already been removed'
    rmtree(path)
    print(f'{name} has been removed from the database')


def list_people_in_references():
    people = os.listdir(REFERENCE_DIR)
    space = ' '*30
    header0 = 'Name'
    header1 = 'Number of Reference Faces'
    print('\nPeople listed in database:')
    print(header0 + space[:-len(header0)] + header1)
    print('-'*(len(header0) + len(space) + len(header1)))
    for p in people:
        name = p + space[:-len(p)]
        n_faces = len(os.listdir(os.path.join(REFERENCE_DIR, p)))
        print(name + str(n_faces))


def create_database_from_references():
    db_connection = sqlite3.connect('face_embeddings.db')
    face_detector = FaceDetector(n_faces=1, face_size=(224, 224))
    face_encoder = FaceEncoder(encoder_type='vgg-face')
    dataframes = []
    for name in os.listdir(REFERENCE_DIR):
        embeddings = read_and_encode_images(os.path.join(REFERENCE_DIR, name), face_detector, face_encoder)
        for embedding in embeddings:
            dataframes.append(pd.DataFrame({'name': [name], 'embedding': [embedding.tobytes()]}))
    df = pd.concat(dataframes)
    print(df.head())
    df.to_sql('face_embeddings', con=db_connection, index=False, if_exists='replace')


if __name__ == '__main__':
    arg_parser = build_parser()
    arguments = arg_parser.parse_args()
    person = arguments.person
    if arguments.add:
        assert person, 'Name of person to be added not specified'
        src_directory = arguments.add
        add_person_to_database(person, src_directory)
    elif arguments.remove:
        assert person, 'Name of person to be removed not specified'
        remove_person_from_references(person)
    elif arguments.build_database:
        create_database_from_references()
    else:
        list_people_in_references()
