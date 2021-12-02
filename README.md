# Facial Recognition

A simple project exploring facial recognition using OpenCV and Tensorflow 2 with Keras. A special thanks to [Sefik 
Serengil](https://github.com/serengil) for his [deeepface](https://github.com/serengil/deepface) repo, which inspired 
this effort. The necessary dependencies can be installed using ```pip install -r requirements.txt``` in the cloned repo 
directory.

## System Architecture
The system consists of the following major components:
* A face detector
* A face encoder
* A similarity search through a repository of known faces to enable recognition

### Face Detection
OpenCV's Haar Cascade detector is used for face detection by default, with MTCNN as an alternative. The Haar Cascade is
preferred for real time detection (especially if running on a laptop or personal PC), but I found MTCNN to be more 
robust across various lighting conditions.

### Face Encoding
This is achieved using the vgg-face model. The weights can be found in Sefik's repo of 
[pretrained weights](https://github.com/serengil/deepface_models/releases).

### Similarity Search
Once a face has been detected and encoded into a vector representation, it is then compared to a collection of reference 
faces using a linear search with a Euclidean distance threshold. The simplest setup is to create and populate 
```./references``` in this repo using a structure like the one shown below.
```
references
    |
    |__Person_A
    |   |__sijvbiu.jpg
    |   |__fjvbked.jpg
    |   |__bvjuhhd.jpg
    |
    |__Person_B
        |__7f84gfb.jpg
        |__cbge77d.jpg
        |__vkjffnn.jpg

```
In the above example 'Person_A' and 'Person_B' are the names of people of interest. This can be done programmatically
using ```manage_reference_faces.py```.
```
manage_reference_faces.py --person Joe_Blogs --add /path/to/folder/with/example/faces
```
People can also be removed.
```
manage_reference_faces.py --person Joe_Blogs --remove
```

Alternatively, references can be converted into a SQLite database of face embeddings.
*Note: ```./references``` must have already been populated in order to perform this operation.*
```
manage_reference_faces.py --build-database
```
This will produce the SQLite database **face_embeddings.db** with a single table called **face_embeddings**. The facial
recogniser will default to using the database if it has been created.

## Facial Recognition Use Cases
Webcam Video Stream 
```
main.py
```
Image
```
main.py --image /path/to/image
```
Image with specific Location for reference faces (this will override the use of the SQLIte database if it exists).
```
main.py --references /path/to/references
```
Server-Client configuration

*Note: By default the client will try to connect to the server on 127.0.0.1:8000 unless specified as shown below.*

**Server**
```
server.py --ip IP_ADDRESS --port PORT
```
**Client with video**
```
client.py --ip IP_ADDRESS --port PORT
```
**Client with video and client side visualisation**
```
client.py --ip IP_ADDRESS --port PORT --show-image
```
**Client with image**
```
client.py --ip IP_ADDRESS --port PORT --face /path/to/image/with/face
```
**Client with image and client side visualisation**
```
client.py --ip IP_ADDRESS --port PORT --face /path/to/image/with/face --show-image
```
