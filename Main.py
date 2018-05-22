from RecognitionFace import build_data_set, face_recognition

if __name__ == '__main__':
    print("Starting taking photos")

    build_data_set()

    print("Finshed taking photos")

    print("Starting to Calculate the dimension reduction matrix")
    # The algorithm
    face_recognition()

