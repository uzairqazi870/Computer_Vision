import argparse
import os

import numpy as np
import pandas as pd
from pathlib import Path

# -- class(label) info
# names = os.listdir("dataset")
# classes = []
# for name in names:
#   classes.append(name[:])

#print(classes)
# class_file = open("class.txt", "r")
# classes = [x.strip() for x in class_file.readlines()]
# class_file.close()

# -- path info
TRAIN_PATH = "./dataset"
TRIPLET_PATH = "triplet.csv"


def list_pictures(directory):
    return [Path(root) / f
            for root, _, files in os.walk(directory) for f in files]


def get_negative_images(all_images, image_names, num_neg_images):
    """
    Get out class images
    """
    random_numbers = np.arange(len(all_images))
    np.random.shuffle(random_numbers)
    if int(num_neg_images) > (len(all_images) - 1):
        num_neg_images = len(all_images) - 1
    neg_count = 0
    negative_images = []
    for random_number in list(random_numbers):
        if all_images[random_number] not in image_names:
            negative_images.append(all_images[random_number])
            neg_count += 1
            if neg_count > (int(num_neg_images) - 1):
                break

    return negative_images


def get_positive_images(image_name, image_names, num_pos_images):
    """
    Get in class images
    """
    random_numbers = np.arange(len(image_names))
    np.random.shuffle(random_numbers)
    if int(num_pos_images) > (len(image_names) - 1):
        num_pos_images = len(image_names) - 1
    pos_count = 0
    positive_images = []
    for random_number in list(random_numbers):
        if image_names[random_number] != image_name:
            positive_images.append(image_names[random_number])
            pos_count += 1
            if int(pos_count) > (int(num_pos_images) - 1):
                break

    return positive_images


def generate_triplets(dataset_path=TRAIN_PATH, TRIPLET_PATH=TRIPLET_PATH, num_neg_images=1, num_pos_images=1):
    """
    Generate pre-sampled triplet dataset

    Parameters
    ----------
    dataset_path: path to dataset
    triplet_path : path to triplets file needed for training
    num_neg_images: number of negative images per query image
    num_pos_images: number of positive images per query image

    Returns
    -------
    Void, setups triplet dataset in .csv file
    """
    print("Grabbing images from: " + str(dataset_path))
    print("Number of Positive image per Query image: " + str(num_pos_images))
    print("Number of Negative image per Query image: " + str(num_neg_images))

    
    # -- class(label) info
    names = os.listdir(dataset_path)
    classes = []
    for name in names:
      classes.append(name[:])

    triplet_df = pd.DataFrame(columns=["query", "positive", "negative"])

    all_images = []
    for class_ in classes:
        all_images += list_pictures(os.path.join(dataset_path, class_))
        #print(all_images)

    for class_ in classes:
        image_names = list_pictures(os.path.join(dataset_path, class_))
        for image_name in image_names:
            query_image = image_name
            positive_images = get_positive_images(image_name, image_names, num_pos_images)
            for positive_image in positive_images:
                negative_images = get_negative_images(all_images, set(image_names), num_neg_images)
                for negative_image in negative_images:
                    row = {"query": query_image,
                           "positive": positive_image,
                           "negative": negative_image}
                    print(row)
                    triplet_df = triplet_df.append(row, ignore_index=True)

    triplet_df.to_csv(TRIPLET_PATH, index=False)
    print("Sampling done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--n_pos', default="1", help='A number of Positive images per Query image')

    parser.add_argument('--n_neg', default="1", help='A number of Negative images per Query image')

    args = parser.parse_args()

    if int(args.n_neg) < 1 or int(args.n_pos) < 1:
        print('Number of Negative(Positive) Images cannot be less than 1...')
        quit()

    generate_triplets(dataset_path=TRAIN_PATH, TRIPLET_PATH=TRIPLET_PATH, num_neg_images=args.n_neg, num_pos_images=args.n_pos)
