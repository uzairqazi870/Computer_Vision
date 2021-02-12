import os
import numpy as np
import pandas as pd
from PIL import Image
#import faiss
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from net import DeepRank
from utils import euclidean_distance, data_transforms, DatasetImageNet


# -- path info
# TRIPLET_PATH = 'triplet.csv'
# MODEL_PATH = 'model/deeprank17.pt'
# EMBEDDING_PATH = 'embedding.txt'


class Prediction:
    def __init__(self, model_path='model/deeprank17.pt', embedding_path='embedding.txt', triplet_path='triplet.csv'):
        self.model = DeepRank()
        self.model.load_state_dict(torch.load(MODEL_PATH)) # load model parameters
        self.train_df = pd.read_csv(TRIPLET_PATH).drop_duplicates('query', keep='first').reset_index(drop=True)

        # check embedding
        if not os.path.exists(EMBEDDING_PATH):
            print('pre-generated [embedding.txt] not exist!')
            self.embedding()
        self.train_embedded = np.fromfile(EMBEDDING_PATH, dtype=np.float32).reshape(-1, 4096)

    def embedding(self):
        """ create embedding textfile with train data """
        print('  ==> Generate embedding...', end='')
        self.model.eval()  # set to eval mode
        if torch.cuda.is_available():
            self.model.to('cuda')

        train_dataset = DatasetImageNet(TRIPLET_PATH, embedding=True, transform=data_transforms['val'])

        embedded_images = []
        for idx in range(len(train_dataset)):
            input_tensor = train_dataset[idx][0]
            input_batch = input_tensor.unsqueeze(0)

            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')

            embedding = self.model(input_batch)
            embedding_np = embedding.cpu().detach().numpy()

            embedded_images.append(embedding_np)  # collect train data's predicted results

        embedded_images_train = np.concatenate(embedded_images, axis=0)
        embedded_images_train.astype('float32').tofile(EMBEDDING_PATH)  # save embedding result
        print('done! [embedding.txt] generated')

    def query_embedding(self, query_image_path):
        """ return embedded query image """
        print(f'Query image [{query_image_path}] embedding...', end='')

        # read query image and pre-processing
        query_image = Image.open(query_image_path).convert('RGB')
        query_image = data_transforms['val'](query_image)
        query_image = query_image[None]  # add new axis. same as 'query_image[None, :, :, :]'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.to(device) 
        self.model.eval()  # set to eval mode
        

        embedding = self.model(query_image.to(device))
        print('done!')
        return embedding.cpu().detach().numpy()

    def save_result(self, result, distances, k, result_name):
        """ save similarity result """
        print('Save predicted result ...', end='')
        fig = plt.figure(figsize=(128, 128))
        columns = k + 1
        ax = []

        for i in range(1, int(columns)+1):
            
            img_path = result[i-1]
            img = mpimg.imread(img_path)  # read image
            ax.append(fig.add_subplot(1, columns, i))
            if i == 1:  # query image
                ax[-1].set_title("query image", fontsize=50)
            else:  # others
                ax[-1].set_title("img_:" + str(i - 1), fontsize=50)
                #ax[-1].set_xlabel('l2-dist=' + str(distances[i-2]))
                #ax[-1].xaxis.label.set_fontsize(25)
            plt.imshow(img, cmap='Greys_r')
        plt.savefig(result_name)  # save as file
        print('done!')

    def predict(self, query_image_path, k=5, save_as='result.png'):
        """ predict top-n similar images """
        # check query path is valid
        if not os.path.exists(query_image_path):
            print(f'[ERROR] invalid query image path: {query_image_path}')
            return

        # embedding query image
        query_embedded = self.query_embedding(query_image_path)
        vec_dim = 4096
        
        index = faiss.IndexFlatL2(vec_dim)
        # n_list = 10
        # USE_GPU = True
        # quantizer = faiss.IndexFlatL2(vec_dim)
        # index = faiss.IndexIVFFlat(quantizer, vec_dim, n_list)
        # if USE_GPU:
        #   print("Use GPU...")
        #   res = faiss.StandardGpuResources()
        #   index = faiss.index_cpu_to_gpu(res, 0, index)
        # index.train(self.train_embedded)
        index.add(self.train_embedded)
        
        #index.nprobe = 2
        distances, indices = index.search(query_embedded, k=5)
        #print(indices)

        extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

        def get_file_list(root_dir):
            file_list = []
            for root, directories, filenames in os.walk(root_dir):
                for filename in filenames:
                    if any(ext in filename for ext in extensions):
                        file_list.append(os.path.join(root, filename))
            return file_list

        # path to the your datasets
        root_dir = 'dataset'
        filenames = get_file_list(root_dir)
        results = []
        results.append(query_image_path)
        for i in indices[0]:
            results.append(filenames[i])
            #print(filenames[i])

        distances = distances[0]  

        #  by euclidean distance, find top ranked similar images
        # image_dist = euclidean_distance(self.train_embedded, query_embedded)
        # image_dist_indexed = zip(image_dist, range(image_dist.shape[0]))
        # image_dist_sorted = sorted(image_dist_indexed, key=lambda x: x[0])

        # # top 5 images
        # predicted_images = [(img[0], self.train_df.loc[img[1], "query"]) for img in image_dist_sorted[:result_num]]
        
        # make png file
        
        self.save_result(results, distances, k, result_name=save_as)


def pred_main( MODEL_PATH, EMBEDDING_PATH, TRIPLET_PATH, image_path = './dataset/100s/charcoal_11_100s_negative.jpg',):
    predictor = Prediction(model_path=MODEL_PATH, embedding_path=EMBEDDING_PATH, triplet_path=TRIPLET_PATH)
    # image_path1 = './data/002.american-flag/002_0002.jpg'
    # #image_path2 = './data/003.backpack/003_0026.jpg'
    # image_path2 = './data/012.binoculars/012_0046.jpg'
    # image_path3 = './data/018.bowling-pin/018_0011.jpg'
        
        
    
    #image_path2 = './dataset/charcoal/charcoal_12_positive.jpg'
    # image_path1 = './dataset/100s/original_100s_41_anchor.jpg'
    # image_path2 = './dataset/100s/original_100s_51_anchor.jpg'
    #image_path3 = './dataset/original/original_21_anchor.jpg'

    # get images for 3 Validation set
    test_images = [image_path]
    for idx, p in enumerate(test_images):
        predictor.predict(p, 5, f'result_{idx}.png')

    return "success"


if __name__ == '__main__':
    pred_main()

