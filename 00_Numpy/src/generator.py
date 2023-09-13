import numpy as np
import json
import matplotlib.pyplot as plt

class ImageGenerator():

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        with open(self.label_path, 'r') as f:
            self.labels = json.load(f) #load labels
        self.__nsamples = len(self.labels) #store length of labels in private variable
        self.__epoch = 0 #initialize a private instance variable to keep track of current epoch
        self.__batch_num = 0 #initialize a private instance variable to keep track of current batch number
        if self.__nsamples < self.batch_size or self.batch_size == 0:
            self.batch_size = self.__nsamples #batchsize should be at least length of labels
        self.__map = np.arange(self.__nsamples) # create an array


        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'} #Dictionary


    def next(self):
        '''
            Creates a batch of images and corresponding labels and return them.

        '''
        if self.__batch_num * self.batch_size >= self.__nsamples: # if condition met go to next batch
            self.__epoch += 1
            self.__batch_num = 0
        
        if self.__batch_num == 0 and self.shuffle == True: # new epoch ho? check it
            np.random.shuffle(self.__map)
        
        # Initialize array
        images = np.zeros((self.batch_size, *tuple(self.image_size))) # * operator is used to unpack the elements of the self.image_size tuple and include them as individual elements within the new tuple
        labels = np.zeros(self.batch_size, dtype=int) # create an array to store labels for each image in the batch

        start = self.__batch_num * self.batch_size # computes the starting index for the current batch
        if (self.__batch_num + 1) * self.batch_size <= self.__nsamples: # is there enough samples to fill a complete batch??
            for i in range(self.batch_size):
                images[i] = self.augment(np.load(f"{self.file_path}/{self.__map[start + i]}.npy")) #load images amd augment it and store it
                labels[i] = self.labels[str(self.__map[start+i])] #load labels and store it
            self.__batch_num += 1 # next batch

        elif self.__batch_num * self.batch_size < self.__nsamples:
            last_batch_size = self.__nsamples - self.__batch_num * self.batch_size #samples in last batch
            for i in range(last_batch_size):
                images[i] = self.augment(np.load(f"{self.file_path}/{self.__map[start + i]}.npy"))
                labels[i] = self.labels[str(self.__map[start + i])]
            
            for i in range(self.batch_size - last_batch_size): # fill remaining space in a batch with new images
                images[last_batch_size + i] = self.augment(np.load(f"{self.file_path}/{self.__map[i]}.npy"))
                labels[last_batch_size + i] = self.labels[str(self.__map[i])]
        
        #print(start, self.__batch_num, self.batch_size, self.__nsamples, self.__epoch)

        
        return images, labels

    def augment(self, img):
        """
            Take single image as input and perform a random transformation (mirror/rotation) and return it.
        """
        # Check image size
        if img.shape != self.image_size:
            img = np.resize(img, self.image_size)

        # Mirroring
        if self.mirroring == True:
            if np.random.choice((True, False)):
                img = np.fliplr(img) # flop left right
            if np.random.choice((True,False)):
                img = np.flipud(img) # flip up down
        
        # Rotation
        if self.rotation == True:
            n_times = np.random.choice((0,1,2,3))
            img = np.rot90(img, n_times)

        return img
     

    def current_epoch(self):

        return self.__epoch

    def class_name(self, x):
        
        return self.labels[str(x)] # return class name for a specific input

    def show(self):
        '''
            Verify if the generator creates batches as required.
        '''
        imgs, labs = self.next()
        fig = plt.figure(figsize=(10,10))
        cols =3
        rows = self.batch_size // 3 + (1 if self.batch_size % 3 else 0)

        for i in range(1, self.batch_size+1):
            img = imgs[i-1]
            lab = self.class_dict[labs[i-1]]

            fig.add_subplot(rows, cols, i)
            plt.imshow(img.astype('uint8'))
            plt.xticks([])
            plt.yticks([])
            plt.title(lab)
        plt.show()
