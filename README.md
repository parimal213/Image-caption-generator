# Image-caption-generator
INTRODUCTION TO OUR PROJECT-

Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English. In this Deep Learning project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

ABOUT THE DATASET- For this image caption generator project , we will be using the Flickr_8K dataset. The dataset can be downloaded from Kaggle.

PREREQUISITE INSTALLATION- These Libraries need to be installed - tensorflow, keras, pillow, numpy.

LET'S START THE PROJECT NOW-

1- IMPORTING ALL NECESSARY LIBRARIES AND DOWNLOADING DATASET:- Just import all the libraries and download dataset from kaggle.

2- GETTING AND PERFORMING DATA CLEANING STEP:- The main text file which contains all image captions is Flickr8k.token in Flickr_8k_text folder. Create a Descriptions.txt file in which each line contains image_path in Flicker 8k Dataset folder and a space followed by a caption.

3- EXTRACT FEATURES FROM IMAGES:- we will use the pre-trained Xception model to extract the features of all the images and map image names to its respective feature vector. We will save the featurs to feature.p file so that we can use it later.

4- LOADING DATASET FOR TRAINING THE MODEL:- In Flickr_8k_text folder, we have Flickr_8k.trainImages.txt file that contains a list of 6000 image names that we will use for training. We will also perform cleaning of descriptions in this step. This step will give us the dictionary for image names and their feature vector which we have previously extracted from the Xception model.

5- TOKENIZING THE VOCABULARY:- In this step we will map each word of the vocabulary with a unique index value. Keras library provides us with the tokenizer function that we will use to create tokens from our vocabulary and save them to a “tokenizer.p” pickle file.

6- CREATING DATA GENERATOR:- We have to train our model on 6000 images and each image will contain 2048 length feature vector and caption is also represented as numbers. This amount of data for 6000 images is not possible to hold into memory so we will be using a generator method that will yield batches. The generator will yield the input and output sequence.

7- DEFINING THE CNN-RNN MODEL:- Feature Extractor – The feature extracted from the image has a size of 2048, with a dense layer, we will reduce the dimensions to 256 nodes. Sequence Processor – An embedding layer will handle the textual input, followed by the LSTM layer. Decoder – By merging the output from the above two layers, we will process by the dense layer to make the final prediction. The final layer will contain the number of nodes equal to our vocabulary size.

8- TRAINING THE MODEL:- We will train the model for 10 epochs on CPU. It will take almost 5 hours. To train the model, we will be using the 6000 training images by generating the input and output sequences in batches and fitting them to the model using model.fit_generator() method.

9- TESTING THE MODEL ON AN EXAMLE IMAGE:- we will feed an image to our model to generate a caption.

FUTURE SCOPE OF THIS PROJECT:- The model fails sometimes on outside image as it depends on data and so it cannot predict the words that are out of vocabulary. But we can improve the model by training it on big datasets and training for more epochs. Overall this is an advanced deep learning project and clears all our concepts.
