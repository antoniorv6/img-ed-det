# Image Edition Detection

Repository for the image edition detection experiments. In this work, we try to predict, from two input images, the edition process that has been done.

Currently, the repository contains a data generator (`Data_generator.py`) which creates the inputs (named `X_source` and `X_target`) and outputs (`Y`) of the detection system.

The current supported editions are:
- X axis flip
- Y axis flip
- Central 50% zoom
- Gaussian Blur
- 3x3 Erosion
- 3x3 Dilation
- 90º rotation
- 180º rotation
- No edition

#

## Instalation
### Local installation
Before running the data generation code, you need download the source data to start generating edited images.

[Download the dataset](https://drive.google.com/file/d/1AvVsyvJNlSXlrLZHtFJPryNaHCZpKhlE/view?usp=sharing)

Once downloaded, execute in the repository folder:

```
tar -xzvf data-img-ed.tgz
```
Then, install the project requirements
```python
pip install -r requirements.txt
```
Try the generator executing

```python
python DataGenerator.py
```

#
## Documentation
### `DataGenerator.py`
This is the data generation file, it contains a main method to make a test execution. This file contains the `DataGen` class, which handles all the data reading and preprocessing to serve the samples to the network.

The main methods to generate data are the following:

**`DataGen()`**

The class constructor handles the data reading and sepparation. It creates two lists, one for the training dataset and another one for the validation one, as we want to ensure that some images are never seen during training. This process is done automatically when the class is constructed, so we do not need to specify any parameters.

The train/validation sepparation is performed with a 75 - 25 % distribution.

**`train_batch()`**

Gets one batch from the **training set**

Inputs:

- BATCH_SIZE: Number of samples the final batch has to contain. These examples are taken from the training

Outputs: 

- X_source: (BATCH_SIZE, IMG_W, IMG_H, 1) array of images. This array represents the source images.

- X_target: (BATCH_SIZE, IMG_W, IMG_H, 1) array of images. This array represents the source images with **one additional edition**, which has to be predicted. 

- Y: (BATCH_SIZE, NUM_OPERATIONS) array. Consists on a one-hot array which marks which operation has been done between X_source and X_target

**`val_batch()`**

Gets one batch from the **validation set**

Inputs:

- BATCH_SIZE: Number of samples the final batch has to contain. These examples are taken from the training

Outputs: 

- X_source: (BATCH_SIZE, IMG_W, IMG_H, 1) array of images. This array represents the source images.

- X_target: (BATCH_SIZE, IMG_W, IMG_H, 1) array of images. This array represents the source images with **one additional edition**, which has to be predicted. 

- Y: (BATCH_SIZE, NUM_OPERATIONS) array. Consists on a one-hot array which marks which operation has been done between X_source and X_target

### Code example in network training

```python
dataGen = DataGen()
BATCH_SIZE = 16

for EPOCH in range(NUM_EPOCHS):
    X_source, X_target, Y = generator.train_batch(BATCH_SIZE)
    # Train model
    model.fit([X_source, X_target], Y, epochs=NUM_INTERNAL_EPOCHS)
    # Validate model
    for i in range(generator.get_val_size() // 16 ):
        X_source_val, X_target_val, Y_val = generator.val_batch(16)
        model.validate([X_source_val, X_target_val], Y_val)
```
#

Ownership:

- Jorge Calvo-Zaragoza (University of Alicante)
- Enrique Más (Facephi)
- Antonio Ríos-Vila (University of Alicante)