# Coastline Segmentation

This project uses the Sentinel-2 Water Edges Dataset (SWED) to train a U-Net model for coastline segmentation.

## Directory Structure

-`data/`: Contains training and test data.

  -`train/`: Directory for training images and labels.

    -`images/`: Directory for training images in `.npy` format.

    -`labels/`: Directory for training labels in `.npy` format.

  -`test/`: Directory for test images and labels.

    -`images/`: Directory for test images in `.npy` format.

    -`labels/`: Directory for test labels in `.npy` format.

-`models/`: Contains model definitions and saved models.

  -`unet.py`: Defines the U-Net architecture used for segmentation.

-`train.py`: Script to train the U-Net model.

-`evaluate.py`: Script to evaluate the model on the test dataset.

-`display_samples.py`: Script to visualize sample images, labels, and predictions.

-`requirements.txt`: Lists the dependencies required for the project.

-`.gitignore`: Specifies files and directories to be ignored by Git.

## Dataset

The Sentinel-2 Water Edges Dataset (SWED) is a dataset designed for deep learning applications in image segmentation. It contains annotated Sentinel-2 imagery with pairs of images and corresponding segmentation labels. The labels contain two classes indicated by binary values – water (1) and non-water (0).

You can download the dataset from the [UK Hydrographic Office](https://openmldata.ukho.gov.uk/) under the Geospatial Commission Data Exploration license.

### Downloading the Dataset

1.**Download the full dataset** (approx. 18GB) or a smaller sample (approx. 1.36GB) from the provided link.

2. Extract the dataset into the `data/` directory, ensuring the following structure:

```

coastline_segmentation/

│

├── data/

│   ├── train/

│   │   ├── images/

│   │   └── labels/

│   ├── test/

│   │   ├── images/

│   │   └── labels/

```

## Setup

1.**Clone the repository**:

```sh

   git clone https://github.com/yourusername/coastline_segmentation.git

   cd coastline_segmentation

```

2.**Install the dependencies**:

```sh

   pip install -r requirements.txt

```

3.**Prepare the data**:

   Ensure that your `data/train` and `data/test` directories contain the `.npy` files for images and labels as described in the dataset section.

## Training the Model

To train the U-Net model, run:

```sh

pythontrain.py

```

This will train the model using the data in `data/train/images` and `data/train/labels`, and save the trained model to `models/coastline_segmentation_model.h5`.

## Evaluating the Model

To evaluate the model on the test dataset, run:

```sh

pythonevaluate.py

```

This will load the test data from `data/test/images` and `data/test/labels`, evaluate the model, and display sample predictions.

## Visualizing Sample Data

To visualize sample images, labels, and predictions, run:

```sh

pythondisplay_samples.py

```

## File Descriptions

### `data/process_data.py`

Responsible for loading and preprocessing the dataset.

### `models/unet.py`

Defines the U-Net architecture used for segmentation.

### `train.py`

Script to train the U-Net model.

### `evaluate.py`

Script to evaluate the trained model on the test dataset.

### `display_samples.py`

Script to visualize sample images, labels, and predictions.

### `requirements.txt`

Lists the dependencies required for the project.

### `.gitignore`

Specifies files and directories to be ignored by Git, including `data/train` and `data/test`.

## License

This project is licensed under the terms specified by the UK Hydrographic Office for the Sentinel-2 Water Edges Dataset (SWED). Please refer to the dataset license for more details.

---

Feel free to contribute to this project by submitting issues or pull requests. For any questions or inquiries, please contact [your email].
