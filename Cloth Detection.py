{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled6.ipynb",
      "private_outputs": true,
      "provenance": [],
      "authorship_tag": "ABX9TyOs+/jsgUKC+mk11YNqsPYa",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/anujgupta100/Cloth-Detection-Using-Pyhton/blob/main/Cloth%20Detection.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4mR1FQAh-Cjk"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# storing the dataset path\n",
        "import tensorflow_datasets as tfds\n",
        "clothing_fashion_mnist = tf.keras.datasets.fashion_mnist\n",
        "\n",
        "# loading the dataset from tensorflow\n",
        "(x_train, y_train),(x_test, y_test) = clothing_fashion_mnist.load_data()\n",
        "\n",
        "# displaying the shapes of training and testing dataset\n",
        "print('Shape of training cloth images: ',\n",
        "\tx_train.shape)\n",
        "\n",
        "print('Shape of training label: ',\n",
        "\ty_train.shape)\n",
        "\n",
        "print('Shape of test cloth images: ',\n",
        "\tx_test.shape)\n",
        "\n",
        "print('Shape of test labels: ',\n",
        "\ty_test.shape)\n"
      ],
      "metadata": {
        "id": "ag_XrL9cBYEr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# storing the class names as it is\n",
        "# not provided in the dataset\n",
        "label_class_names = ['T-shirt/top', 'Trouser',\n",
        "\t\t\t\t\t'Pullover', 'Dress', 'Coat',\n",
        "\t\t\t\t\t'Sandal', 'Shirt', 'Sneaker',\n",
        "\t\t\t\t\t'Bag', 'Ankle boot']\n",
        "\n",
        "# display the first images\n",
        "plt.imshow(x_train[10])\n",
        "plt.colorbar() # to display the colourbar\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "WjfxkTuz5Szh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train = x_train / 310.0 # normalizing the training data\n",
        "x_test = x_test / 200.0 # normalizing the testing data"
      ],
      "metadata": {
        "id": "hfvorPSgBdb_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(15, 5)) # figure size\n",
        "i = 0\n",
        "while i < 20:\n",
        "\tplt.subplot(2, 10, i+1)\n",
        "\t\n",
        "\t# showing each image with colourmap as binary\n",
        "\tplt.imshow(x_train[i], cmap=plt.cm.binary)\n",
        "\t\n",
        "\t# giving class labels\n",
        "\tplt.xlabel(label_class_names[y_train[i]])\n",
        "\ti = i+1\n",
        "\t\n",
        "plt.show() # plotting the final output figure\n"
      ],
      "metadata": {
        "id": "d2Bug00jCR-m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Building the model\n",
        "model = tf.keras.Sequential([\n",
        "\ttf.keras.layers.Flatten(input_shape=(28, 28)),\n",
        "\ttf.keras.layers.Dense(128, activation='relu'),\n",
        "\ttf.keras.layers.Dense(10)\n",
        "])\n"
      ],
      "metadata": {
        "id": "JfJ6E1zIEPqo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# compiling the model\n",
        "model.compile(optimizer='adam',\n",
        "\t\t\t\t\tloss=tf.keras.losses.SparseCategoricalCrossentropy(\n",
        "\t\t\t\t\t\tfrom_logits=True),\n",
        "\t\t\t\t\tmetrics=['accuracy'])\n"
      ],
      "metadata": {
        "id": "PoMpsFZ1EPxU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(x_train, y_train, epochs=10)"
      ],
      "metadata": {
        "id": "e-OfNO_jrrbM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# calculating loss and accuracy score\n",
        "test_loss, test_acc = model.evaluate(x_test,\n",
        "                                           y_test,\n",
        "                                           verbose=2)\n",
        "print('\\nTest loss:', test_loss)\n",
        "print('\\nTest accuracy:', test_acc)"
      ],
      "metadata": {
        "id": "_NQj4-i1uY0s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# using Softmax() function to convert\n",
        "# linear output logits to probability\n",
        "prediction_model = tf.keras.Sequential(\n",
        "\t[model, tf.keras.layers.Softmax()])\n",
        "\n",
        "# feeding the testing data to the probability\n",
        "# prediction model\n",
        "prediction = prediction_model.predict(x_test)\n",
        "\n",
        "# predicted class label\n",
        "print('Predicted test label:', np.argmax(prediction[1]))\n",
        "\n",
        "# predicted class label name\n",
        "print(label_class_names[np.argmax(prediction[1])])\n",
        "\n",
        "# actual class label\n",
        "print('Actual test label:', y_test[1])\n"
      ],
      "metadata": {
        "id": "4NV8PCKLyB1H"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# assigning the figure size\n",
        "plt.figure(figsize=(15, 6))\n",
        "i = 0\n",
        "\n",
        "# plotting total 24 images by iterating through it\n",
        "while i < 24:\n",
        "\timage, actual_label = x_test[i], y_test[i]\n",
        "\tpredicted_label = np.argmax(prediction[i])\n",
        "\tplt.subplot(3, 8, i+1)\n",
        "\tplt.tight_layout()\n",
        "\tplt.xticks([])\n",
        "\tplt.yticks([])\n",
        "\t\n",
        "\t# display plot\n",
        "\tplt.imshow(image)\n",
        "\t\n",
        "\n",
        "\tcolor, label = ('green', 'Correct Prediction')\n",
        "\t\n",
        "\t# plotting labels and giving color to it\n",
        "\t# according to its correctness\n",
        "\tplt.title(label, color=color)\n",
        "\t\n",
        "\t# labelling the images in x-axis to see\n",
        "\t# the correct and incorrect results\n",
        "\tplt.xlabel(\" {} ~ {} \".format(\n",
        "\t\tlabel_class_names[actual_label],\n",
        "\tlabel_class_names[predicted_label]))\n",
        "\t\n",
        "\t# labelling the images orderwise in y-axis\n",
        "\tplt.ylabel(i)\n",
        "\t\n",
        "\t# incrementing counter variable\n",
        "\ti += 1\n"
      ],
      "metadata": {
        "id": "uxBu0p4Mygc2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "NbWgQ_GKEP2E"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}