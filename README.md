# advancedVAE

## Credits & Acknowledgments  
This project is a reimplementation of [VAE-Pytorch] by [explainingai-code] (https://github.com/explainingai-code/VAE-Pytorch).  
The code has been rewritten from scratch while maintaining the core concepts and functionalities of the original implementation.  


## Features  
- By modifying the config files, VAE with various latent dimensions and conditional VAE can be built.

## Description of Files
- **extract_mnist.py** - Extracts MNIST data from the CSV file.  
- **load_data.py** - Creates a custom dataset.
- **model.py** - Compatible with .yaml config files to create various VAE models. 
- **engine.py** - Defines the train and test steps (for 1 epoch).  
- **main.py** - Trains the model
- **infer.py**
  1. Reconstruct images
  2. Visualize scatter plot in latent dimensions
  3. Visualize images along the line interpolated between 2 images in the latent dimensions
  4. Visualize manifold
