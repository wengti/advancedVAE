# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:10:15 2025

@author: User
"""

import torch
import torchvision
from einops import rearrange
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pickle
import cv2
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def reconstruction(model, dataset):

    # save name
    save_folder = Path(f"./{model.config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)

    save_image_name = "reconstruction.jpg"
    save_file = save_folder / save_image_name

    # Get 100 images from dataset
    rand_num = torch.randint(0, len(dataset)-1, (100,))
    test_imgs = torch.cat([dataset[num][0][None, :] for num in rand_num], dim=0)

    # Handle the label
    labels = None
    if model.config['conditional']:
        labels = torch.tensor([dataset[num][1] for num in rand_num])

    # Pass the images and label into the model for inference / reconstruction
    model.eval()
    with torch.inference_mode():
        test_imgs = test_imgs.to(device)

        if labels is not None:
          labels = labels.to(device)

        X_recon, _, _ = model(x = test_imgs, label = labels)

        # Adjust the value of images to be in the range of 0 - 1
        test_imgs = (test_imgs + 1) / 2
        X_recon = 1 - (X_recon + 1) / 2

        # Put images side by side
        out = torch.hstack([test_imgs, X_recon])
        output = rearrange(out, "B C H W -> B () H (C W)")

        # Convert to 10 x 10 grid
        grid = torchvision.utils.make_grid(output, nrow = 10)

        # Save image
        output_img = torchvision.transforms.ToPILImage()(grid)
        output_img.save(save_file)

        print(f"[INFO] Reconstructed images have been saved in {save_file}.")

# =============================================================================
# reconstruction(model = model0,
#                dataset = testData)
# =============================================================================




##############################################################################



def visualize_scatter(model, dataloader):

    # save name
    save_folder = Path(f"./{model.config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)

    save_image_name = "scatter_plot.jpg"
    save_image_file = save_folder / save_image_name

    save_pca_name = "PCA.pkl"
    save_pca_file = save_folder / save_pca_name

    # Initialize list to store tensors for plot
    means = []
    labels_plt = []

    # Inference
    model.eval()
    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            labels = None
            if model.config['conditional']:
                labels = y

            _, mean, _ = model(x = X,
                               label = labels)

            means.append(mean) # [ tensor(32xlatent_dim), tensor(32xlatent_dim) ]
            labels_plt.append(y) # [torch.tensor([1,2,3...]), torch.tensor([1,2,3]), ]

        means = torch.cat(means, dim=0) # tensor( [Number of samples x 2] )
        labels_plt = torch.cat(labels_plt) # torch.tensor([1,2,3,1,2,3....])

        # Project to lower dimensionality if needed
        if model.config['latent_dim'] != 2:
            print(f"[INFO] This model has a latent dimensionality of {model.config['latent_dim']}, which is more than 2.")
            print("[INFO] Proceed to create a PCA file to project it to a dimensionality of 2 for plotting")

            _, _, V = torch.pca_lowrank(means, center = True, niter = 2)
            proj_means = torch.matmul(means, V[:,:2])
            means = proj_means

            pickle.dump(V, open(save_pca_file, "wb"))

        # Plot the scatter plot
        means = means.cpu().detach().numpy() # Converted to numpy, stored on cpu
        labels_plt = labels_plt.cpu().detach().numpy() # Converted to numpy, stored on cpu

        for num in range(model.config['num_classes']):
            idxs = np.where(labels_plt == num)[0]

            plt.scatter(x = means[idxs,0], y = means[idxs,1],
                        s = 10, label = str(num))

        # Save images
        plt.title("Scatter Plot")
        plt.legend()
        plt.savefig(save_image_file)

# =============================================================================
# visualize_scatter(model = model0,
#                   dataloader = testDataLoader)
# =============================================================================



###############################################################################


def visualize_interpolation(model, dataset):

    # name for save folder
    save_folder = Path(f"{model.config['task_name']}/interpolation")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)

    # Create 2 samples of imgs and labels
    rand_num = torch.randint(0, len(dataset)-1, (2,))
    test_imgs = torch.cat([dataset[num][0][None,:] for num in rand_num], dim=0).to(device) # tensor, 2x1x28x28, cuda

    labels = None
    if model.config['conditional']:
        labels = torch.tensor([dataset[num][1] for num in rand_num]).to(device) # tensor, [1,2...], cuda

    # Inference
    model.eval()
    with torch.inference_mode():

        # Obtain the mean of the 2 imgs
        _, mean, _ = model(x = test_imgs, label = labels) # mean: tensor, 2xlatent_dim, cuda

        # Set 1 as start and 1 as end mean
        mean_start = mean[0] # shape: latent_dim
        mean_end = mean[1] # shape: latent_dim

        # Create interpolated means point along a straight line that connect them
        factor = torch.linspace(0, 1.0, 500).to(device) # shape: 500
        mean_plt = torch.matmul(factor[:, None], mean_end[None, :]) + torch.matmul(1 - factor[:, None], mean_start[None, :]) #shape: 500x2

        # Create target label for generation
        prod_labels = None
        if model.config['conditional']:
            prod_labels = labels[1].repeat((500)) # tensor, shape: 500, cuda

        # Generate all 500 imgs
        X_recon = model.generate(z = mean_plt,
                                 label = prod_labels)

        # Save imgs
        for idx, img in enumerate(X_recon):

            img = ((img+1)/2)*255 #Convert to range of 0 - 255
            img = img.permute(1,2,0) # Convert to 28x28x1
            img = img.cpu().detach().numpy() # Convert to numpy array
            img = np.uint8(img) # Make sure its in integer

            save_img_name = f"Image{idx}.png"
            save_img_file = save_folder / save_img_name

            cv2.imwrite(save_img_file, img)

        print(f"[INFO] Interpolated images have been saved into {save_folder}.")

# =============================================================================
# visualize_interpolation(model = model0,
#                         dataset = testData)
# =============================================================================




###############################################################################

def visualize_manifold(model, dataset, low, high):

    # save name
    save_folder = Path(f"./{model.config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)

    save_pca_name = "PCA.pkl"
    save_pca_file = save_folder / save_pca_name

    # setting up loops if required
    classes = 1
    if model.config['conditional']:
        classes = model.config['num_classes']

    for idx in range(classes):

        # Create mean values in space of -4 to 4
        xs = torch.linspace(low, high, 30)
        ys = torch.linspace(low, high, 30)
        xs, ys = torch.meshgrid([xs, ys], indexing = 'ij')

        xs = xs.reshape((-1, 1)) #900x1
        ys = ys.reshape((-1, 1)) #900x1

        zs = torch.cat([xs,ys], dim=1).to(device) # 900x2, cuda

        # Convert to higher dimensions if needed
        if model.config['latent_dim'] !=2:
            print(f"[INFO] The specified latente dimension of this model is {model.config['latent_dim']}.")
            print("[INFO] Proceed to project it to a higher dimensionality.")

            assert os.path.exists(save_pca_file), f"[INFO] Cannot locate the required file for means projection in {save_pca_file}. Please check."

            V = pickle.load(file = open(save_pca_file, "rb"))
            proj_zs = torch.matmul(zs, (V[:,:2].T).to(device))
            zs = proj_zs # 900 x lat_dim, cuda

        # Create labels
        labels = None
        if model.config['conditional']:
            labels = torch.tensor([idx]).repeat((30*30)).to(device) # 900, cuda, tensor

        # Inference
        model.eval()
        with torch.inference_mode():

            X_recon = model.generate(z = zs,
                                     label = labels) # 900x1x28x28

            # Save image
            grid = torchvision.utils.make_grid(X_recon, nrow = 30)
            output_img = torchvision.transforms.ToPILImage()(grid)

            save_image_name = f"Manifold for class {idx}.png"
            save_image_file = save_folder / save_image_name

            output_img.save(save_image_file)
            print(f"[INFO] Manifold for class {idx} has been saved into {save_image_file}.")

# =============================================================================
# visualize_manifold(model = model0,
#                    dataset = testData,
#                    low = -2,
#                    high = 2)
# =============================================================================
            
            
        
        
        
        

            
            
            
        
        


       
        
        
        
    
    