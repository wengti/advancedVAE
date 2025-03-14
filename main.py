import torch
import torch.nn as nn
from load_data import custom_data
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import yaml
from model import VAE
from tqdm.auto import tqdm
from engine import train_step, test_step
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# seed
torch.manual_seed(111)
torch.cuda.manual_seed(111)


# 0. Load config

configPath = "./no_condt_2lat.yaml"

with open(configPath, "r") as file:
    try:
        config = yaml.safe_load(file)
        print(f"[INFO] {configPath} has been successfully loaded.")
    except yaml.YAMLError as exc:
        print(exc)

for key in config.keys():
    print(f"[INFO] {key} : {config[key]}")



# 1. Load dataset
trainData = custom_data(directory = "./data/train")
testData = custom_data(directory = "./data/test")

# 2. Visualize the dataset
randNum = torch.randint(0, len(trainData)-1, (9,))

for i, num in enumerate(randNum):

    trainImg, trainLabel = trainData[num]

    trainImgPlt = ((trainImg+1)/2).permute(1,2,0)

    plt.subplot(3,3, i+1)
    plt.imshow(trainImgPlt, cmap="gray")
    plt.title(f"Label: {trainLabel}")
    plt.axis(False)

plt.tight_layout()
plt.show()


print(f"[INFO] Number of images in the dataset : {len(trainData)}")
print(f"[INFO] The size of an image            : {trainImg.shape}")
print(f"[INFO] The value range in the image    : from {trainImg.min()} to {trainImg.max()} ")
print(f"[INFO] The classes available           : {trainData.classes}")

# 3. Load dataloader
trainDataLoader = DataLoader(dataset = trainData,
                             batch_size = config['batch_size'],
                             shuffle = True)
testDataLoader = DataLoader(dataset = testData,
                            batch_size = config['batch_size'],
                            shuffle = False)

# 4. Visualize dataloaders
trainImgBatch, trainLabelBatch = next(iter(trainDataLoader))

print(f"[INFO] The number of batches in the dataloader: {len(trainDataLoader)}")
print(f"[INFO] Number of images per batch             : {trainImgBatch.shape[0]}")
print(f"[INFO] Size of an image                       : {trainImgBatch[0].shape}")

# 5. Create model
model0 = VAE(config = config,
             device = device).to(device)

# 6. Verify model
# =============================================================================
# from tochinfo import summary
#
# summary(model = model0,
#         input_size = [1,1,28,28],
#         col_names = ['input_size', 'output_size', 'trainable', 'num_params'],
#         row_settings = ['var_names'])
# =============================================================================

# 7. Create optimizer
optimizer = torch.optim.Adam(params = model0.parameters(),
                             lr = config['learning_rate'])

scheduler = ReduceLROnPlateau(optimizer = optimizer,
                              factor = 0.5,
                              patience = 1)


# 8. Save model
def save_model(model, save_dir, save_name):
    save_dir = Path(save_dir)

    if not save_dir.is_dir():
        save_dir.mkdir(parents = True,
                       exist_ok = True)

    assert save_name.endswith(".pt") or save_name.endswith(".pth"), f"[INFO] {save_name} is not valid. Please check the file extension."
    save_file = save_dir / save_name

    torch.save(obj = model.state_dict(),
               f = save_file)

# 9. Create training loops
trainLossList = []
testLossList = []
bestLoss = np.inf

for epoch in tqdm(range(config['epochs'])):

    trainResult = train_step(model = model0,
                             dataloader = trainDataLoader,
                             device = device,
                             optimizer = optimizer)

    testResult = test_step(model = model0,
                           dataloader = testDataLoader,
                           device = device)

    trainLossList.append(trainResult['loss'].cpu().detach().numpy())
    testLossList.append(testResult['loss'].cpu().detach().numpy())

    scheduler.step(trainResult['loss'])

    print(f"[INFO] Current epoch : {epoch}")
    print(f"[INFO] Train Loss    : {trainResult['loss']:.4f}")
    print(f"[INFO] Test Loss     : {testResult['loss']:.4f}")

    if trainResult['loss'] < bestLoss:
        print(f"[INFO] The best loss has been improved from {bestLoss} to {trainResult['loss']}.")
        bestLoss = trainResult['loss']

        print("[INFO] Saving this as the best model...")
        save_model(model = model0,
                   save_dir = f"./{model0.config['task_name']}",
                   save_name = "best.pt")

