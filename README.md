Description:
Entropy-Unet is a deep neural network model trained to predict entropy by utilizing conformational features extracted from trajectory files generated through meta-dynamics simulations

Overview
Metadynamics simulations generate extensive conformational ensembles. Entropy-Unet converts these ensembles into the coordination resolved entropy (CRE). By combining:

UNet – a convolutional network originally designed for biomedical image segmentation,

Conformational features – obtained from LAMMPS (compute entropy/atom),

Reference entropy labels – computed by the CRE method,

the model learns to directly map local structural descriptors to per‑particle entropy probabilities. The output is a probability distribution of entropy values for each particle, enabling detailed analysis of entropy in complex molecular systems.

System Requirements
Operating System: Linux (recommended) or Windows

Python: 3.7

PyTorch: 2.1.0 (with CUDA 11.8 support)

Additional Python packages:

torchaudio 2.1.0.dev20230329+cu118

torchvision 0.16.0.dev20230329+cu118

torchsummary 1.5.1

numpy 1.23.5

Note: GPU acceleration is strongly advised for training. The specified versions have been tested; newer versions may work but are not guaranteed.

Feature Engineering with LAMMPS
Conformational features are extracted from metadynamics trajectory files using the LAMMPS compute entropy/atom command. This compute calculates an entropy estimate for each atom based on the local environment.

CRE Lable engineering:
run CRE_lable.py

Train Entropy-Unet：
run train_entropy.py

Predict CRE:
run prefict_entropy.py

Result:
Per-particle entropy probability distribution
