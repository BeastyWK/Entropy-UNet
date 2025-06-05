Description:
Entropy-Unet is a deep neural network model trained to predict entropy by utilizing conformational features extracted from trajectory files generated through meta-dynamics simulations

Software prerequisites(linux or windows):
python                       3.7
torchaudio                   2.1.0.dev20230329+cu118
torchsummary                 1.5.1
torchvision                  0.16.0.dev20230329+cu118
numpy                        1.23.5

Feature engineering:
compute entropy/atom command of lammps

CRE Lable engineering:
run CRE_lable.py

Train Entropy-Unet：
run train_entropy.py

Predict CRE:
run prefict_entropy.py

Result:
Per-particle entropy probability distribution
