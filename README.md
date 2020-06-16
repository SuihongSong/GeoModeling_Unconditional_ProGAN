Code for this project is improved from the original code of Progressive GANs (https://github.com/tkarras/progressive_growing_of_gans). We thank the authors for their great job. This repository goes with paper titled "Geological Facies Modeling Based on Progressive Growing of Generative Adversarial Networks (GANs)" by Suihong Song, Tapan Mukerji, and Jiagen Hou. This repository includes 3 big folders: "Training data", "Code", and "Results of trained generators".

################################################################

Training

Download “Training data” and “Code” folders.
Before start training GANs, set paths and other hyperparameters in “config.py” file. If using conventional GAN training process, uncomment the line of code (“#desc += '-nogrowing' …”) in “config.py” file.
Run code in “Run Code.ipynb” file.
###############################################################

Assessment of the trained generator during or after training

Visual assessment by randomly plotting facies models produced from trained generator and from training facies models: see “Application_and_analyses_of_Trained_model.ipynb” file in such as Jupyter notebook.
Calculate multi-scale sliced Wasserstein distance (MS-SWD) values for different networks during training, distribution of facies models in MDS plot based on MS-SWD for different networks during training, or comparison of facies models produced by conventionally and progressively trained generators against training facies models at the same MDS plot. (1) Uncomment corresponding line of code in the last block of code in “config.py” file (2) If try with our trained generators, modify “dataset” and “result_dir” parameters in “config.txt” in generated result file. (3) Run code in “Run Code.ipynb” file.
Variogram: see “Variogram Ploting.ipynb”.
Channel sinuosity and width: run matlab code “AnalysesofChannelSinuosityandWidth.m” to calculate channel sinuosity and channel width for each facies model. When running the code, every time when one facies model pops up, choose one channel, representative of the facies model, and double click the channel to pop up another facies model. During this process, “endpoints”, “startpoints”, “reallengths” (arclength), and “widths” (half width) can be calculated, thus the sinuosity (arclength/straight-line length) and width of double-clicked channels can be calculated.
