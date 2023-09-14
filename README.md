# ErosionAttack

### Overview

Erosion Attack: Harnessing Corruption To Improve Adversarial Examples

The code is repository for ["Erosion Attack: Harnessing Corruption To Improve Adversarial Examples"](https://ieeexplore.ieee.org/abstract/document/10102830) (IEEE TIP).


### Prerequisites

python **3.6**  
tensorflow **1.14**  

### Pipeline 
<img src="/figure/overview.png" width = "1000" height = "200" align=center/>

### Run the Code  
Train single erosion network used in single-model attacks or multiple erosion networks used in ensemble-attacks. 

(1) `train_aux_i3.py` for InceptionV3, 

(2) `train_aux_i4.py` for InceptionV4, 

(3) `train_aux_ir2.py` for InceptionResNetV2, 

(4) `train_aux_r50.py` for ResNet50. 

Pre-trained weights of erosion networks above are available at [here](https://drive.google.com/drive/folders/1hHoj7WX9EftMEqZ4sL50Ygiu6_41PJDV?usp=drive_link). 

Run EA to generate adversarial examples: `erosion_attack.py`.  

### Experimental Results

<b>Standalone Experiment</b>

<img src="/figure/exp1.png" width = "500" height = "300" align=center/>

<b>Combination Experiment</b>

<img src="/figure/exp2.png" width = "1000" height = "370" align=center/>

<3>Ensemble Experiment</b>

<img src="/figure/exp3.png" width = "1000" height = "280" align=center/>

### Citation
If you find this project is useful for your research, please consider citing:

	@article{huang2023erosion,
	  title={Erosion Attack: Harnessing Corruption To Improve Adversarial Examples},
	  author={Huang, Lifeng and Gao, Chengying and Liu, Ning},
	  journal={IEEE Transactions on Image Processing},
	  year={2023},
	  publisher={IEEE}
	}




