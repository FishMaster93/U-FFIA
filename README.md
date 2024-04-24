# U-FFIA
Multimodal Fish Feeding Intensity Assessment in Aquaculture (The paper you can find on Arxiv: https://arxiv.org/pdf/2309.05058.pdf)
The dataset you can find on https://zenodo.org/records/11059975

Abstract: The main challenges surrounding FFIA are two-fold. 1) robustness: existing work has mainly leveraged single-modality (e.g., vision, audio) methods, which have a high sensitivity to input noise. 2) efficiency: FFIA models are generally expected to be employed on devices. This presents a challenge in terms of computational efficiency. In this work, we first introduce an audio-visual dataset, AV-FFIA, consisting of 27,000 labeled audio and video clips that capture different levels of fish feeding intensity. Then, we introduce a multi-modal approach for FFIA by leveraging single-modality pre-trained models and modality-fusion methods, with benchmark studies on AV-FFIA, which demonstrate the advantages of the multi-modal approach over the single-modality based approach, especially in noisy environments. While multimodal approaches provide a performance gain for FFIA, it inherently increases the computational cost, as it requires independent encoders to process the input data from the individual modalities. To overcome this issue, we further present a novel unified mixed-modality based method for FFIA, termed as U-FFIA. U-FFIA is a single model capable of processing audio, visual, or audio-visual modalities, by leveraging modality dropout during training and knowledge distillation from single-modality pre-trained models. We demonstrate that U-FFIA can achieve performance better than or on par with the state-of-the-art modality-specific FFIA models, with significantly lower computational overhead.

 Video frames and mel spectrogram visualizations of four different fish feeding intensity: “Strong”, “Medium”, “Weak” and “None”.
 
![image](https://github.com/FishMaster93/U-FFIA/blob/main/fish_feeding-min.png) 


The pipeline of the paper is shown below:
![image](https://github.com/FishMaster93/U-FFIA/blob/main/overall.png) 

