# How to run U-FFIA?
1. Install the environment in `environment.yml`
2. Using the code of `dataset/fish_video_dataset.py` (the decord is a fast tool to read video frames as tensor and save to pickle), you need to download the video dataset and preprocess it.
3. You can run the main.py for audio modality or main_video.py for video modality and mian_av.py for audio-visual fusion; the main_unified.py and main_kl_unfied.py is used for our proposed unified model.
4. You can find the trainer in `tasks/xx.py` and the model located in `models/model_zoo/xx.py`
5. The hyper-parameters can be found at `config/xx/xx.yaml`
6. The pre-trained model folder can be found at: https://drive.google.com/drive/folders/1fh-Lo3S7-aTgfPni5-IeG5_-P7MBKBfL?usp=drive_link


# U-FFIA
Multimodal Fish Feeding Intensity Assessment in Aquaculture (The paper you can find on Arxiv: https://arxiv.org/pdf/2309.05058.pdf)

The dataset can be found at: https://zenodo.org/records/11059975

The pre-trained model folder can be found at: https://drive.google.com/drive/folders/1fh-Lo3S7-aTgfPni5-IeG5_-P7MBKBfL?usp=drive_link


Abstract: The main challenges surrounding FFIA are two-fold. 1) robustness: existing work has mainly leveraged single-modality (e.g., vision, audio) methods, which have a high sensitivity to input noise. 2) efficiency: FFIA models are generally expected to be employed on devices. This presents a challenge in terms of computational efficiency. In this work, we first introduce an audio-visual dataset, AV-FFIA, consisting of 27,000 labeled audio and video clips that capture different levels of fish feeding intensity. Then, we introduce a multi-modal approach for FFIA by leveraging single-modality pre-trained models and modality-fusion methods, with benchmark studies on AV-FFIA, which demonstrate the advantages of the multi-modal approach over the single-modality based approach, especially in noisy environments. While multimodal approaches provide a performance gain for FFIA, it inherently increase the computational cost, as it requires independent encoders to process the input data from the individual modalities. To overcome this issue, we further present a novel unified mixed-modality based method for FFIA, termed as U-FFIA. U-FFIA is a single model capable of processing audio, visual, or audio-visual modalities, by leveraging modality dropout during training and knowledge distillation from single-modality pre-trained models. We demonstrate that U-FFIA can achieve performance better than or on par with the state-of-the-art modality-specific FFIA models, with significantly lower computational overhead.


 Video frames and mel spectrogram visualizations of four different fish feeding intensities: “Strong”, “Medium”, “Weak” and “None”.
 
![image](https://github.com/FishMaster93/U-FFIA/blob/main/fish_feeding-min.png) 


The pipeline of the paper is shown below:
![image](https://github.com/FishMaster93/U-FFIA/blob/main/pipline.png) 
