# Segmentation of Lesions & Organs on PSMA PET/CT Images of Metastatic Prostate Cancer Using Self-Supervised Learning Swin UNTER
PSMA PET/CT imaging is widely used for image analysis of prostate cancer. We can seek unknown features by analyzing segmented parts. Since manual segmentation is labor-intensive, automated segmentation methods are preferred. The lack of annotated images makes DL segmentation models challenging. We developed Swin UNETR for automated segmentation of lesions and organs on [68Ga]Ga-PSMA-11 PET/CT images of mCRPC cases.

# Install Dependencies
Install dependencies using:
```bash
pip install -r requirements.txt
```

# Preprocessing
Before pretraining and fine-tuning, data (PET and CT images) should be preprocessed:
```bash
python preprocess.py --in_dir=<Input-directory(PET and CT)> --out_dir=<Output-directory>
```

# Pre-Training
Pre-Train Swin UNETR encoder on unlabeled data
```bash
python main.py --exp=<Experiment Name> --in_channels=2 --data_dir=<Data-Path> --json_list=<Json List Path> \
--lr=6e-6 --lrdecay --batch_size=<Batch Size> --num_steps=<Number of Steps>
```

# Fine-Tuning
Fine-Tuning Swin UNETR on labeled data:
```bash
python main.py --exp=<Experiment Name> --data_dir=<Data-Path> --json_list=<Json List Path> --in_channels=2 --out_channels=12 \
--pretrained_model_name=<Pretrained Encoder Name> --batch_size=<Batch Size> --max_epochs=<Epochs> --use_ssl_pretrained \
--ssl_pretrained_path=<Pretrained Model Path> --use_checkpoint
```

# Evaluation
Evaluating Swin UNETR
```bash

python test.py --pretrained_dir=<Pretrained Model Path> --data_dir=<Data-Path> --exp_name=<Experiment Name> \
--json_list=<Json List Path> --pretrained_model_name=<Pretrained Model Name> --save
```

# Citation
If you find our work useful, please cite the following paper:
```bibtex
@article{YourPaper,
  author = {Yazdani, Elmira, et al.},
  title = {Automated segmentation of lesions and organs at risk on [68Ga] Ga-PSMA-11 PET/CT images using self-supervised learning with Swin UNETR},
  journal = {Cancer Imaging},
  year = {2024},
  doi = {[https://doi.org/10.1186/s40644-024-00675-x]},
}
```

# Acknowledgement
Models Implantation and SSL Pipeline are based on [MONAI](https://github.com/Project-MONAI/MONAI) and [This](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) repository.
