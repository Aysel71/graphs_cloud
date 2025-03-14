# PointCloudSuperResolution.pytorch

A PyTorch implementation of point cloud super-resolution with modern neural network architectures, including Transformer-based models for high-quality upsampling.

## 🌟 **Features**  
✅ High-quality point cloud upsampling (4x resolution)  
✅ Multiple architecture implementations:  
   - Graph Convolutional Network (GCN)  
   - Transformer-based model  
✅ Optimized for GPU acceleration  
✅ Mixed precision training support  
✅ Compatible with PyTorch Geometric  

---

## 📖 **Overview**  
This repository provides a PyTorch-based implementation for point cloud super-resolution. The goal is to upsample sparse point clouds to higher resolutions using deep learning techniques. The repository includes multiple architectures, including GCN-based models and a novel Transformer-based approach.

---

## 🏗️ **Model Architecture**  

### 🚀 Transformer-based Generator  
1. **Feature Extraction** – A Transformer-based feature extractor captures both local and global relationships between points.  
2. **Upsampling Module** – Two sequential upsampling modules double the number of points at each step, achieving 4x upsampling.  
3. **Feature Propagation** – Efficient feature interpolation transfers features from the original to the generated points.  

### 🔎 GCN-based Generator  
1. **FeatureNetGCN** – Extracts features using graph convolution on k-nearest neighbors.  
2. **ResGraphConvUnpool** – Residual graph convolution blocks for upsampling.  
3. **Feature Interpolation** – Transfers features from original to new points.  

---

## 🛠️ **Installation**  

### **Requirements**  
- Python 3.7+  
- PyTorch 1.10+  
- PyTorch Geometric  
- `torch_scatter`, `torch_sparse`  
- `h5py` (for dataset handling)  

### **Setup**  
```bash
# Create a conda environment
conda create -n point_cloud_env python=3.10
conda activate point_cloud_env

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
pip install torch-scatter torch-sparse h5py tqdm pyyaml
```

---

## 🚦 **Usage**  

### **Training**  
To train the model:  
```bash
python train.py config/train_config_transformer.yaml
```

👉 Different configurations are available:  
- `train_config_res_gcn.yaml` – GCN-based model  
- `train_config_transformer.yaml` – Transformer-based model  

---

### **Evaluation**  
To evaluate a trained model:  
```bash
python eval.py config/eval_config.yaml
```

---

### **Inference**  
For inference on your own point clouds:  
```bash
python inference.py --model_path path/to/model.pth --input path/to/input.ply --output path/to/output.ply
```

---

## 📊 **Dataset**  
The model is trained on the **PU-GAN dataset** (or similar point cloud datasets). Data preparation scripts are included to convert various formats to the required HDF5 format.  

---

## 🏆 **Results**  
Our Transformer-based model achieves state-of-the-art results, outperforming GCN-based approaches in terms of:  
✔️ Point distribution uniformity  
✔️ Surface detail preservation  
✔️ Edge and feature reconstruction  

---

## 📚 **Citing**  
If you use this code in your research, please cite our work:  

```bibtex
@misc{PointCloudSuperResolution,
  author = {Your Name},
  title = {PointCloudSuperResolution.pytorch},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/PointCloudSuperResolution.pytorch}}
}
```

---

## 📝 **License**  
This project is licensed under the **MIT License** – see the LICENSE file for details.

---

## ❤️ **Acknowledgments**  
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) – for the graph neural network primitives  
- [PU-GAN](https://liruihui.github.io/publication/PU-GAN/) – for the dataset and inspiration  

---

💡 **Contributions are welcome!** Feel free to open issues or submit pull requests. 😎


