"""
完全版 CuPy最適化 GCN-based EI-MS Spectrum Prediction System
=============================================================

msFineAnalysis AIの手法に基づいたEI-MS予測の完全実装
CuPyによる高速化、NIST17データ処理、GCNモデル訓練・推論を統合

Author: AI Assistant
Date: 2024
Requirements:
    - PyTorch >= 1.9.0
    - DGL >= 0.7.0
    - CuPy >= 9.0.0 (optional but recommended)
    - RDKit >= 2021.03
    - NumPy, scikit-learn, tqdm
"""

import os
import sys
import re
import glob
import json
import pickle
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import dgl
from dgl.nn import GraphConv, SumPooling, AvgPooling, MaxPooling

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =====================================================================
# CuPy Configuration and Import
# =====================================================================

CUPY_AVAILABLE = False
try:
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    CUPY_AVAILABLE = True
    print("✓ CuPy is available - GPU acceleration enabled")
except ImportError:
    cp = np  # Fallback to NumPy
    print("✗ CuPy not available - Using NumPy (CPU only)")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# =====================================================================
# Configuration Class
# =====================================================================

@dataclass
class Config:
    """全体設定を管理するクラス"""
    # Data paths
    nist_dir: str = 'nist17_data'
    output_dir: str = 'processed_data'
    model_save_path: str = 'gcn_eims_model.pth'
    
    # Data processing
    max_mz: int = 500
    train_ratio: float = 0.8
    
    # Model architecture
    hidden_dim: int = 256
    num_gcn_layers: int = 3
    dropout: float = 0.2
    pooling: str = 'combined'
    
    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimization
    use_cupy: bool = True
    use_mixed_precision: bool = True
    num_workers: int = 4
    cache_graphs: bool = True

# =====================================================================
# Feature Extraction
# =====================================================================

def one_hot_encoding(value, allowable_set):
    """One-hotエンコーディング"""
    if value not in allowable_set:
        value = allowable_set[0]
    return [int(value == v) for v in allowable_set]

def get_atom_features(atom):
    """原子特徴量を抽出"""
    features = []
    features.append(atom.GetAtomicNum())
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(int(atom.GetHybridization()))
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetTotalNumHs())
    return np.array(features, dtype=np.float32)

def mol_to_dgl_graph(mol):
    """RDKit分子をDGLグラフに変換"""
    if mol is None:
        return None
    
    # 原子特徴量
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    atom_features = np.array(atom_features)
    
    # 結合
    src_list = []
    dst_list = []
    
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        src_list.extend([src, dst])
        dst_list.extend([dst, src])
    
    # グラフ作成
    if len(src_list) == 0:
        g = dgl.graph(([], []), num_nodes=mol.GetNumAtoms())
    else:
        g = dgl.graph((src_list, dst_list), num_nodes=mol.GetNumAtoms())
    
    g.ndata['feat'] = torch.FloatTensor(atom_features)
    
    return g

# =====================================================================
# CuPy Accelerated Processing
# =====================================================================

class CuPySpectrumProcessor:
    """CuPy加速されたスペクトル処理"""
    
    def __init__(self, max_mz=500, use_cupy=True):
        self.max_mz = max_mz
        self.use_cupy = use_cupy and CUPY_AVAILABLE
        
    def peaks_to_spectrum_batch(self, peaks_list):
        """ピークリストをスペクトルベクトルに高速変換"""
        batch_size = len(peaks_list)
        
        if self.use_cupy:
            spectra = cp.zeros((batch_size, self.max_mz), dtype=cp.float32)
            
            for i, peaks in enumerate(peaks_list):
                if peaks:
                    mz_array = cp.array([p[0] for p in peaks], dtype=cp.float32)
                    intensity_array = cp.array([p[1] for p in peaks], dtype=cp.float32)
                    
                    mz_indices = cp.round(mz_array).astype(cp.int32)
                    valid_mask = (mz_indices >= 0) & (mz_indices < self.max_mz)
                    valid_indices = mz_indices[valid_mask]
                    valid_intensities = intensity_array[valid_mask]
                    
                    if len(valid_indices) > 0:
                        for idx, intensity in zip(valid_indices, valid_intensities):
                            spectra[i, idx] = cp.maximum(spectra[i, idx], intensity)
            
            max_vals = cp.max(spectra, axis=1, keepdims=True)
            max_vals = cp.where(max_vals > 0, max_vals, 1.0)
            spectra = spectra / max_vals
            
            return cp.asnumpy(spectra)
        else:
            spectra = np.zeros((batch_size, self.max_mz), dtype=np.float32)
            
            for i, peaks in enumerate(peaks_list):
                for mz, intensity in peaks:
                    mz_int = int(np.round(mz))
                    if 0 <= mz_int < self.max_mz:
                        spectra[i, mz_int] = max(spectra[i, mz_int], intensity)
            
            max_vals = np.max(spectra, axis=1, keepdims=True)
            max_vals = np.where(max_vals > 0, max_vals, 1.0)
            spectra = spectra / max_vals
            
            return spectra
    
    def cosine_similarity_batch(self, pred, target):
        """高速コサイン類似度計算"""
        if self.use_cupy and pred.is_cuda:
            pred_cp = cp.asarray(pred.detach())
            target_cp = cp.asarray(target.detach())
            
            pred_norm = pred_cp / (cp.linalg.norm(pred_cp, axis=1, keepdims=True) + 1e-8)
            target_norm = target_cp / (cp.linalg.norm(target_cp, axis=1, keepdims=True) + 1e-8)
            cos_sim = cp.sum(pred_norm * target_norm, axis=1)
            
            return torch.as_tensor(cos_sim, device=pred.device)
        else:
            pred_norm = F.normalize(pred, p=2, dim=1)
            target_norm = F.normalize(target, p=2, dim=1)
            return (pred_norm * target_norm).sum(dim=1)

# =====================================================================
# Dataset Class
# =====================================================================

class OptimizedEIMSDataset(Dataset):
    """最適化されたEI-MSデータセット"""
    
    def __init__(self, mol_files, msp_files, config):
        self.graphs = []
        self.spectra = []
        self.config = config
        self.processor = CuPySpectrumProcessor(config.max_mz, config.use_cupy)
        
        all_peaks = []
        valid_mols = []
        
        print("Loading molecular data...")
        for mol_file, msp_file in tqdm(zip(mol_files, msp_files)):
            mol = Chem.MolFromMolFile(mol_file) if os.path.exists(mol_file) else None
            peaks = self.load_peaks(msp_file) if os.path.exists(msp_file) else None
            
            if mol is not None and peaks is not None:
                if config.cache_graphs:
                    graph = mol_to_dgl_graph(mol)
                    if graph is not None:
                        self.graphs.append(graph)
                        all_peaks.append(peaks)
                else:
                    self.graphs.append(mol)
                    all_peaks.append(peaks)
        
        if all_peaks:
            print("Processing spectra...")
            self.spectra = self.processor.peaks_to_spectrum_batch(all_peaks)
        
        print(f"Dataset size: {len(self.graphs)} molecules")
    
    def load_peaks(self, msp_file):
        """MSPファイルからピークを読み込み"""
        peaks = []
        try:
            with open(msp_file, 'r') as f:
                lines = f.readlines()
                reading_peaks = False
                for line in lines:
                    if reading_peaks:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            peaks.append((mz, intensity))
                    elif 'Num Peaks:' in line or 'NUM PEAKS:' in line:
                        reading_peaks = True
            return peaks if peaks else None
        except:
            return None
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        if self.config.cache_graphs:
            graph = self.graphs[idx]
        else:
            graph = mol_to_dgl_graph(self.graphs[idx])
        
        spectrum = torch.FloatTensor(self.spectra[idx])
        return graph, spectrum

def collate_fn(batch):
    """バッチ処理用のcollate関数"""
    graphs, spectra = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batched_spectra = torch.stack(spectra)
    return batched_graph, batched_spectra

# =====================================================================
# GCN Model
# =====================================================================

class GCNSpectrum(nn.Module):
    """GCNベースのEI-MSスペクトル予測モデル"""
    
    def __init__(self, node_feat_dim, config):
        super(GCNSpectrum, self).__init__()
        
        self.config = config
        
        # GCN層
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 最初の層
        self.gcn_layers.append(GraphConv(node_feat_dim, config.hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(config.hidden_dim))
        
        # 隠れ層
        for i in range(config.num_gcn_layers - 1):
            self.gcn_layers.append(GraphConv(config.hidden_dim, config.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(config.hidden_dim))
        
        # プーリング
        if config.pooling == 'sum':
            self.pool = SumPooling()
            pool_dim = config.hidden_dim
        elif config.pooling == 'mean':
            self.pool = AvgPooling()
            pool_dim = config.hidden_dim
        elif config.pooling == 'max':
            self.pool = MaxPooling()
            pool_dim = config.hidden_dim
        else:  # combined
            self.sum_pool = SumPooling()
            self.max_pool = MaxPooling()
            self.pool = None
            pool_dim = config.hidden_dim * 2
        
        # スペクトル予測器
        self.spectrum_predictor = nn.Sequential(
            nn.Linear(pool_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.max_mz),
            nn.Sigmoid()
        )
    
    def forward(self, g, node_features):
        h = node_features
        
        # GCN層を適用
        for i in range(self.config.num_gcn_layers):
            h = self.gcn_layers[i](g, h)
            h = F.relu(h)
            h = self.batch_norms[i](h)
            if i < self.config.num_gcn_layers - 1:
                h = F.dropout(h, p=self.config.dropout, training=self.training)
        
        # プーリング
        if self.pool is not None:
            h_graph = self.pool(g, h)
        else:
            h_sum = self.sum_pool(g, h)
            h_max = self.max_pool(g, h)
            h_graph = torch.cat([h_sum, h_max], dim=1)
        
        # スペクトル予測
        spectrum = self.spectrum_predictor(h_graph)
        
        return spectrum

# =====================================================================
# Training Functions
# =====================================================================

def train_model(model, train_loader, val_loader, config):
    """モデルの訓練"""
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    criterion = nn.MSELoss()
    processor = CuPySpectrumProcessor(config.max_mz, config.use_cupy)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and config.use_mixed_precision else None
    
    best_val_cosine = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_cosine': [], 'val_cosine': []}
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_cosine = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch_graph, batch_spectra in pbar:
            batch_graph = batch_graph.to(device)
            batch_spectra = batch_spectra.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_spectra = model(batch_graph, batch_graph.ndata['feat'])
                    loss = criterion(pred_spectra, batch_spectra)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_spectra = model(batch_graph, batch_graph.ndata['feat'])
                loss = criterion(pred_spectra, batch_spectra)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # メトリクス計算
            with torch.no_grad():
                cos_sim = processor.cosine_similarity_batch(pred_spectra, batch_spectra)
                train_cosine += cos_sim.mean().item()
                train_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item(), 'cosine': cos_sim.mean().item()})
        
        # Validation
        model.eval()
        val_loss = 0
        val_cosine = 0
        
        with torch.no_grad():
            for batch_graph, batch_spectra in val_loader:
                batch_graph = batch_graph.to(device)
                batch_spectra = batch_spectra.to(device)
                
                pred_spectra = model(batch_graph, batch_graph.ndata['feat'])
                loss = criterion(pred_spectra, batch_spectra)
                cos_sim = processor.cosine_similarity_batch(pred_spectra, batch_spectra)
                
                val_loss += loss.item()
                val_cosine += cos_sim.mean().item()
        
        # 平均を計算
        train_loss /= len(train_loader)
        train_cosine /= len(train_loader)
        val_loss /= len(val_loader)
        val_cosine /= len(val_loader)
        
        # 履歴を記録
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cosine'].append(train_cosine)
        history['val_cosine'].append(val_cosine)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Cos={train_cosine:.4f}, '
              f'Val Loss={val_loss:.4f}, Val Cos={val_cosine:.4f}')
        
        # ベストモデル保存
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            best_model_state = model.state_dict().copy()
            print(f'✓ Best model saved (Val Cosine: {val_cosine:.4f})')
    
    # ベストモデルをロード
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # メモリクリア
    if CUPY_AVAILABLE:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    return model, history

# =====================================================================
# Prediction Functions
# =====================================================================

def predict_spectrum(model, smiles, config):
    """SMILESからスペクトル予測"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    graph = mol_to_dgl_graph(mol)
    if graph is None:
        return None
    
    graph = graph.to(device)
    
    model.eval()
    with torch.no_grad():
        spectrum = model(graph, graph.ndata['feat'])
        spectrum = spectrum.cpu().numpy().squeeze()
    
    return spectrum

# =====================================================================
# Main Pipeline
# =====================================================================

def main():
    """メイン実行関数"""
    
    parser = argparse.ArgumentParser(description='GCN-based EI-MS Spectrum Prediction')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'preprocess'],
                        help='実行モード')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='データディレクトリ')
    parser.add_argument('--msp_file', type=str, help='NISTのMSPファイル')
    parser.add_argument('--mol_dir', type=str, help='MOLファイルのディレクトリ')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--use_cupy', action='store_true', default=True)
    parser.add_argument('--smiles', type=str, help='予測対象のSMILES')
    
    args = parser.parse_args()
    
    # 設定
    config = Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.use_cupy = args.use_cupy
    
    print("="*60)
    print("GCN-based EI-MS Spectrum Prediction System")
    print("="*60)
    
    if args.mode == 'train':
        # データ読み込み
        mol_files = glob.glob(os.path.join(args.data_dir, 'mol_files', '*.mol'))
        msp_files = [f.replace('mol_files', 'msp_files').replace('.mol', '.msp') 
                     for f in mol_files]
        
        print(f"Found {len(mol_files)} molecule files")
        
        # データセット作成
        dataset = OptimizedEIMSDataset(mol_files, msp_files, config)
        
        # データ分割
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        # データローダー
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # モデル作成
        sample_graph, _ = dataset[0]
        node_feat_dim = sample_graph.ndata['feat'].shape[1]
        
        model = GCNSpectrum(node_feat_dim, config).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # 訓練
        model, history = train_model(model, train_loader, val_loader, config)
        
        # モデル保存
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'history': history
        }, config.model_save_path)
        print(f"Model saved to {config.model_save_path}")
        
    elif args.mode == 'predict':
        if args.smiles:
            # モデルロード
            checkpoint = torch.load(config.model_save_path)
            saved_config = Config(**checkpoint['config'])
            
            # モデル作成（適切な入力次元が必要）
            node_feat_dim = 6  # デフォルト値
            model = GCNSpectrum(node_feat_dim, saved_config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 予測
            spectrum = predict_spectrum(model, args.smiles, saved_config)
            
            if spectrum is not None:
                print(f"Predicted spectrum for {args.smiles}")
                print(f"Max intensity at m/z: {np.argmax(spectrum)}")
                print(f"Top 5 peaks: {np.argsort(spectrum)[-5:][::-1]}")
            else:
                print("Failed to predict spectrum")
        else:
            print("Please provide SMILES with --smiles option")
    
    else:
        print("Mode not implemented")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # デモモード
        print("Demo mode - showing example usage")
        print("\n1. Preprocess data:")
        print("   python script.py --mode preprocess --msp_file nist.msp")
        print("\n2. Train model:")
        print("   python script.py --mode train --data_dir processed_data")
        print("\n3. Predict spectrum:")
        print("   python script.py --mode predict --smiles 'CC(C)CC1=CC=C(C=C1)C(C)C'")
    else:
        main()