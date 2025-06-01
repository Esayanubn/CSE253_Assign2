import os
import numpy as np
import torch
from pathlib import Path
import pypianoroll
from tqdm import tqdm
import requests
import tarfile
import json

def download_lpd_dataset():
    """下载LPD数据集的一个子集"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用LPD-5数据集（5个轨道的子集）
    url = "http://hog.ee.columbia.edu/craffel/lmd/lpd_5.tar.gz"
    tar_path = data_dir / "lpd_5.tar.gz"
    
    if not tar_path.exists():
        print("下载LPD-5数据集...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    f.write(data)
                    pbar.update(len(data))
    
    # 解压数据集
    if not (data_dir / "lpd_5").exists():
        print("解压数据集...")
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(data_dir)

def process_lpd_files():
    """处理LPD文件，提取和弦和旋律"""
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    lpd_dir = data_dir / "raw/lpd_5/lpd_5_full/0"
    
    chord_sequences = []
    melody_sequences = []
    
    print("处理LPD文件...")
    for npz_file in tqdm(list(lpd_dir.glob("**/*.npz"))):
        try:
            # 加载pianoroll数据
            multitrack = pypianoroll.load(str(npz_file))
            
            # 获取钢琴轨道（通常包含和弦和旋律）
            piano_track = None
            for track in multitrack.tracks:
                if track.program == 0 and not track.is_drum:  # 钢琴音色
                    piano_track = track
                    break
            
            if piano_track is None:
                continue
            
            # 将pianoroll转换为音符序列
            piano_roll = piano_track.pianoroll
            notes = []
            for time_step in range(piano_roll.shape[0]):
                active_notes = np.where(piano_roll[time_step] > 0)[0]
                if len(active_notes) > 0:
                    notes.append(active_notes.tolist())
            
            if len(notes) >= 32:  # 确保序列长度足够
                # 提取和弦和旋律
                chords = []
                melody = []
                
                for note_group in notes:
                    if len(note_group) >= 3:  # 可能是和弦
                        chords.append(note_group)
                    else:  # 可能是旋律音符
                        melody.append(note_group[0] if note_group else 0)
                
                if chords and melody:
                    # 确保序列长度一致
                    min_length = min(len(chords), len(melody))
                    if min_length >= 32:
                        chord_sequence = [chord_to_sequence(chord) for chord in chords[:min_length]]
                        melody_sequence = melody[:min_length]
                        
                        chord_sequences.append(chord_sequence)
                        melody_sequences.append(melody_sequence)
        
        except Exception as e:
            print(f"处理文件 {npz_file} 时出错: {e}")
    
    if not chord_sequences:
        print("警告：没有找到有效的和弦和旋律对！")
        print("请检查LPD数据集是否正确解压到data/raw/lpd_5目录")
        return
    
    # 保存处理后的数据
    torch.save({
        'chord_sequences': chord_sequences,
        'melody_sequences': melody_sequences
    }, processed_dir / "processed_data.pt")
    
    print(f"处理了 {len(chord_sequences)} 个文件")
    print(f"总序列数: {sum(len(seq) for seq in chord_sequences)}")
    print(f"平均序列长度: {np.mean([len(seq) for seq in chord_sequences]):.2f}")

def chord_to_sequence(chord):
    """将和弦转换为数值序列"""
    # 获取和弦的根音
    root = min(chord)
    # 计算和弦类型
    intervals = [pitch - root for pitch in chord]
    chord_type = get_chord_type(intervals)
    return root * 10 + chord_type

def get_chord_type(intervals):
    """根据音程判断和弦类型"""
    intervals = sorted(intervals)
    if len(intervals) == 3:
        if intervals == [0, 4, 7]:
            return 1  # 大三和弦
        elif intervals == [0, 3, 7]:
            return 2  # 小三和弦
    elif len(intervals) == 4:
        if intervals == [0, 4, 7, 11]:
            return 3  # 大七和弦
        elif intervals == [0, 3, 7, 10]:
            return 4  # 小七和弦
        elif intervals == [0, 4, 7, 10]:
            return 5  # 属七和弦
    return 0  # 其他类型

if __name__ == "__main__":
    # download_lpd_dataset()
    process_lpd_files() 