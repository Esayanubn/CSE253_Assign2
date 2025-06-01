import os
import numpy as np
import torch
from pathlib import Path
import pypianoroll
from tqdm import tqdm
import matplotlib.pyplot as plt
import music21

def visualize_tracks(pianoroll, output_dir='test_output'):
    """可视化钢琴轨道和和弦轨道"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取轨道
    piano_track = pianoroll.tracks[1]  # 钢琴轨道
    chord_track = pianoroll.tracks[2]  # 和弦轨道
    
    # 转换为二进制
    piano_track.binarize()
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 绘制钢琴轨道
    ax1.imshow(piano_track.pianoroll.T, aspect='auto', origin='lower')
    ax1.set_title('Piano Track (Melody)')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Pitch')
    
    # 绘制和弦轨道
    ax2.imshow(chord_track.pianoroll.T, aspect='auto', origin='lower')
    ax2.set_title('Chord Track')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Pitch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tracks_visualization.png'))
    plt.close()

def analyze_sequence(pianoroll, start_time=0, length=32):
    """分析指定时间段的音符"""
    piano_track = pianoroll.tracks[1]
    chord_track = pianoroll.tracks[2]
    
    piano_track.binarize()
    
    # 提取指定时间段
    piano_sequence = piano_track[start_time:start_time + length]
    chord_sequence = chord_track[start_time:start_time + length]
    
    # 分析每个时间步
    print("\n时间步分析:")
    for t in range(length):
        # 获取钢琴音符
        piano_notes = piano_sequence[t].nonzero()[0]
        # 获取和弦音符
        chord_notes = chord_sequence[t].nonzero()[0]
        
        print(f"\n时间步 {t}:")
        if len(piano_notes) > 0:
            print(f"钢琴音符: {[music21.note.Note(pitch).nameWithOctave for pitch in piano_notes]}")
        else:
            print("钢琴: 休止符")
            
        if len(chord_notes) > 0:
            print(f"和弦音符: {[music21.note.Note(pitch).nameWithOctave for pitch in chord_notes]}")
            # 尝试识别和弦类型
            chord = music21.chord.Chord([music21.note.Note(pitch) for pitch in chord_notes])
            print(f"和弦类型: {chord.commonName}")
        else:
            print("和弦: 无")

def test_separation():
    """测试和弦和旋律的分离效果"""
    lpd_dir = Path("data/raw/lpd_5/lpd_5_full/0")
    if not lpd_dir.exists():
        print("Error: LPD dataset not found. Please download it first.")
        return
    
    # 获取第一个.npz文件
    npz_files = list(lpd_dir.glob("**/*.npz"))
    if not npz_files:
        print("No .npz files found!")
        return
    
    test_file = npz_files[0]
    print(f"Testing with file: {test_file}")
    
    # 加载pianoroll
    pianoroll = pypianoroll.load(test_file)
    
    # 可视化轨道
    visualize_tracks(pianoroll)
    
    # 分析前32个时间步
    analyze_sequence(pianoroll, start_time=0, length=32)
    
    # 保存为MIDI文件以便试听
    midi_path = os.path.join('test_output', 'test_sample.mid')
    pianoroll.write(midi_path)
    print(f"\nMIDI文件已保存到: {midi_path}")

if __name__ == "__main__":
    test_separation() 