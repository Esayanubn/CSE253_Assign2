import torch
import pretty_midi
import numpy as np
from pathlib import Path
from models.transformer import ChordToMelodyTransformer
import music21

def generate_melody(model, chord_sequence, max_length=32, temperature=1.0, device='cpu'):
    """根据和弦序列生成旋律"""
    model.eval()
    with torch.no_grad():
        # 准备输入
        chord_sequence = torch.tensor(chord_sequence).unsqueeze(0).to(device)
        
        # 生成旋律
        generated = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # 获取模型输出
            output = model(chord_sequence, generated)
            next_note = output[:, -1, :] / temperature
            next_note = torch.softmax(next_note, dim=-1)
            next_note = torch.multinomial(next_note, 1)
            
            # 更新生成序列
            generated = torch.cat([generated, next_note], dim=1)
        
        return generated[0].cpu().numpy()

def play_generated_melody(melody_sequence, output_path='generated_melody.mid', tempo=120):
    """将生成的旋律转换为MIDI并保存"""
    # 创建music21流
    stream = music21.stream.Stream()
    
    # 设置速度
    tempo_mark = music21.tempo.MetronomeMark(number=tempo)
    stream.append(tempo_mark)
    
    # 添加音符
    for note in melody_sequence:
        if note > 0:  # 不是休止符
            n = music21.note.Note(note)
            n.quarterLength = 1.0  # 四分音符
            stream.append(n)
        else:
            r = music21.note.Rest()
            r.quarterLength = 1.0
            stream.append(r)
    
    # 保存为MIDI文件
    stream.write('midi', fp=output_path)
    print(f"生成的音乐已保存到: {output_path}")

def main():
    # 设置参数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'models/best_model.pth'
    
    # 加载模型
    model = ChordToMelodyTransformer(vocab_size=128).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 示例和弦序列（使用MIDI音符编号）
    chord_sequences = [
        [60, 64, 67],  # C大三和弦 (C4, E4, G4)
        [55, 59, 62, 65],  # G7 (G3, B3, D4, F4)
        [53, 57, 60]   # F大三和弦 (F3, A3, C4)
    ]
    
    # 生成旋律
    for i, chord_seq in enumerate(chord_sequences):
        print(f"生成第 {i+1} 个和弦序列的旋律...")
        generated_melody = generate_melody(model, chord_seq, temperature=0.8, device=DEVICE)
        
        # 保存生成的音乐
        output_path = f'generated_melody_{i+1}.mid'
        play_generated_melody(generated_melody, output_path=output_path, tempo=120)

def chord_to_sequence(chord):
    """将和弦转换为数值序列"""
    # 获取和弦的根音（MIDI音符编号）
    root = min(chord)
    # 计算和弦类型
    intervals = [pitch - root for pitch in chord]
    chord_type = get_chord_type(intervals)
    return root

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
    main() 