import torch
import pretty_midi
import numpy as np
from pathlib import Path
from models.transformer import ChordToMelodyTransformer
import music21

def generate_melody(model, chord_sequence, max_length=128, temperature=1.0, device='cuda'):
    """根据和弦序列生成旋律"""
    model.eval()
    with torch.no_grad():
        # 准备输入
        chord_sequence = torch.tensor(chord_sequence).unsqueeze(0).to(device)
        
        # 生成旋律
        generated = []
        current_input = chord_sequence
        
        for _ in range(max_length):
            output = model(current_input, max_len=current_input.size(1) + 1)
            next_note = output[:, -1, :] / temperature
            next_note = torch.softmax(next_note, dim=-1)
            next_note = torch.multinomial(next_note, 1)
            generated.append(next_note.item())
            
            # 更新输入
            current_input = torch.cat([current_input, next_note], dim=1)
        
        return generated

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
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 示例和弦序列
    # C大三和弦
    c_major = music21.chord.Chord(['C4', 'E4', 'G4'])
    # G属七和弦
    g7 = music21.chord.Chord(['G4', 'B4', 'D5', 'F5'])
    # F大三和弦
    f_major = music21.chord.Chord(['F4', 'A4', 'C5'])
    
    # 将和弦转换为序列
    chord_sequence = [
        chord_to_sequence(c_major),
        chord_to_sequence(g7),
        chord_to_sequence(f_major)
    ]
    
    # 生成旋律
    generated_melody = generate_melody(model, chord_sequence, temperature=0.8)
    
    # 保存生成的音乐
    play_generated_melody(generated_melody, tempo=120)

def chord_to_sequence(chord):
    """将和弦转换为数值序列"""
    # 获取和弦的根音
    root = chord.root().midi
    # 计算和弦类型
    chord_type = get_chord_type(chord)
    return root * 10 + chord_type

def get_chord_type(chord):
    """获取和弦类型"""
    if len(chord.pitches) == 3:
        if chord.isMajorTriad():
            return 1  # 大三和弦
        elif chord.isMinorTriad():
            return 2  # 小三和弦
    elif len(chord.pitches) == 4:
        if chord.isDominantSeventh():
            return 3  # 属七和弦
        elif chord.isMajorSeventh():
            return 4  # 大七和弦
        elif chord.isMinorSeventh():
            return 5  # 小七和弦
    return 0  # 其他类型

if __name__ == "__main__":
    main() 