import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["PUT YOUR 1st TEXT HERE", "PUT YOUR 2nd TEXT HERE"]

wavs = chat.infer(texts)

for i in range(len(wavs)):
    torchaudio.save(f"./data/basic_output{i}.mp3", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
