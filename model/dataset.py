from torch.utils.data import Dataset
import torchaudio
import random
import glob

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir):
        super().__init__()
        
        self.filenames = glob.glob(audio_dir+"/*.mp3")
        _, self.sr = torchaudio.load(self.filenames[0])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self.filenames[index])
   
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            16000,
            dtype=waveform.dtype
        )
        waveform = resampler(waveform)
        audio = waveform.mean(dim=0, keepdim=True)

        crop = random.randint(0, 390000)
        return audio[:, crop:crop+16000*5]
