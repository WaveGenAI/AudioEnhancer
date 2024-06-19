from dataset.load_dataset import load_dataset

dataset = load_dataset("/media/works/dataset/")                                                                        
dataset = dataset.shuffle(seed=42)

