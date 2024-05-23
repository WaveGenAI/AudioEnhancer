import data

bulder = data.DatasetBuilder([data.codec.Encodec()])
bulder.build_dataset("data/audio/base", "data/audio/compressed")
