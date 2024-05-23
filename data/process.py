#
# Created on Thu May 23 2024
#
# The MIT License (MIT)
# Copyright (c) 2024 WaveAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

"""
Module to build the dataset
"""


from data.codec.codec import Codec
import typing
import os


class DatasetBuilder:
    def __init__(self, codec: typing.List[type[Codec]]):
        self._codec = codec

    def build_dataset(self, source_path: str, target_path: str) -> None:
        # in development

        for file_name in os.listdir(source_path):
            file_path = os.path.join(source_path, file_name)
            for codec in self._codec:
                codec_instance = codec()
                encoded_audio = codec_instance.encode(file_path)
                target_file_path = os.path.join(
                    target_path, f"{file_name}.{codec_instance.__class__.__name__}"
                )
                codec_instance.save(target_file_path, encoded_audio)
