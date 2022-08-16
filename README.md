# RAVDESS2Mix
RAVDESS2Mix is the test set created to study emotional affection for target speech extraction (TSE) and blind speech separation (BSS).

## Getting started
Install the packages
```bash
pip install -r requirements.txt
```
Generate testset:
```bash
$ python --version
Python 3.9.12

$ cd RAVDESS2Mix

# Download RAVDESS (speech)
$ wget -cO - https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1 > RAVDESS_audio_speech.zip && unzip RAVDESS_audio_speech.zip -d RAVDESS_audio_speech

# Download meta-list file from LibriMix repository
$ wget https://raw.githubusercontent.com/JorisCos/LibriMix/master/metadata/LibriSpeech/test-clean.csv

# Main prepare script
$ python prepare_RAVDESS2Mix.py --librispeech-dir LibriSpeech --ravdess-dataset-dir RAVDESS_audio_speech --librispeech-test-meta-file test-clean.csv --ravdess2mix-dir RAVDESS2Mix
```

## Citations
In case of using the dataset please cite:\
J. Svec, K. Zmolikova, M. Kocour, M. Delcroix, T. Ochiai, L. Mosner, J. Cernocky: [Analysis of impact of emotions on target speech extraction and speech separation](https://arxiv.org/abs/2208.07091).
```
@article{svec2022analysis,
  title={Analysis of impact of emotions on target speech extraction and speech separation},
  author={\v{S}vec, J{\'a}n and \v{Z}molikov{\'a}, Kate\v{r}ina and Kocour, Martin and Delcroix, Marc and Ochiai, Tsubasa and Mo\v{s}ner, Ladislav and \v{C}ernock\'{y}, Jan}
  journal={IWAENC},
  year={2022},
}
```
This work using the Ryerson Audio-Visual Database of Emotional Speech and Song ([RAVDESS](https://zenodo.org/record/1188976#.YvuBWC8Ro6h)) ([article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)).

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Contact
If you have any comment or question, please contact isvecjan@fit.vutbr.cz
