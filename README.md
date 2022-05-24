
# Exploring Cross-Video and Cross-Modality Signals for Weakly-Supervised Audio-Visual Video Parsing 

<img src="https://raw.githubusercontent.com/facebookresearch/unbiased-teacher/main/teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the PyTorch implementation of our paper: <br>
**Exploring Cross-Video and Cross-Modality Signals for Weakly-Supervised Audio-Visual Video Parsing**<br>
[Yan-Bo Lin](https://genjib.github.io/), [Hung-Yu Tseng](https://hytseng0509.github.io/), [Hsin-Ying Lee](http://hsinyinglee.com/), [Yen-Yu Lin](https://sites.google.com/site/yylinweb/), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
Advances in Neural Information Processing Systems (NeurIPS), 2021 <br>

[paper](https://openreview.net/pdf?id=V5V1vGrI2z) | [dataset](https://github.com/YapengTian/AVVP-ECCV20) 

### 📝 Preparation 
1. `pip3 install requirements.txt`
2. Following [AVVP](https://github.com/YapengTian/AVVP-ECCV20), prepare pre-extracted features in `.feats/r2plus1d_18`, `.feats/res152`, and `.feats/vggish`


### 📚 Train and evaluate

Simply run `bash run.sh`


## 🎓 Cite

If you use this code in your research, please cite:

```bibtex
@article{lin2021exploring,
  title={Exploring Cross-Video and Cross-Modality Signals for Weakly-Supervised Audio-Visual Video Parsing},
  author={Lin, Yan-Bo and Tseng, Hung-Yu and Lee, Hsin-Ying and Lin, Yen-Yu and Yang, Ming-Hsuan},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## License

This project is licensed under CC-BY-NC 4.0 License, as found in the LICENSE file.