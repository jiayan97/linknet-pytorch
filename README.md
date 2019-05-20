# LinkNet

*Now in experimental release, suggestions welcome*.

This is a Pytorch reimplementation of [LinkNet](http://cn.arxiv.org/pdf/1811.06410v1) for Scene Graph Generation.

Core code is [rel_model_linknet.py](https://github.com/jiayan97/linknet-pytorch/blob/master/lib/rel_model_linknet.py), built on top of [neural-motifs](https://github.com/rowanz/neural-motifs).

## Setup

* Install Python3.6 & PyTorch3.  ``` conda install pytorch=0.3.0 torchvision=0.2.0 cuda90 -c pytorch ```
* Download Visual Genome dataset, see data/stanford_filtered/README.md for details.
* Compile everything, run ```make``` in the main directory.
* Fix PYTHONPATH. ```export PYTHONPATH=/data/yjy/Workspace/linknet```
* Click [here](https://blog.csdn.net/weixin_38651565/article/details/87901172) for more detailed instructions.


## Train

* Train Object Detection  ([You can also download the pretrained detector checkpoint here.](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX))

```
CUDA_VISIBLE_DEVICES=0,1,2 python models/train_detector.py -b 6 -lr 1e-3 -save_dir checkpoints/vgdet -nepoch 50 -ngpu 3 -nwork 3 -p 100 -clip 5
```

* Train Scene Graph Classification

```
CUDA_VISIBLE_DEVICES=0 python models/train_rels.py -m sgcls -model linknet -b 6 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/linknet-sgcls -nepoch 50 -use_bias
```

* Refine Scene Graph Detection

```
CUDA_VISIBLE_DEVICES=0 python models/train_rels.py -m sgdet -model linknet -b 6 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/linknet-sgcls/vgrel-10.tar -save_dir checkpoints/linknet-sgdet -nepoch 10 -use_bias
```

## Test

* Evaluate Predicate Classification

```
CUDA_VISIBLE_DEVICES=0 python models/eval_rels.py -m predcls -model linknet -b 6 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/linknet-sgcls/vgrel-10.tar -nepoch 50 -use_bias -cache linknet_predcls
```

* Evaluate Scene Graph Classification

```
CUDA_VISIBLE_DEVICES=0 python models/eval_rels.py -m sgcls -model linknet -b 6 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/linknet-sgcls/vgrel-10.tar -nepoch 50 -use_bias -cache linknet_sgcls
```

* Evaluate Scene Graph Detection

```
CUDA_VISIBLE_DEVICES=0 python models/eval_rels.py -m sgdet -model linknet -b 6 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/linknet-sgdet/vgrel-18.tar -nepoch 50 -use_bias -cache linknet_sgdet
```

## Result

|            Mode            | R@20 | R@50 | R@100 |
| :------------------------: | :--: | :--: | :---: |
|  Predicate Classification  | 58.8 | 65.5 | 67.4  |
| Scene Graph Classification | 32.6 | 35.5 | 36.1  |
| Scene Graph Detection      | 13.6 | 20.5 | 25.0  |

## TODO

* Use Faster RCNN with a ResNet backbone

## Contact

For any question, please contact:

```
Jiayan Yang: jiayanyang97@gmail.com
Zhiwei Dong: kivee@foxmail.com
```