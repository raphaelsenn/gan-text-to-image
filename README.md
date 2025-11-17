# gan-text-to-image
Text-to-image synthesis using text-conditional deep convolutional generative adversarial networks, and techniques described in the paper ["Generative Adversarial Text to Image Synthesis"](https://arxiv.org/abs/1605.05396).

|  Architecture  |  GAN-CLS algorithm  |
| -- | -- |
|![architecture](./assets/architecture.png) | ![gan_cls](./assets/cls_algorithm.png) |
| Architecture of the text-conditional deep convolutional generative adversarial network. Taken from [Reed et al., 2014](https://arxiv.org/abs/1605.05396).| GAN-CLS algorithm used to train the text-conditional deep convolutional generative adversarial network. Taken from [Reed et at., 2014](https://arxiv.org/abs/1605.05396).

## Images generated using this repository

| Caption | Ground truth (validation set) | This repository |
| -------- | ------------------------------| --------------- |
| a living room has a couch, chair, fireplace, windows, and a potted plant. | ![coco_living_room](./assets/1138.png) | ![coco_fake_living_room](./assets/1138_fake.png) |
| this table is filled with a variety of different dishes. | ![coco_food](./assets/196.png) | ![coco_fake_food](./assets/196_fake.png) |
|  a white toilet sitting inside of a bathroom. | ![coco_toilet](./assets/2388.png) | ![coco_toilet_fake](./assets/2388_fake.png) |
|a group of people that are wearing snow skis and holding poles. | ![coco_group](./assets/761.png)| ![coco_group_fake](./assets/761_fake.png) |
| a group of men on a field playing baseball. | ![coco_man_baseball](./assets/357.png) | ![coco_man_baseball_fake](./assets/357_fake.png) |
| a kitchen with a slanted ceiling and skylight. | ![coco_kitchen](./assets/164.png) | ![coco_fake_kitchen](./assets/164_fake.png) |
| a young woman is skiing down the mountain slope. | ![coco_women_skiing](./assets/785.png) | ![coco_fake_women_skiing](./assets/785_fake.png) |
| an airplane is shown taking off into the sky. | ![coco_airplane](./assets/1029.png) | ![coco_airplane_fake](./assets/1029_fake.png) |

## Usage

```bash
python3 generate.py --prompt="a sheep is standing on a green field of gras."
```

| Example usage | Possible output |
|---------------|-----------------|
| `python3 generate.py --prompt="a kite flies over flags posted on a windy beach."` | ![coco_fake_kite](./assets/fake_kite.png) |
| `python3 generate.py --prompt="a food tray with french fries and a sandwich."` | ![coco_fake_fastfood](./assets/fake_fastfood.png) |
| `python3 generate.py --prompt="a sheep on a green field of gras."` | ![coco_fake_sheep](./assets/fake_sheep.png) |


## Reproduction

* Install all requiered software in [requirements.txt](requirements.txt).

* Download the [COCO 2014 dataset](https://cocodataset.org/#download).

* Ensure the correct path to COCO (images and ann.json) in the files: [train_text.py](./train_text.py), [train.py](train.py) and [prepro.py](prepro.py) (inserting the correct path into the config).

1. Pre-train the deep convolutional-recurrent text encoder on structured joint embedding loss as follows:

```bash
python3 train_text_encoder.py --epochs=200
--batch_size=64 --device="cuda"
```

2. Use the trained CNN-RNN text encoder network to embed the text descriptions:

```bash
python3 prepro.py --device="cuda"
```
3. Train the text-conditioned deep convolutional generative adversarial network:

```bash
python3 train.py --epochs=200 --batch_size=64 --device="cuda"
```

## Experimental setup

* OS: Fedora Linux 42 (Workstation Edition) x86_64
* CPU: AMD Ryzen 5 2600X (12) @ 3.60 GHz
* GPU: NVIDIA GeForce RTX 3060 ti (8GB VRAM)
* RAM: 32 GB DDR4 3200 MHz

Training takes approximately 16 hours on my setup.

## Citations

```bibtex
@misc{reed2016generativeadversarialtextimage,
      title={Generative Adversarial Text to Image Synthesis}, 
      author={Scott Reed and Zeynep Akata and Xinchen Yan and Lajanugen Logeswaran and Bernt Schiele and Honglak Lee},
      year={2016},
      eprint={1605.05396},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/1605.05396}, 
}
```

```bibtex
@misc{radford2016unsupervisedrepresentationlearningdeep,
      title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks}, 
      author={Alec Radford and Luke Metz and Soumith Chintala},
      year={2016},
      eprint={1511.06434},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1511.06434}, 
}
```

```bibtex
@misc{reed2016learningdeeprepresentationsfinegrained,
      title={Learning Deep Representations of Fine-grained Visual Descriptions}, 
      author={Scott Reed and Zeynep Akata and Bernt Schiele and Honglak Lee},
      year={2016},
      eprint={1605.05395},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1605.05395}, 
}
```