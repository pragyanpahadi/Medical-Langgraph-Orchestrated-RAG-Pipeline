# Vision model checkpoint

Drop the trained ResNet50 state dict here as:

    vision_model.pth

`vision_module.py` always loads this exact filename (path configurable via
`vision.checkpoint_path` in `config.yaml`, but the convention is to keep the
name fixed). To update to a better model later, just overwrite this file with
a new checkpoint trained the same way (ResNet50 backbone, binary AD/NORMAL head)
— no code changes needed.

This file is git-ignored (see `.gitignore`) since checkpoints are large
binaries; each collaborator downloads their own copy from Colab and drops it
here locally.
