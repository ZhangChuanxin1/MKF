# Local dataset preparation

No silent dataset/model substitution or automatic download occurs. Large
images, trained weights and ciphertext/key fixtures are supplied locally.

## MNIST

KKLSS CNN expects `software/kklss/cnn/data/mnist_test.csv`: exactly 10,000
rows, 785 integers per row, label then 28x28 pixel bytes, no header and **no
trailing blank line**. The Go reader divides pixels by 255; do not pre-scale.

Provide original test IDX files, gzip-compressed or raw:

```powershell
py -3 scripts/convert_mnist_idx.py `
  --images .local/data/t10k-images-idx3-ubyte.gz `
  --labels .local/data/t10k-labels-idx1-ubyte.gz `
  --output software/kklss/cnn/data/mnist_test.csv
```

The converter checks headers, dimensions, counts, payload sizes and labels,
refuses overwrite and writes SHA-256 metadata. Generated files are ignored
by Git. Included weights are unchanged; conversion does not prove training
provenance or manuscript accuracy.

## CIFAR-10 / ResNet-20

- Frozen model source commit: `786c16252c0fc58ee9adac063f8337cc4a7a497a`, included under third_party.
- Checkpoint: `cifar10_resnet20-4118986f.pt`, SHA-256
  `4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8`.
- Official CIFAR Python archive: 170,498,071 bytes; MD5
  `c58f30108f718f92721af3b95e74349a`, checked by the reference tool.
- Folded weights, FC bias, calibrated scales and Go fixtures must come from
  the same audited source run and be supplied separately.

Pass explicit asset paths to the plaintext tools. Do not calibrate training
ranges using held-out test data. Synthetic smoke inputs are not CIFAR accuracy
measurements. Review redistribution rights before adding external assets.
