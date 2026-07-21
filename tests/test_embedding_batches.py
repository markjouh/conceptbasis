import torch
from PIL import Image

from conceptbasis.embedding import image_batches


def test_threaded_image_batches_preserve_serial_values_and_order(tmp_path):
    paths = []
    for value in range(7):
        path = tmp_path / f"{value}.png"
        Image.new("RGB", (2, 2), (value, value, value)).save(path)
        paths.append(str(path))

    def preprocess(image):
        return torch.tensor(list(image.getpixel((0, 0))), dtype=torch.float32)

    serial = list(image_batches(paths, preprocess, 3, workers=0, prefetch_batches=1))
    threaded = list(image_batches(paths, preprocess, 3, workers=4, prefetch_batches=2))

    assert len(threaded) == len(serial)
    for expected, actual in zip(serial, threaded):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
