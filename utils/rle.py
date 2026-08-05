import re
import numpy as np

def rle_encode(mask: np.ndarray, left: int, top: int, right: int, bottom: int):
    pixels = mask.flatten(order="C")
    diff = np.diff(pixels)
    change_idx = np.where(diff != 0)[0] + 1
    splits = np.split(pixels, change_idx)

    result = [(block[0], len(block)) for block in splits]

    rle = []
    idx = 0
    for val, count in result:
        if idx == 0:
            if val == 1:  
                rle.append(0)              
                rle.append(float(count))  
            else:
                rle.append(float(count))
            idx = 1
        else:
            rle.append(count)

    # bbox
    rle.extend([float(left), float(top), float(right), float(bottom)])
    return rle


def rle_decode(rle, shape):
    """Decode RLE from CVAT XML string or CVAT backup list."""

    if isinstance(rle, str):
        rle_numbers = [int(num) for num in re.findall(r"\d+", rle)]

    elif isinstance(rle, list):
        # Backup stores RLE followed by left, top, right, bottom
        if len(rle) < 5:
            raise ValueError("Invalid backup RLE.")
        rle_numbers = [int(v) for v in rle[:-4]]

    else:
        raise TypeError(f"Unsupported RLE type: {type(rle)}")

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    index = 0
    i = 0

    while i < len(rle_numbers):
        start = rle_numbers[i]
        i += 1

        length = 0
        if i < len(rle_numbers):
            length = rle_numbers[i]
            i += 1

        index += start

        if length:
            if index + length > img.size:
                raise ValueError(
                    f"RLE segment exceeds mask size: "
                    f"index={index}, length={length}, size={img.size}"
                )

            img[index:index + length] = 1
            index += length

    return img.reshape(shape)