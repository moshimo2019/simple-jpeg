from collections import OrderedDict
import numpy as np
from math import ceil
import cv2
from io import BytesIO
from pathlib import Path
from PIL import Image

from utils import *
from huffman import *


def decode_header(reader: BytesIO):
    read = reader.read
    assert read(2) == MARKER.SOI  # Start of Image
    marker = read(2)
    if marker == MARKER.APP0:  # Application Marker 0
        APP0 = dict(
            length=read(2),
            identifier=read(5),
            version=read(2),  # b'\x01\x01'
            unit=read(1),
            x_density=read(2),
            y_density=read(2),
        )
        APP0['thumbnail-data'] = read(bytes2int(APP0['length']) - 14)
        marker = read(2)

    APPns = []
    while MARKER.APPn[0] <= marker <= MARKER.APPn[1]:  # Application Marker n
        APPn = dict(marker=marker, length=read(2))
        APPns.append(APPn)
        APPn['content'] = read(bytes2int(APPn['length']) - 2)
        marker = read(2)

    qts = {}
    while marker == MARKER.DQT:  # Define Quantization Table
        length, byte = bytes2int(read(2)), read(1)[0]
        precision, id_ = byte >> 4, byte & 0b1111
        assert precision == 0, 'only support uint8'
        table_size = (precision + 1) * 64
        assert length == table_size + 3
        dtype = np.uint8 if precision == 0 else np.uint16
        qts[id_] = np.frombuffer(
            read(table_size), dtype=dtype, count=64).astype(np.int32)
        marker = read(2)

    assert marker == MARKER.SOF0  # Start of Frame
    length = bytes2int(read(2))
    precision = read(1)[0]
    assert precision == 8, 'only support 8'
    height, width = bytes2int(read(2)), bytes2int(read(2))
    cop_num = read(1)[0]  # 1: grayscale; 3: YCrCb;
    assert cop_num in (1, 3), 'only support grayscale and RGB images'
    # components_name: ['Y', 'Cr', 'Cb']
    cop_infos = {}
    for i in range(cop_num):
        cop_id, ratio, qt_id = read(3)  # cop_id: 1, 2, 3; qt_id: 0, 1
        horizontal_ratio, vertical_ratio = ratio >> 4, ratio & 15
        cop_infos[cop_id] = ComponentInfo(
            id_=cop_id,
            horizontal=horizontal_ratio,
            vertical=vertical_ratio,
            qt_id=qt_id,
            dc_ht_id=None,
            ac_ht_id=None,
        )

    marker = read(2)
    hts = {}  # (type, id): table
    # type 0: DC, 1: AC
    # Huffman table id: 0, 1
    while marker == MARKER.DHT:  # Define Huffman Table
        length, byte = bytes2int(read(2)), read(1)[0]
        type_, id_ = byte >> 4, byte & 0b1111
        count = np.frombuffer(read(16), dtype=np.uint8, count=16)
        total = count.sum()
        weigh = np.frombuffer(read(total), dtype=np.uint8, count=total)
        assert length == 19 + total
        hts[type_, id_] = canonical_huffman_table(count, weigh)
        marker = read(2)

    DRI = None
    if marker == MARKER.DRI:  # Define Restart Interval
        DRI = dict(marker=marker, length=bytes2int(read(2)))
        DRI['content'] = read(length - 2)
        marker = read(2)
    assert DRI is None, 'not support DRI yet'

    assert marker == MARKER.SOS
    length = bytes2int(read(2))
    assert read(1)[0] == cop_num
    for i in range(cop_num):
        cop_id, ht_id = read(2)
        # ht_id: (dc_ht_id, ac_ht_id)
        cop_infos[cop_id].dc_ht_id = ht_id >> 4
        cop_infos[cop_id].ac_ht_id = ht_id & 0b1111
    # cop_id: 1 Y, 2 Cb, 3 Cr
    cop_infos = [cop_infos[i + 1] for i in range(cop_num)]
    assert read(3) == b'\x00\x3f\x00'
    return qts, hts, cop_infos, height, width


def decode_mcu(stream, mcu_num, block_num, hts):
    mcu = np.zeros([mcu_num, block_num, 64], dtype=np.int32)
    for cur in mcu:
        for dct, (dc_ht, ac_ht) in zip(cur, hts):
            # decode DC coefficients
            dc_size = read_huffman_code(dc_ht, stream)
            dc_complement = stream.read_int(dc_size)
            dct[0] = decode_2s_complement(dc_complement, dc_size)

            ac_index = 1
            rss_ = []
            while ac_index < 64:
                # obtain (run, size) pair
                rs = read_huffman_code(ac_ht, stream)
                run_length, ac_size = rs >> 4, rs & 15
                if (run_length, ac_size) == (0, 0):
                    dct[ac_index:] = 0
                    break
                dct[ac_index:ac_index + run_length] = 0
                ac_index += run_length
                ac_complement = stream.read_int(ac_size)
                dct[ac_index] = decode_2s_complement(ac_complement, ac_size)
                rss_.append((run_length, ac_size,
                             decode_2s_complement(ac_complement, ac_size)))
                ac_index += 1
    # Due to byte alignment, some bits (< 8) may be not used.
    return mcu


def iDPCM(mcu, cop_infos):
    index, mcu_num = 0, mcu.shape[0]
    for cop in cop_infos:
        num = cop.horizontal * cop.vertical
        dc = mcu[:, index:index + num, 0].flat
        for i in range(1, num * mcu_num):
            dc[i] += dc[i - 1]
        index += num
    return mcu


def de_quantization(mcu, cop_infos, qts):
    index = 0
    for cop in cop_infos:
        num = cop.horizontal * cop.vertical
        mcu[:, index:index + num] *= qts[cop.qt_id][None, None]
        index += num
    return mcu


def zigzag_decode(zz):
    dct = np.zeros((zz.shape[0], zz.shape[1], BH, BW), dtype=np.float32)
    trace = zigzag_points(BH, BW)
    for i, p in enumerate(trace):
        dct[:, :, p[0], p[1]] = zz[:, :, i]
    return dct


def iDCT(dct):
    block = np.zeros_like(dct)
    # the input of DCT/iDCT must be float number
    for i in range(dct.shape[0]):
        for j in range(dct.shape[1]):
            block[i, j] = cv2.idct(dct[i, j])
    return block


def decode_jpeg(reader):
    qts, hts, cop_infos, height, width = decode_header(reader)
    remains = reader.read()
    data, marker = remains[:-2].replace(b'\xff\x00', b'\xff'), remains[-2:]
    bit_stream = BitStreamReader(data)
    assert marker == MARKER.EOI

    # for blocks in MCU, choose the corresponding huffman tables to decode
    mcu_hts = []
    for cop in cop_infos:
        for _ in range(cop.horizontal * cop.vertical):
            mcu_hts.append((hts[0, cop.dc_ht_id], hts[1, cop.ac_ht_id]))

    # the height and width of MCU
    mh, mw = 8 * cop_infos[0].vertical, 8 * cop_infos[0].horizontal
    mcu_num = ceil(height / mh) * ceil(width / mw)
    block_num = len(mcu_hts)  # the number of blocks in each MCU
    mcu = decode_mcu(bit_stream, mcu_num, block_num, mcu_hts)

    # reverse the differential encoding (DPCM) of DC coefficients
    mcu = iDPCM(mcu, cop_infos)
    # inverse quantization
    mcu = de_quantization(mcu, cop_infos, qts)

    # zigzag to block, the returned result is float32
    mcu = zigzag_decode(mcu)

    # iDCT
    block = iDCT(mcu)

    # arrange blocks to form images
    ims, index = [], 0
    for cop in cop_infos:
        hr, vr = cop.horizontal, cop.vertical
        num = hr * vr
        # the number of MCUs in each column/row
        nh, nw = ceil(height / mh), ceil(width / mw)
        if num == 1:  # Cb, Cr, or Y when 4:4:4
            im = restore_image(block[:, index], nh, nw)
        else:  # Y when 4:2:2 or 4:2:0
            mat = block[:, index:index + num].reshape(
                -1, vr, hr, BH, BW).swapaxes(2, 3).reshape(-1, vr*BH, hr*BW)
            im = restore_image(mat, nh, nw)
        index += num
        ims.append(im)
    # reverse DC level shift for luminance
    ims[0] += 128
    if len(ims) == 3:
        # interpolating for chroma
        if ims[0].shape != ims[1].shape:
            h_, w_ = ims[0].shape
            for i in range(1, 3):
                ims[i] = cv2.resize(ims[i], (w_, h_))
        im = np.stack(ims, axis=-1)
        im = YCbCr2RGB(im)
    else:
        im = ims[0]
    im = im[:height, :width].round()
    im[im < 0], im[im > 255] = 0, 255
    return im.astype(np.uint8)


def read_jpeg(filename):
    reader = BytesIO(Path(filename).read_bytes())
    return decode_jpeg(reader)


def main():
    im = read_jpeg('image/color.jpg')
    Image.fromarray(im).save('image/color-decode.bmp')


if __name__ == '__main__':
    main()
