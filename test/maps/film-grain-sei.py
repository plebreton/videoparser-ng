#!/usr/bin/env python3
"""
Insert a film grain characteristics SEI (H.274, model 0, luma only) before the
first slice of every picture of an H.264 or HEVC Annex B stream, so that the
decoder applies film grain to the output frames.

Usage: film-grain-sei.py h264|hevc <in.annexb> <out.annexb>
"""
import re
import sys


class BitWriter:
    def __init__(self):
        self.bits = []

    def u(self, n, v):
        self.bits += [(v >> (n - 1 - i)) & 1 for i in range(n)]

    def ue(self, v):
        v += 1
        n = v.bit_length()
        self.u(n - 1, 0)
        self.u(n, v)

    def se(self, v):
        self.ue(2 * v - 1 if v > 0 else -2 * v)


def film_grain_payload(codec):
    w = BitWriter()
    w.u(1, 0)  # film_grain_characteristics_cancel_flag
    w.u(2, 0)  # film_grain_model_id: frequency filtering
    w.u(1, 0)  # separate_colour_description_present_flag
    w.u(2, 0)  # blending_mode_id: additive
    w.u(4, 4)  # log2_scale_factor
    w.u(1, 1); w.u(1, 0); w.u(1, 0)  # comp_model_present_flag (luma only)
    w.u(8, 0)  # num_intensity_intervals_minus1
    w.u(3, 2)  # num_model_values_minus1
    w.u(8, 0); w.u(8, 255)  # intensity interval bounds
    w.se(100); w.se(8); w.se(8)  # strength, horizontal and vertical cut-off
    if codec == "h264":
        w.ue(1)  # film_grain_characteristics_repetition_period
    else:
        w.u(1, 1)  # film_grain_characteristics_persistence_flag
    w.u(1, 1)  # payload_bit_equal_to_one (byte alignment of the payload)
    while len(w.bits) % 8:
        w.u(1, 0)
    return bytes(int("".join(map(str, w.bits[i:i + 8])), 2) for i in range(0, len(w.bits), 8))


def emulation_prevention(data):
    out, zeros = bytearray(), 0
    for b in data:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def sei_nal(codec, slice_header):
    payload = film_grain_payload(codec)
    sei = bytes([19, len(payload)]) + payload + b"\x80"  # type 19, size, rbsp trailing bits
    if codec == "h264":
        header = bytes([0x06])
    else:
        header = bytes([39 << 1, slice_header[1]])  # prefix SEI, same temporal id
    return b"\x00\x00\x00\x01" + header + emulation_prevention(sei)


def first_slice_of_picture(codec, nal):
    if codec == "h264":
        return (nal[0] & 0x1F) in (1, 5) and nal[1] & 0x80  # first_mb_in_slice == 0
    return ((nal[0] >> 1) & 0x3F) < 32 and nal[2] & 0x80  # first_slice_segment_in_pic_flag


def main():
    codec, src, dst = sys.argv[1:4]
    data = open(src, "rb").read()
    starts = [m.start() for m in re.finditer(b"\x00\x00\x01", data)]
    out, pictures = bytearray(), 0
    for i, s in enumerate(starts):
        begin = s - 1 if s > 0 and data[s - 1] == 0 else s
        end = starts[i + 1] if i + 1 < len(starts) else len(data)
        if i + 1 < len(starts) and data[end - 1] == 0:
            end -= 1
        nal = data[s + 3:end]
        if first_slice_of_picture(codec, nal):
            out += sei_nal(codec, nal)
            pictures += 1
        out += data[begin:end]
    open(dst, "wb").write(out)
    print(f"{dst}: film grain SEI before {pictures} pictures")


if __name__ == "__main__":
    main()
