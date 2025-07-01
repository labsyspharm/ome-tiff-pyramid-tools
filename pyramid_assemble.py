from __future__ import print_function, division
import warnings
import sys
import os
import re
import io
import argparse
import pathlib
import struct
import itertools
import uuid
import multiprocessing
import concurrent.futures
import numpy as np
import tifffile
import zarr
import skimage.transform
# This API is apparently changing in skimage 1.0 but it's not clear to
# me what the replacement will be, if any. We'll explicitly import
# this so it will break loudly if someone tries this with skimage 1.0.
try:
    from skimage.util.dtype import _convert as dtype_convert
except ImportError:
    from skimage.util.dtype import convert as dtype_convert


def format_shape(shape):
    return "%d x %d" % (shape[1], shape[0])


def error(path, msg):
    print(f"\nERROR: {path}: {msg}")
    sys.exit(1)


class SampleSplitter:

    def __init__(self, zimg, channel):
        self.zimg = zimg
        self.channel = channel

    def __getitem__(self, key):
        return self.zimg[key + (self.channel,)]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_paths", metavar="input.tif", type=pathlib.Path, nargs="+",
        help="List of TIFF files to combine. All images must have the same"
        " dimensions and pixel type. All pages of multi-page images will be"
        " included by default; the suffix ,p may be appended to the filename to"
        " specify a single page p.",
    )
    parser.add_argument(
        "out_path", metavar="output.ome.tif", type=pathlib.Path,
        help="Output filename. Script will exit immediately if file exists.",
    )
    parser.add_argument(
        "--pixel-size", metavar="MICRONS", type=float, default=None,
        help="Pixel size in microns. Will be recorded in OME-XML metadata.",
    )
    parser.add_argument(
        "--channel-names", metavar="CHANNEL", nargs="+",
        help="Channel names. Will be recorded in OME-XML metadata. Number of"
        " names must match number of channels in final output file."
    )
    parser.add_argument(
        "--tile-size", metavar="PIXELS", type=int, default=1024,
        help="Width of pyramid tiles in output file (must be a multiple of 16);"
        " default is 1024",
    )
    parser.add_argument(
        "--split-rgb", action="store_true", default=False,
        help="Split RGB images into three discrete channels in output file"
        " (useful to allow merging RGB and non-RGB images)"
    )
    parser.add_argument(
        "--mask", action="store_true", default=False,
        help="Adjust processing for label mask or binary mask images (currently"
        " just switch to nearest-neighbor downsampling)",
    )
    parser.add_argument(
        "--num-threads", metavar="N", type=int, default=0,
        help="Number of parallel threads to use for image downsampling; default"
        " is number of available CPUs"
    )
    args = parser.parse_args()
    in_paths = args.in_paths
    out_path = args.out_path
    is_mask = args.mask
    if out_path.exists():
        error(out_path, "Output file already exists, remove before continuing.")

    if args.num_threads == 0:
        if hasattr(os, 'sched_getaffinity'):
            args.num_threads = len(os.sched_getaffinity(0))
        else:
            args.num_threads = multiprocessing.cpu_count()
        print(
            f"Using {args.num_threads} worker threads based on detected CPU"
            " count."
        )
        print()
    tifffile.TIFF.MAXWORKERS = args.num_threads
    tifffile.TIFF.MAXIOWORKERS = args.num_threads * 5

    in_imgs = []
    num_channels = 0
    print("Scanning input images")
    for i, path in enumerate(in_paths, 1):
        spath = str(path)
        if match := re.search(r",(\d+)$", spath):
            c = int(match.group(1))
            path = pathlib.Path(spath[:match.start()])
        else:
            c = None
        tiff = tifffile.TiffFile(path)
        series = tiff.series[0]
        shape = (series.sizes["height"], series.sizes["width"])
        dtype = series.dtype
        is_rgb = False
        if series.axes == "YX":
            channels = 1
        elif series.axes == "YXS":
            if series.sizes["sample"] != 3:
                error(path, "sample count not supported: {series.sizes['sample']}")
            channels = 3 if args.split_rgb else 1
            is_rgb = True
        elif series.axes == "CYX":
            channels = series.sizes["channel"]
        elif series.axes == "QYX":
            channels = series.sizes["other"]
        else:
            error(
                path, f"image axes combination not supported: {series.axes}",
            )
        pages = series.levels[0]
        if c is not None:
            pages = [pages[c]]
        imgs = [zarr.open(p.aszarr()) for p in pages]
        if is_rgb and args.split_rgb:
            assert len(imgs) == 1
            imgs = [SampleSplitter(imgs[0], i) for i in range(3)]
        if i == 1:
            base_shape = shape
            base_rgb = is_rgb and not args.split_rgb
            base_dtype = dtype
            if dtype == np.uint32 or dtype == np.int32:
                if not is_mask:
                   error(
                       path,
                       "32-bit images are only supported in --mask mode."
                       " Please contact the authors if you need support for"
                       " intensity-based 32-bit images."
                    )
            elif dtype == np.uint16 or dtype == np.uint8:
                pass
            else:
                error(
                    path,
                    f"Can't handle dtype '{dtype}' yet, please contact the"
                    f" authors."
                )
        else:
            if shape != base_shape:
                error(
                    path,
                    f"Expected shape {base_shape} to match first input image,"
                    f" got {shape} instead."
                )
            if is_rgb != base_rgb and not args.split_rgb:
                error(
                    path,
                    f"Can't mix RGB and non-RGB images."
                )
            if dtype != base_dtype:
                error(
                    path,
                    f"Expected dtype '{base_dtype}' to match first input image,"
                    f" got '{dtype}' instead."
                )
        print(f"    file {i}")
        print(f"        path: {path}")
        f_channels = 'RGB' if is_rgb else channels
        if is_rgb and args.split_rgb:
            f_channels = '3 (RGB-split)'
        print(f"        properties: shape={shape} dtype={dtype}, channels={f_channels}")
        if c is not None:
            print(f"        using single channel: {c}")
        in_imgs.extend(imgs)
        num_channels += channels
    print()

    num_levels = max(np.ceil(np.log2(max(base_shape) / args.tile_size)) + 1, 1)
    factors = 2 ** np.arange(num_levels)
    shapes = np.ceil(np.array(base_shape) / factors[:,None]).astype(int)
    cshapes = np.ceil(shapes / args.tile_size).astype(int)

    if args.channel_names and len(args.channel_names) != num_channels:
        error(
            out_path,
            f"Number of channel names ({len(args.channel_names)}) does not"
            f" match number of channels in final image ({num_channels})."
        )

    print("Pyramid level sizes:")
    for i, shape in enumerate(shapes):
        print(f"    level {i + 1}: {format_shape(shape)}", end="")
        if i == 0:
            print(" (original size)", end="")
        print()
    print()

    pool = concurrent.futures.ThreadPoolExecutor(args.num_threads)

    def tiles0():
        ts = args.tile_size
        ch, cw = cshapes[0]
        for c, zimg in enumerate(in_imgs, 1):
            print(f"    channel {c}")
            for j in range(ch):
                for i in range(cw):
                    tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                    yield tile

    def tiles(level):
        tiff_out = tifffile.TiffFile(args.out_path, is_ome=False)
        series = tiff_out.series[0]
        zimg = zarr.open(series.aszarr(level=level - 1))
        ts = args.tile_size * 2

        def tile(coords):
            c, j, i = coords
            if series.axes in ("YX", "YXS"):
                assert c == 0
                tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            else:
                tile = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            if is_mask:
                tile = tile[::2, ::2]
            else:
                factors = (2, 2)
                if base_rgb:
                    factors += (1,)
                tile = skimage.transform.downscale_local_mean(tile, factors)
                tile = np.round(tile).astype(base_dtype)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(num_channels), range(ch), range(cw))
        yield from map(tile, coords)
        # yield from pool.map(tile, coords)

    metadata = {
        "UUID": uuid.uuid4().urn,
    }
    if args.pixel_size:
        metadata.update({
            "PhysicalSizeX": args.pixel_size,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": args.pixel_size,
            "PhysicalSizeYUnit": "µm",
        })
    if args.channel_names:
        metadata.update({
            "Channel": {"Name": args.channel_names},
        })
    photometric = "rgb" if base_rgb else "minisblack"
    print(f"Writing level 1: {format_shape(shapes[0])}")
    with tifffile.TiffWriter(args.out_path, ome=True, bigtiff=True) as writer:
        wshape = (num_channels,) + tuple(shapes[0])
        if base_rgb:
            wshape += (3,)
        writer.write(
            data=tiles0(),
            shape=wshape,
            subifds=num_levels - 1,
            dtype=base_dtype,
            photometric=photometric,
            tile=(args.tile_size, args.tile_size),
            compression="adobe_deflate",
            predictor=True,
            metadata=metadata,
        )
        print()
        for level, shape in enumerate(shapes[1:], 1):
            print(
                f"Resizing image for level {level + 1}: {format_shape(shape)}"
            )
            wshape = (num_channels,) + tuple(shape)
            if base_rgb:
                wshape += (3,)
            writer.write(
                data=tiles(level),
                shape=wshape,
                subfiletype=1,
                dtype=base_dtype,
                photometric=photometric,
                tile=(args.tile_size, args.tile_size),
                compression="adobe_deflate",
                predictor=True,
            )
        print()


if __name__ == '__main__':
    main()
