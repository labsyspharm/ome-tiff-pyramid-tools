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
    print("Scanning input images")
    for i, path in enumerate(in_paths, 1):
        spath = str(path)
        if match := re.search(r",(\d+)$", spath):
            c = int(match.group(1))
            path = pathlib.Path(spath[:match.start()])
        else:
            c = None
        img_in = zarr.open(tifffile.imread(path, key=c, level=0, aszarr=True))
        if img_in.ndim == 2:
            shape = img_in.shape
            imgs = [img_in]
        elif img_in.ndim == 3:
            shape = img_in.shape[1:]
            imgs = [
                zarr.open(tifffile.imread(path, key=i, level=0, aszarr=True))
                for i in range(img_in.shape[0])
            ]
        else:
            error(
                path, f"{img_in.ndim}-dimensional images are not supported",
            )
        if i == 1:
            base_shape = shape
            dtype = img_in.dtype
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
            if img_in.dtype != dtype:
                error(
                    path,
                    f"Expected dtype '{dtype}' to match first input image,"
                    f" got '{img_in.dtype}' instead."
                )
        print(f"    file {i}")
        print(f"        path: {spath}")
        print(f"        properties: shape={img_in.shape} dtype={img_in.dtype}")
        in_imgs.extend(imgs)
    print()

    num_channels = len(in_imgs)
    num_levels = np.ceil(np.log2(max(base_shape) / args.tile_size)) + 1
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
            img = zimg[:]
            for j in range(ch):
                for i in range(cw):
                    tile = img[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                    yield tile
            del img

    def tiles(level):
        tiff_out = tifffile.TiffFile(args.out_path, is_ome=False)
        zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
        ts = args.tile_size * 2

        def tile(coords):
            c, j, i = coords
            if zimg.ndim == 2:
                assert c == 0
                tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            else:
                tile = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            if is_mask:
                tile = tile[::2, ::2]
            else:
                tile = skimage.transform.downscale_local_mean(tile, (2, 2))
                tile = np.round(tile).astype(dtype)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(num_channels), range(ch), range(cw))
        yield from pool.map(tile, coords)

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
    print(f"Writing level 1: {format_shape(shapes[0])}")
    with tifffile.TiffWriter(args.out_path, ome=True, bigtiff=True) as writer:
        writer.write(
            data=tiles0(),
            shape=(num_channels,) + tuple(shapes[0]),
            subifds=num_levels - 1,
            dtype=dtype,
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
            writer.write(
                data=tiles(level),
                shape=(num_channels,) + tuple(shape),
                subfiletype=1,
                dtype=dtype,
                tile=(args.tile_size, args.tile_size),
                compression="adobe_deflate",
                predictor=True,
            )
        print()


if __name__ == '__main__':
    main()
