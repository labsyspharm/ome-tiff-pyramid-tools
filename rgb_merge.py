import argparse
import itertools
import concurrent.futures
import multiprocessing
import numpy as np
import ome_types
import os
import pathlib
import skimage.transform
import sys
import tifffile
import tqdm
import uuid
import zarr


def error(path, msg):
    print(f"\nERROR: {path}: {msg}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert OME-TIFF with separate R/G/B channels to true RGB",
    )
    parser.add_argument(
        "input", help="Path to input image", metavar="input.ome.tif",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output", help="Path to output image", metavar="output.ome.tif",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--num-threads", metavar="N", type=int, default=0,
        help="Number of parallel threads to use for image downsampling; default"
        " is number of available CPUs"
    )

    args = parser.parse_args()

    tiff = tifffile.TiffFile(args.input)
    if len(tiff.series) != 1:
        error(
            args.input,
            f"Input must contain only one OME image; found {len(tiff.series)}"
            " instead"
        )
    series = tiff.series[0]
    if series.axes != "CYX":
        error(
            args.input,
            "Input must have shape (channel, height, width); found"
            f" {series.dims} = {series.shape} instead"
        )
    if series.shape[0] != 3:
        error(
            args.input,
            f"Input must have exactly 3 channels; found {series.shape[0]}"
            " instead"
        )
    if series.dtype != "uint8":
        error(
            args.input,
            f"Input must have pixel type uint8; found {series.dtype}"
        )

    if args.output.exists():
        error(args.output, "Output file exists, remove before continuing")

    if args.num_threads == 0:
        if hasattr(os, 'sched_getaffinity'):
            args.num_threads = len(os.sched_getaffinity(0))
        else:
            args.num_threads = multiprocessing.cpu_count()
        print(
            f"Using {args.num_threads} worker threads based on available CPUs"
        )
        print()

    image0 = zarr.open(series.aszarr(level=0))
    metadata = ome_types.from_xml(tiff.ome_metadata, parser="xmlschema")

    base_shape = image0.shape[1:]
    tile_size = 1024
    num_levels = np.ceil(np.log2(max(base_shape) / tile_size)) + 1
    factors = 2 ** np.arange(num_levels)
    shapes = [
        tuple(s) for s in
        (np.ceil(np.array(base_shape) / factors[:, None])).astype(int)
    ]
    cshapes = [
        tuple(s) for s in
        np.ceil(np.divide(shapes, tile_size)).astype(int)
    ]
    print("Pyramid level sizes:")
    for i, shape in enumerate(shapes):
        shape_fmt = "%d x %d" % (shape[1], shape[0])
        print(f"    Level {i + 1}: {shape_fmt}", end="")
        if i == 0:
            print(" (original size)", end="")
        print()
    print()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads)

    def tiles0():
        zimg = image0
        ts = tile_size
        ch, cw = cshapes[0]
        for j in range(ch):
            for i in range(cw):
                tile = zimg[:, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                tile = tile.transpose(1, 2, 0)
                # Must copy() to provide contiguous array for jpeg encoder.
                yield tile.copy()

    def tiles(level):
        if level == 0:
           yield from tiles0()
        tiff_out = tifffile.TiffFile(args.output)
        zimg = zarr.open(tiff_out.series[0].aszarr(level=level - 1))
        ts = tile_size * 2

        def tile(coords):
            j, i = coords
            tile = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
            tile = skimage.transform.downscale_local_mean(tile, (2, 2, 1))
            tile = np.round(tile).astype(np.uint8)
            return tile

        ch, cw = cshapes[level]
        coords = itertools.product(range(ch), range(cw))
        yield from pool.map(tile, coords)

    def progress(level):
        ch, cw = cshapes[level]
        t = tqdm.tqdm(
            tiles(level),
            desc=f"    Level {level + 1}",
            total=ch * cw,
            unit="tile",
        )
        # Fix issue with tifffile's peek_iterator causing a missed update.
        t.update()
        return iter(t)

    software = tiff.pages[0].software
    px = metadata.images[0].pixels.physical_size_x_quantity.m_as("micron")
    py = metadata.images[0].pixels.physical_size_y_quantity.m_as("micron")
    print("Writing new OME-TIFF:")
    with tifffile.TiffWriter(args.output, ome=True, bigtiff=True) as writer:
        writer.write(
            data=progress(0),
            shape=shapes[0] + (3,),
            subifds=num_levels - 1,
            dtype="uint8",
            tile=(tile_size, tile_size),
            compression="jpeg",
            compressionargs={"level": 90},
            software=software,
            metadata={
                "UUID": uuid.uuid4().urn,
                "Creator": software,
                "Name": series.name,
                "Pixels": {
                    "PhysicalSizeX": px, "PhysicalSizeXUnit": "\u00b5m",
                    "PhysicalSizeY": py, "PhysicalSizeYUnit": "\u00b5m"
                },
            },
        )
        for level, shape in enumerate(shapes[1:], 1):
            writer.write(
                data=progress(level),
                shape=shape + (3,),
                subfiletype=1,
                dtype="uint8",
                tile=(tile_size, tile_size),
                compression="jpeg",
                compressionargs={"level": 90},
            )
        print()


if __name__ == '__main__':
    main()
