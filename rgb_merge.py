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
        tiff_out = tifffile.TiffFile(args.output, is_ome=False)
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

    metadata.uuid = uuid.uuid4().urn
    # Reconfigure metadata for a single 3-sample channel.
    mpixels = metadata.images[0].pixels
    del mpixels.channels[1:]
    del mpixels.planes[1:]
    mpixels.channels[0].name = None
    mpixels.channels[0].samples_per_pixel = 3
    mpixels.tiff_data_blocks = [ome_types.model.TiffData(plane_count=1)]
    # Drop the optional PyramidResolution annotation rather than recompute it.
    metadata.structured_annotations = [
        a for a in metadata.structured_annotations
        if a.namespace != "openmicroscopy.org/PyramidResolution"
    ]
    ome_xml = metadata.to_xml()
    # Hack to work around ome_types always writing the default color.
    ome_xml = ome_xml.replace('Color="-1"', "")

    software = tiff.pages[0].software
    print("Writing new OME-TIFF:")
    with tifffile.TiffWriter(args.output, ome=False, bigtiff=True) as writer:
        writer.write(
            data=progress(0),
            shape=shapes[0] + (3,),
            subifds=num_levels - 1,
            dtype="uint8",
            tile=(tile_size, tile_size),
            compression="jpeg",
            compressionargs={"level": 90},
            software=software,
            description=ome_xml.encode(),
            metadata=None,
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
