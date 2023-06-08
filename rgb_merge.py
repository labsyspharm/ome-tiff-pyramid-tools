import argparse
import sys
import tifffile
import tqdm
import uuid
import zarr


def error(path, msg):
    print(f"\nERROR: {path}: {msg}")
    sys.exit(1)


def tiles(zimg):
    th, tw = zimg.chunks[1:]
    ch, cw = zimg.cdata_shape[1:]
    for j in range(ch):
        for i in range(cw):
            tile = zimg[:, th * j : th * (j + 1), tw * i : tw * (i + 1)]
            tile = tile.transpose(1, 2, 0)
            # Must copy() to provide contiguous array for jpeg encoder.
            yield tile.copy()


def progress(zimg, level):
    ch, cw = zimg.cdata_shape[1:]
    total = ch * cw
    t = tqdm.tqdm(tiles(zimg), desc=f"  Level {level}", total=total)
    # Fix issue with tifffile's peek_iterator causing a missed update.
    t.update()
    return iter(t)


def main():
    parser = argparse.ArgumentParser(
        description="Convert OME-TIFF with separate R/G/B channels to true RGB",
        epilog="Note that any existing pyramid levels are retained, but a"
        " pyramid will not be added if not already present. JPEG compression"
        " will be used.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", help="Path to input image", metavar="input.ome.tif"
    )
    parser.add_argument(
        "output", help="Path to output image", metavar="output.ome.tif"
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

    software = tiff.pages[0].software
    with tifffile.TiffWriter(args.output, ome=True, bigtiff=True) as writer:
        pyramid = zarr.open(series.aszarr())
        if isinstance(pyramid, zarr.Array):
            pyramid = {0: pyramid}
        writer.write(
            data=progress(pyramid[0], 1),
            shape=pyramid[0].shape[1:] + (3,),
            subifds=len(pyramid) - 1,
            dtype="uint8",
            tile=pyramid[0].chunks[1:],
            compression="jpeg",
            software=software,
            metadata={
                "UUID": uuid.uuid4().urn,
                "Creator": software,
                "Name": series.name,
            },
        )
        for level in range(1, len(pyramid)):
            zimg = pyramid[level]
            writer.write(
                data=progress(zimg, level + 1),
                shape=zimg.shape[1:] + (3,),
                subfiletype=1,
                dtype="uint8",
                tile=zimg.chunks[1:],
                compression="jpeg",
            )
        print()


if __name__ == '__main__':
    main()
