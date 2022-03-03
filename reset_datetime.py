import argparse
from datetime import datetime
import sys
import tiffsurgeon


def parse_args():
    parser = argparse.ArgumentParser(
        description="Set all DateTime tags in a TIFF to one value (in place).",
    )
    parser.add_argument(
        "image_path", help="TIFF file to process"
    )
    parser.add_argument(
        "datetime",
        help="YYYY:MM:DD HH:MM:SS string to replace all DateTime values."
        "Note the use of colons in the date, as required by the TIFF standard.",
    )
    argv = sys.argv[1:]
    # Allow date and time to be passed as separate args for convenience.
    if len(argv) == 3:
        argv[1] = " ".join(argv[1:3])
        del argv[2]
    args = parser.parse_args(argv)
    return args


def main():

    args = parse_args()

    try:
        tiff = tiffsurgeon.TiffSurgeon(
            args.image_path, encoding="utf-8", writeable=True
        )
    except tiffsurgeon.FormatError as e:
        print(f"TIFF format error: {e}")
        sys.exit(1)
    try:
        datetime.strptime(args.datetime, "%Y:%m:%d %H:%M:%S")
    except ValueError as e:
        print(f"Invalid datetime: {e}")
        sys.exit(1)
    new_datetime = args.datetime.encode("ascii")
    assert len(new_datetime) == 19, \
        "length of DateTime string must be exactly 19"

    tiff.read_ifds()

    subifds = [
        tiff.read_ifd(v)
        for i in tiff.ifds if 330 in i.tags
        for v in i.tags[330].value
    ]
    offsets = [
        i.tags[306].offset_range.start
        for i in tiff.ifds + subifds if 306 in i.tags
    ]
    if offsets:
        for x in offsets:
            tiff.file.seek(x)
            tiff.file.write(new_datetime)
        print(
            f"Successfully replaced {len(offsets)} DateTime values in"
            f" {args.image_path}"
        )
    else:
        print(
            f"No DateTime tags found in {args.image_path} -- file not modified"
        )

    tiff.close()



if __name__ == "__main__":
    main()
