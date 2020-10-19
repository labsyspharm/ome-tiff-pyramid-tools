import sys
import os
import argparse
import struct
import re
import fractions
import io
import xml.etree.ElementTree
import collections
import reprlib
import dataclasses
from typing import List, Any


datatype_formats = {
    1: "B", # BYTE
    2: "s", # ASCII
    3: "H", # SHORT
    4: "I", # LONG
    5: "I", # RATIONAL (pairs)
    6: "b", # SBYTE
    7: "B", # UNDEFINED
    8: "h", # SSHORT
    9: "i", # SLONG
    10: "i", # SRATIONAL (pairs)
    11: "f", # FLOAT
    12: "d", # DOUBLE
    13: "I", # IFD
    16: "Q", # LONG8
    17: "q", # SLONG8
    18: "Q", # IFD8
}
rational_datatypes = {5, 10}


class TiffSurgeon:
    """Read, manipulate and write IFDs in BigTIFF files."""

    def __init__(self, path, *, writeable=False, encoding=None):
        self.path = path
        self.writeable = writeable
        self.encoding = encoding
        self.endian = ""
        self.ifds = None
        self.file = open(self.path, "r+b" if self.writeable else "rb")
        self._validate()

    def _validate(self):
        signature = self.read("2s")
        signature = signature.decode("ascii", errors="ignore")
        if signature == "II":
            self.endian = "<"
        elif signature == "MM":
            self.endian = ">"
        else:
            raise FormatError(f"Not a TIFF file (signature is '{signature}').")
        version = self.read("H")
        if version == 42:
            raise FormatError("Cannot process classic TIFF, only BigTIFF.")
        offset_size, reserved, first_ifd_offset = self.read("H H Q")
        if version != 43 or offset_size != 8 or reserved != 0:
            raise FormatError("Malformed TIFF, giving up!")
        self.first_ifd_offset = first_ifd_offset

    def read(self, fmt, *, file=None):
        if file is None:
            file = self.file
        endian = self.endian or "="
        size = struct.calcsize(endian + fmt)
        raw = file.read(size)
        value = self.unpack(fmt, raw)
        return value

    def write(self, fmt, *values):
        if not self.writeable:
            raise ValueError("File is opened as read-only.")
        raw = self.pack(fmt, *values)
        self.file.write(raw)

    def unpack(self, fmt, raw):
        assert self.endian or re.match(r"\d+s", fmt), \
            "can't unpack non-string before endianness is detected"
        fmt = self.endian + fmt
        size = struct.calcsize(fmt)
        values = struct.unpack(fmt, raw[:size])
        if len(values) == 1:
            return values[0]
        else:
            return values

    def pack(self, fmt, *values):
        assert self.endian, "can't pack without endian set"
        fmt = self.endian + fmt
        raw = struct.pack(fmt, *values)
        return raw

    def read_ifds(self):
        ifds = [self.read_ifd(self.first_ifd_offset)]
        while ifds[-1].offset_next:
            ifds.append(self.read_ifd(ifds[-1].offset_next))
        self.ifds = ifds

    def read_ifd(self, offset):
        self.file.seek(offset)
        num_tags = self.read("Q")
        buf = io.BytesIO(self.file.read(num_tags * 20))
        offset_next = self.read("Q")
        try:
            tags = TagSet([self.read_tag(buf) for i in range(num_tags)])
        except FormatError as e:
            raise FormatError(f"IFD at offset {offset}, {e}") from None
        ifd = Ifd(tags, offset, offset_next)
        return ifd

    def read_tag(self, buf):
        tag = Tag(*self.read("H H Q 8s", file=buf))
        value, offset_range = self.tag_value(tag)
        tag = dataclasses.replace(tag, value=value, offset_range=offset_range)
        return tag

    def append_ifd_sequence(self, ifds):
        """Write list of IFDs as a chained sequence at the end of the file.

        Returns a list of new Ifd objects with updated offsets.

        """
        self.file.seek(0, os.SEEK_END)
        new_ifds = []
        for ifd in ifds:
            offset = self.file.tell()
            self.write("Q", len(ifd.tags))
            for tag in ifd.tags:
                self.write_tag(tag)
            offset_next = self.file.tell() + 8 if ifd is not ifds[-1] else 0
            self.write("Q", offset_next)
            new_ifd = dataclasses.replace(
                ifd, offset=offset, offset_next=offset_next
            )
            new_ifds.append(new_ifd)
        return new_ifds

    def append_tag_data(self, code, datatype, value):
        """Build new tag and write data to the end of the file if necessary.

        Returns a Tag object corresponding to the passed parameters. This
        function only writes any "overflow" data and not the IFD entry itself,
        so the returned Tag must still be written to an IFD.

        If the value is small enough to fit in the data field within an IFD, no
        data will actually be written to the file and the returned Tag object
        will have the value encoded in its data attribute. Otherwise the data
        will be appended to the file and the returned Tag's data attribute will
        encode the corresponding offset.

        """
        fmt = datatype_formats[datatype]
        # FIXME Should we perform our own check that values match datatype?
        # struct.pack will do it but the exception won't be as understandable.
        original_value = value
        if isinstance(value, str):
            if not self.encoding:
                raise ValueError(
                    "ASCII tag values must be bytes if encoding is not set"
                )
            value = [value.encode(self.encoding) + b"\x00"]
            count = len(value[0])
        elif isinstance(value, bytes):
            value = [value + b"\x00"]
            count = len(value[0])
        else:
            try:
                len(value)
            except TypeError:
                value = [value]
            count = len(value)
        struct_count = count
        if datatype in rational_datatypes:
            value = [i for v in value for i in v.as_integer_ratio()]
            count //= 2
        byte_count = struct_count * struct.calcsize(fmt)
        if byte_count <= 8:
            data = self.pack(str(struct_count) + fmt, *value)
            data += bytes(8 - byte_count)
        else:
            self.file.seek(0, os.SEEK_END)
            data = self.pack("Q", self.file.tell())
            self.write(str(count) + fmt, *value)
        # TODO Compute and set offset_range.
        tag = Tag(code, datatype, count, data, original_value)
        return tag

    def write_first_ifd_offset(self, offset):
        self.file.seek(8)
        self.write("Q", offset)

    def write_tag(self, tag):
        self.write("H H Q 8s", tag.code, tag.datatype, tag.count, tag.data)

    def tag_value(self, tag):
        """Return decoded tag data and the file offset range."""
        fmt = datatype_formats[tag.datatype]
        count = tag.count
        if tag.datatype in rational_datatypes:
            count *= 2
        byte_count = count * struct.calcsize(fmt)
        if byte_count <= 8:
            value = self.unpack(str(count) + fmt, tag.data)
            offset_range = range(0, 0)
        else:
            offset = self.unpack("Q", tag.data)
            self.file.seek(offset)
            value = self.read(str(count) + fmt)
            offset_range = range(offset, offset + byte_count)
        if tag.datatype == 2:
            value = value.rstrip(b"\x00")
            if self.encoding:
                try:
                    value = value.decode(self.encoding)
                except UnicodeDecodeError as e:
                    raise FormatError(f"tag {tag.code}: {e}") from None
        elif tag.datatype in rational_datatypes:
            value = [
                fractions.Fraction(*v) for v in zip(value[::2], value[1::2])
            ]
            if len(value) == 1:
                value = value[0]
        return value, offset_range

    def close(self):
        self.file.close()


@dataclasses.dataclass(frozen=True)
class Tag:
    code: int
    datatype: int
    count: int
    data: bytes
    value: Any = None
    offset_range: range = None

    _vrepr = reprlib.Repr()
    _vrepr.maxstring = 60
    _vrepr.maxother = 60
    vrepr = _vrepr.repr

    def __repr__(self):
        return (
            self.__class__.__qualname__ + "("
            + f"code={self.code!r}, datatype={self.datatype!r}, "
            + f"count={self.count!r}, data={self.data!r}, "
            + f"value={self.vrepr(self.value)}"
            + ")"
        )

@dataclasses.dataclass(frozen=True)
class TagSet:
    """Container for Tag objects as stored in a TIFF IFD.

    Tag objects are maintained in a list that's always sorted in ascending order
    by the tag code. Only one tag for a given code may be present, which is where
    the "set" name comes from.

    """

    tags: List[Tag] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.codes) != len(set(self.codes)):
            raise ValueError("Duplicate tag codes are not allowed.")

    def __repr__(self):
        ret = type(self).__name__ + "(["
        if self.tags:
            ret += "\n"
        ret += "".join([f"    {t},\n" for t in self.tags])
        ret += "])"
        return ret

    @property
    def codes(self):
        return [t.code for t in self.tags]

    def __getitem__(self, code):
        for t in self.tags:
            if code == t.code:
                return t
        else:
            raise KeyError(code)

    def __delitem__(self, code):
        try:
            i = self.codes.index(code)
        except ValueError:
            raise KeyError(code) from None
        self.tags[:] = self.tags[:i] + self.tags[i+1:]

    def __contains__(self, code):
        return code in self.codes

    def __len__(self):
        return len(self.tags)

    def __iter__(self):
        return iter(self.tags)

    def get(self, code, default=None):
        try:
            return self[code]
        except KeyError:
            return default

    def get_value(self, code, default=None):
        tag = self.get(code)
        if tag:
            return tag.value
        else:
            return default

    def insert(self, tag):
        """Add a new tag or replace an existing one."""
        for i, t in enumerate(self.tags):
            if tag.code == t.code:
                self.tags[i] = tag
                return
            elif tag.code < t.code:
                break
        else:
            i = len(self.tags)
        n = len(self.tags)
        self.tags[i:n+1] = [tag] + self.tags[i:n]


@dataclasses.dataclass(frozen=True)
class Ifd:
    tags: TagSet
    offset: int
    offset_next: int

    @property
    def nbytes(self):
        return len(self.tags) * 20 + 16

    @property
    def offset_range(self):
        return range(self.offset, self.offset + self.nbytes)


class FormatError(Exception):
    pass


def fix_attrib_namespace(elt):
    """Prefix un-namespaced XML attributes with the tag's namespace."""
    # This fixes ElementTree's inability to round-trip XML with a default
    # namespace ("cannot use non-qualified names with default_namespace option"
    # error). 7-year-old BPO issue here: https://bugs.python.org/issue17088
    # Code inspired by https://gist.github.com/provegard/1381912 .
    if elt.tag[0] == "{":
        uri, _ = elt.tag[1:].rsplit("}", 1)
        new_attrib = {}
        for name, value in elt.attrib.items():
            if name[0] != "{":
                # For un-namespaced attributes, copy namespace from element.
                name = f"{{{uri}}}{name}"
            new_attrib[name] = value
        elt.attrib = new_attrib
    for child in elt:
        fix_attrib_namespace(child)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an OME-TIFF legacy pyramid to the BioFormats 6"
            " OME-TIFF pyramid format in-place.",
    )
    parser.add_argument("image", help="OME-TIFF file to convert")
    parser.add_argument(
        "-n",
        dest="channel_names",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Channel names to be inserted into OME metadata. Number of names"
            " must match number of channels in image. Be sure to put quotes"
            " around names containing spaces or other special shell characters."
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    image_path = sys.argv[1]
    try:
        tiff = TiffSurgeon(image_path, encoding="utf-8", writeable=True)
    except FormatError as e:
        print(f"TIFF format error: {e}")
        sys.exit(1)

    tiff.read_ifds()

    # ElementTree doesn't parse xml declarations so we'll just run some sanity
    # checks that we do have UTF-8 and give it a decoded string instead of raw
    # bytes. We need to both ensure that the raw tag bytes decode properly and
    # that the declaration encoding is UTF-8 if present.
    try:
        omexml = tiff.ifds[0].tags.get_value(270, "")
    except FormatError:
        print("ImageDescription tag is not a valid UTF-8 string (not an OME-TIFF?)")
        sys.exit(1)
    if re.match(r'<\?xml [^>]*encoding="(?!UTF-8)[^"]*"', omexml):
        print("OME-XML is encoded with something other than UTF-8.")
        sys.exit(1)

    xml_ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    if xml_ns["ome"] not in omexml:
        print("Not an OME-TIFF.")
        sys.exit(1)
    if (
        "Faas" not in tiff.ifds[0].tags.get_value(305, "")
        or 330 in tiff.ifds[0].tags
    ):
        print("Not a legacy OME-TIFF pyramid.")
        sys.exit(1)

    # All XML manipulation assumes the document is valid OME-XML!
    root = xml.etree.ElementTree.fromstring(omexml)
    image = root.find("ome:Image", xml_ns)
    pixels = image.find("ome:Pixels", xml_ns)
    size_x = int(pixels.get("SizeX"))
    size_y = int(pixels.get("SizeY"))
    size_c = int(pixels.get("SizeC"))
    size_z = int(pixels.get("SizeZ"))
    size_t = int(pixels.get("SizeT"))
    num_levels = len(root.findall("ome:Image", xml_ns))
    page_dims = [(ifd.tags[256].value, ifd.tags[257].value) for ifd in tiff.ifds]

    if len(root) != num_levels:
        print("Top-level OME-XML elements other than Image are not supported.")
    if size_z != 1 or size_t != 1:
        print("Z-stacks and multiple timepoints are not supported.")
        sys.exit(1)
    if size_c * num_levels != len(tiff.ifds):
        print("TIFF page count does not match OME-XML Image elements.")
        sys.exit(1)
    if any(dims != (size_x, size_y) for dims in page_dims[:size_c]):
        print(f"TIFF does not begin with SizeC={size_c} full-size pages.")
        sys.exit(1)
    for level in range(1, num_levels):
        level_dims = page_dims[level * size_c : (level + 1) * size_c]
        if len(set(level_dims)) != 1:
            print(
                f"Pyramid level {level + 1} out of {num_levels} has inconsistent"
                f" sizes:\n{level_dims}"
            )
            sys.exit(1)
    if args.channel_names and len(args.channel_names) != size_c:
        print(
            f"Wrong number of channel names -- image has {size_c} channels but"
            f" {len(args.channel_names)} names were specified:"
        )
        for i, n in enumerate(args.channel_names, 1):
            print(f"{i:4}: {n}")
        sys.exit(1)

    print("Input image summary")
    print("===================")
    print(f"Dimensions: {size_x} x {size_y}")
    print(f"Number of channels: {size_c}")
    print(f"Pyramid sub-resolutions ({num_levels - 1} total):")
    for dim_x, dim_y in page_dims[size_c::size_c]:
        print(f"    {dim_x} x {dim_y}")
    software = tiff.ifds[0].tags.get_value(305, "<not set>")
    print(f"Software: {software}")
    print()

    print("Updating OME-XML metadata...")
    # We already verified there is nothing but Image elements under the root.
    for other_image in root[1:]:
        root.remove(other_image)
    for tiffdata in pixels.findall("ome:TiffData", xml_ns):
        pixels.remove(tiffdata)
    new_tiffdata = xml.etree.ElementTree.Element(
        f"{{{xml_ns['ome']}}}TiffData",
        attrib={"IFD": "0", "PlaneCount": str(size_c)},
    )
    # A valid OME-XML Pixels begins with size_c Channels; then comes TiffData.
    pixels.insert(size_c, new_tiffdata)

    if args.channel_names:
        print("Renaming channels...")
        channels = pixels.findall("ome:Channel", xml_ns)
        for channel, name in zip(channels, args.channel_names):
            channel.attrib["Name"] = name

    fix_attrib_namespace(root)
    # ElementTree.tostring would have been simpler but it only supports
    # xml_declaration and default_namespace starting with Python 3.8.
    xml_file = io.BytesIO()
    tree = xml.etree.ElementTree.ElementTree(root)
    tree.write(
        xml_file,
        encoding="utf-8",
        xml_declaration=True,
        default_namespace=xml_ns["ome"],
    )
    new_omexml = xml_file.getvalue()

    print("Writing new TIFF headers...")
    stale_ranges = [ifd.offset_range for ifd in tiff.ifds]
    main_ifds = tiff.ifds[:size_c]
    channel_sub_ifds = [tiff.ifds[c + size_c : : size_c] for c in range(size_c)]
    for i, (main_ifd, sub_ifds) in enumerate(zip(main_ifds, channel_sub_ifds)):
        for ifd in sub_ifds:
            if 305 in ifd.tags:
                stale_ranges.append(ifd.tags[305].offset_range)
                del ifd.tags[305]
            ifd.tags.insert(tiff.append_tag_data(254, 3, 1))
        if i == 0:
            stale_ranges.append(main_ifd.tags[305].offset_range)
            stale_ranges.append(main_ifd.tags[270].offset_range)
            old_software = main_ifd.tags[305].value.replace("Faas", "F*a*a*s")
            new_software = f"pyramid_upgrade.py (was {old_software})"
            main_ifd.tags.insert(tiff.append_tag_data(305, 2, new_software))
            main_ifd.tags.insert(tiff.append_tag_data(270, 2, new_omexml))
        else:
            if 305 in main_ifd.tags:
                stale_ranges.append(main_ifd.tags[305].offset_range)
                del main_ifd.tags[305]
        sub_ifds[:] = tiff.append_ifd_sequence(sub_ifds)
        offsets = [ifd.offset for ifd in sub_ifds]
        main_ifd.tags.insert(tiff.append_tag_data(330, 16, offsets))
    main_ifds = tiff.append_ifd_sequence(main_ifds)
    tiff.write_first_ifd_offset(main_ifds[0].offset)

    print("Clearing old headers and tag values...")
    # We overwrite all the old IFDs and referenced data values with obvious
    # "filler" as a courtesy to anyone who might need to poke around in the TIFF
    # structure down the road. A real TIFF parser wouldn't see the stale data,
    # but a human might just scan for the first thing that looks like a run of
    # OME-XML and not realize it's been replaced with something else. The filler
    # content is the repeated string "unused " with square brackets at the
    # beginning and end of each filled IFD or data value.
    filler = b"unused "
    f_len = len(filler)
    for r in stale_ranges:
        tiff.file.seek(r.start)
        tiff.file.write(b"[")
        f_total = len(r) - 2
        for i in range(f_total // f_len):
            tiff.file.write(filler)
        tiff.file.write(b" " * (f_total % f_len))
        tiff.file.write(b"]")

    tiff.close()

    print()
    print("Success!")


if __name__ == "__main__":
    main()
