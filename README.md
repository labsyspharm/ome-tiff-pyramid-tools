# ome-tiff-pyramid-tools

Python tools for creating and manipulating OME-TIFF multi-resolution pyramid images. 

## pyramid_assemble.py

Combine several separate TIFF files into an OME-TIFF legacy pyramid.

Example:
```
pyramid_assemble.py channel-1.tif channel-2.tif channel-3.tif pyramid.tif --pixel-size 0.65
```

### Requirements

* Python 3.6 or higher
* tifffile>=2020.9.28
* scikit-image
* zarr

## pryamid_upgrade.py

Upgrade legacy OME-TIFF pyramids to the BioFormats 6 OME-TIFF pyramid format. Also
can optionally add channel names to the OME-XML metadata. Does not read or modify
pixel data at all so it runs more or less instantaneously. Note that it modifies
the input file in place so make your own backups if you want to retain the original
version.

Example:
```
pyramid_upgrade.py pyramid.tif -n name1 name2 name3
```

### Requirements

Python 3.7 or higher, or Python 3.6 with the `dataclasses` backport installed.

### Todo

* Split out TiffSurgeon into its own package with better documentation and
  rationale.
* Include a small example input file for testing.
