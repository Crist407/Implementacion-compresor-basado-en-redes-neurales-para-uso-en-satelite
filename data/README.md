# SORTENY RAW dataset

This directory contains the RAW crops used by the reproducible examples.

## Format

- Dimensions: `8 x 512 x 512`
- Data type: unsigned 16-bit integer (`uint16`)
- Byte order: little-endian on the reference platform
- Layout: BSQ, band by band
- File size per crop: `8 * 512 * 512 * 2 = 4,194,304` bytes

The canonical 8-band order used by the C tools is:

| Band index | Sentinel-2 band |
|---:|---|
| 0 | B02 |
| 1 | B03 |
| 2 | B04 |
| 3 | B05 |
| 4 | B06 |
| 5 | B07 |
| 6 | B08 |
| 7 | B8A |

`Sentinel2A_crop_test/` contains 120 RAW crops. The root-level RAW file is a
single canonical sample used by the quickstart commands.

## Opening with Fiji/ImageJ

Use `File -> Import -> Raw...` and set:

- Image type: `16-bit Unsigned`
- Width: `512`
- Height: `512`
- Number of images: `8`
- Offset to first image: `0`
- Gap between images: `0`
- Byte order: little-endian

Fiji will open the file as an 8-slice stack. For a visible RGB inspection, use
the approximate true-color bands:

- Red: B04, slice 3
- Green: B03, slice 2
- Blue: B02, slice 1

The RAW files are not directly viewable as ordinary images because they contain
eight spectral bands and no image header.

## Generating quicklooks

A simple quicklook package can be generated from one RAW file with:

```bash
python3 src/python/utils/manual_roi_quicklook.py \
  --input data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw \
  --output-dir output/checkpoints/quicklook_example \
  --pattern empty \
  --skip-qmap
```

The output directory contains PNG files intended only for visual inspection.
Generated quicklooks are not committed to Git by default.
