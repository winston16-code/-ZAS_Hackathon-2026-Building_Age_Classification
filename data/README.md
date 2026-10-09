# Data (not included)

The Landsat time series and building-age labels were provided by the organisers of the
ZAS 2026 Data Science Regional Hackathon and are not redistributed in this repository.

To run the notebooks, place the two files here:

```
data/
├── madrid_train.parquet      # 76,263 Madrid pixels, labelled (weighted_mean_year)
└── amsterdam_data.parquet    # 25,992 Amsterdam pixels
```

Each row is one pixel-year (1984–2025) with up to 3 Landsat photos: 6 surface-reflectance bands
(Blue, Green, Red, NIR, SWIR1, SWIR2), QA_PIXEL / QA_RADSAT flags, day of year, scene id and a
validity flag, plus pixel coordinates (`px_key`, `py_key`), coverage and `weighted_mean_year`.

Intermediate folders created by the notebooks (`clean/`, `features/`, `submission/`) also land here and are git-ignored.
