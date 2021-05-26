# SWIFT Automated synoptic plotting

Automatically-generated charts of weather features such as jets,
convergence lines, troughs and waves across African domains.

The charts show key diagnostics from Numerical Weather Prediction
(NWP) models and have been developed as part of the GCRF African SWIFT
project as an aid to training and operational forecasting.

## Python environment

The Python environment requirements can be recreated from
`environment.yml` using conda:
```
conda env create -f environment.yml
conda activate swift_synoptic
```

## Data pre-processing

The data required to generate the charts is derived from NCEP/NOAA
public domain GFS data.

*Documentation in progress*

## Generating charts

The package contains a convenience script that produces automated
plots of synoptic features across African domains.  Default behaviour
is to display plots on screen; optionally, plots can be saved to a
specified output directory.  Note that this script currently assumes
that there is a `$SWIFT_GFS` environment variable which gives the
location of the preprocessed GFS data.

```
usage: chart.py [-h] [-o [OUTPUT_DIR]] domain timestamp forecast_hour [chart_type]

Plot synoptic chart

positional arguments:
  domain                Domain specified as standardised domain name (WA, EA or PA)
  timestamp             Timestamp for chart data in format "YYYYmmddHH"
  forecast_hour         Forecast hour as non-negative integer multiple of 3 (max 72)
  chart_type            Chart type (low, jets, conv or synth) (default: low)

optional arguments:
  -h, --help            show this help message and exit
  -o [OUTPUT_DIR], --output-dir [OUTPUT_DIR]
                        Path to output directory
```

Example:

```
python synoptic/chart.py WA 2020090300 3 jets
```

## Funding

This work was supported by UK Research and Innovation as part of the
Global Challenges Research Fund, African SWIFT programme, grant number
NE/P021077/1.
