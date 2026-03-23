# pyImageProc
Python module to calibrate directory trees of image files

## Usage

The script can process FITS files from separate directories for different frame types:

```bash
python pyimageproc.py /path/to/lights \
    --output /path/to/output \
    --bias-dir /path/to/biases \
    --dark-dir /path/to/darks \
    --flat-dir /path/to/flats \
    --flatdark-dir /path/to/flatdarks \
    --temp-tol 3.0 \
    --stack-method sigma_clip_mean
```

### Command Line Options

- `root`: Directory containing light frames (positional argument)
- `--output`: Output directory (required)
- `--bias-dir`: Directory containing bias frames (optional, defaults to root)
- `--dark-dir`: Directory containing dark frames (optional, defaults to root)
- `--flat-dir`: Directory containing flat frames (optional, defaults to root)
- `--flatdark-dir`: Directory containing flat-dark frames (optional, defaults to root)
- `--temp-tol`: Temperature tolerance for calibration matching (default: 3.0°C)
- `--stack-method`: Stacking method (sigma_clip_mean, median, mean)
- `--scan-only`: Only scan and create project descriptors, don't process 
