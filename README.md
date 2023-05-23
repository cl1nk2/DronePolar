# DronePolar - post-processing raw polarizational images
This is a tool for the ImagingSource DYK 33GX250 Polarsens industrial camera, processing the raw images captured from it. The use-case is such camera attached to a drone, capturing images, and post-processing them with this tool.

The output will be a `.svg` file.

### How to use (Linux)
1. Install the dependencies:
```bash
# Debian-based:
sudo apt install git python3 python3-pip
# Red Hat-based:
sudo dnf install git python3 python3-pip
# Arch-based:
sudo pacman -S git python3 python3-pip
```
2. Navigate to the directory you would like the tool to be installed and execute the following command:
```bash
git clone https://github.com/cl1nk2/DronePolar.git
```
3. Change into the directory of `DronePolar`.
```bash
cd DronePolar/
```
4. Install pip dependencies.
```bash
pip install -r requirements.txt
```
5. Run the Python script `polar.py` as such:
```bash
python3 polar.py [path_of_the_image_file]
```

### Other tools included
- SolarCalculator
- FlightInfo
- CalculateWaterSurface