import os
import requests
from datetime import datetime

# List of UUID and filename pairs
uuid_filename_pairs = [
    ("01c15a65-bf19-4af7-8e69-4eb7ec397eb7","noaa_tmmc_prls"),
    ("b7114ae2-b2cb-40fe-af33-820db2db7755","noaa_currituck_hampton_inn"),
    ("a8b8920e-40d5-4a45-83d5-96b367d51e5f","noaa_currituck_sailfish"),
    #("006516c2-c796-414f-9528-6828ae10af56","noaa_nwlon_charleston"),
    ("716d8417-d0c3-4ed9-b48c-9094be8fe205","uncw_cms_dock_north"),
    ("c28e3eb0-24a0-42a8-b9ab-ed74a9281e4b","uncw_cms_dock_south"),
    ("4e15fae5-f8d0-4f85-ae55-930be11d2cad","uncw_masonboro_inlet"),
    #("5900cae6-36a6-4f39-adf9-5660ff5e97e5","ucsc_walton_lighthouse"),
    ("70714572-3436-45ef-b4f9-1df59e5c293f","nerrs_northinlet"),
    #("560fc1e1-a7f6-47d4-8b2d-8bf356e20c96","nanoos_newport_shilo"),
    #("199994fb-ce80-459d-b334-3b486dac0607","nanoos_bastendorff"),
    ("f8d45540-07ec-45bd-bb63-33c3bee0027c","maracoos_carovabeach"),
    ("3d2695dc-46f1-4f4a-bbd4-b731721650a2","maracoos_oceancity"),
    ("f8d45540-07ec-45bd-bb63-33c3bee0027c","maracoos_vabeach_hamptonos"),
    ("e158b8b5-e70f-4ac8-ba98-2ebdb395db4e","nerrs_folly6thavenue"),
    # Add more pairs as needed
]

# Authorization Token
token = "xxx" #use your WebCOOS API token here

# Create 'jsonl' directory if it does not exist
os.makedirs("jsonl", exist_ok=True)

# Base URL for the API
base_url = "https://app.webcoos.org/webcoos/api/v1/services/{}/elements/latest/"

# Generate a uniform execution timestamp for this batch run
timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

# Loop through the list and make requests
for uuid, filename in uuid_filename_pairs:
    # Create the URL with the current UUID
    url = base_url.format(uuid)

    # Make the GET request
    headers = {
        "accept": "application/json",
        "Authorization": f"Token {token}"
    }
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a unique .jsonl file matching your naming conventions
        jsonl_filename = f"jsonl/{filename}_{timestamp_suffix}.jsonl"
        with open(jsonl_filename, "w") as file:
            file.write(response.text)
        print(f"Successfully wrote to {jsonl_filename}")
    else:
        print(f"Failed to fetch data for UUID: {uuid}, Status Code: {response.status_code}")
